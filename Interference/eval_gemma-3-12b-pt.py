import re
import traceback
from peft import PeftModel
import torch
import os
import json
from tqdm import tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoTokenizer, Gemma3ForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
torch._dynamo.disable()
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

# Load half-precision model
base_model_path = "/root/autodl-tmp/gemma-3-12b-pt"
model = Gemma3ForConditionalGeneration.from_pretrained(
    base_model_path,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(base_model_path, use_fast=True)
lora_model_path = "/root/LLaMA-Factory-main/saves/gemma-3-12b-pt-12346-6000/lora/sft"

# lora微调时打开
# model = PeftModel.from_pretrained(model, lora_model_path)

def extract_mcq_only(question: str) -> str:
    # Match complete multiple choice question (based on (A)...(E)...)
    match = re.search(r"(Which of the following.*?\(E\)[^)]+?\.)", question)
    if match:
        return match.group(1).strip()
    return question

def load_data(input_file, image_dir):
    """Read input JSON, return image paths, questions and corresponding metadata lists."""
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                line = json.loads(line)
                data.append(line)
    print(f"Read {len(data)} records")
    
    image_paths = []
    questions = []
    metadata = []

    for item in data:
        image_path = os.path.join(image_dir, item["image_path"])
        if not os.path.exists(image_path):
            print(f"Image file does not exist: {image_path}")
            continue
        image_paths.append(image_path)
        raw_question = item["conversation"]["Question"]
        questions.append(extract_mcq_only(raw_question))
        metadata.append({
            "image_path": item["image_path"],
            "coordinates": item["coordinates"],
            "type": item["type"],
            "answer": item["conversation"]["Answer"],
            # Add question field for later reference
            "question": extract_mcq_only(raw_question)
        })
    
    return image_paths, questions, metadata


def batch_predict(image_path, question, model, processor):
    """Predict for an image and question, return model's raw output."""
    results = []
    system_message = "You are an AI model for violation dectection. The format of the violation region coordinates should be <boxes>[[(?),(?)],...]</boxes>, that is, the violation area may be multiple or even one. You must limit your analysis process into one paragraph."
    # Load image
    image = Image.open(image_path).convert("RGB")
    # Construct messages - commented code shows alternative approach
    # messages = [
    #     [
    #         {
    #             "role": "system",
    #             "content": [
    #                 {"type": "text", "text": "You are an AI model for violation dectection. The format of the violation region coordinates should be <boxes>[[(?),(?)],...]</boxes>, that is, the violation area may be multiple or even one. You must limit your analysis process into one paragraph."}
    #             ]
    #         },
    #         {
    #             "role": "user", "content": [
    #                 {"type": "image", "image": image},
    #                 {"type": "text",  "text": question+" Please present your answer starting with The answer is (X = option).If the answer is \"no violation\",please provide a brief analysis;if not,immediately specify the anomaly violation coordinates [[(?, ?), (?, ?)],...] and immediately followed by your analysis."}
    #             ]
    #         }
    #     ]
    #     for image, question in zip(images, batch_questions)
    # ]
    user_message = question
    noncot_message = "Please present your answer with: The answer is (X). The violation region coordinates are [[(?, ?), (?, ?)],......]."
    cot_message = "Please present your answer starting with: The answer is (X). If the answer is \"no defect\", please provide a brief analysis; if not, immediately specify the violation region coordinates [[(?, ?), (?, ?)],......] and immediately followed by your analysis."
    text = f"<start_of_image>\n{user_message}  Please provide your description and prediction. If there is no danger in the next step, only output [safe]; if not, immediately specify the coordinates of the violation area [(?, ?),(?, ?)] and immediately output [unsafe]. And output indicates which of the following six scenarios this image belongs to: <welding or cutting operations>, <work at heights>, <staying under the crane>, <crossing the fence>, <electrical inspection pole operation> or <others>."
    try:
        # Construct inputs using processor
        inputs = processor(
            text=text,
            images=image,
            return_tensors="pt"
        ).to(model.device, torch.bfloat16)
        
        # Generate model output
        generate_ids = model.generate(
            **inputs, 
            max_new_tokens=1096,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=processor.tokenizer.pad_token_id
        )
        
        # Remove input part, keep only newly generated tokens
        generated_ids_trimmed = generate_ids[:, inputs.input_ids.shape[1]:]
        responses = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
    except Exception as e:
        print(f"Error during processing: {e}")
        traceback.print_exc()
        responses = ["Error"]
    
    return responses[0]


def load_processed_image_paths(output_file):
    """Read existing output file, return set of processed image_paths.
       Uses JSON Lines format, one JSON object per line."""
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        processed.add(record["image_path"])
                    except Exception as e:
                        print(f"Failed to read a line: {e}")
    return processed


def main(batch_size=16):  # 默认批次大小为1，可根据需求调整
    input_file = "/root/LLaMA-Factory-main/data/test-1642.jsonl"
    output_file = "/root/LLaMA-Factory-main/results-1642/gemma-3-12b-lora-test.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    image_dir = "/root/images/test-images-1642"

    image_paths, questions, metadata = load_data(input_file, image_dir)
    processed_image_paths = load_processed_image_paths(output_file)

    with open(output_file, "a", encoding="utf-8") as out_f:
        # 按batch_size分批处理数据
        for batch_start in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, len(image_paths))
            batch_image_paths = image_paths[batch_start:batch_end]
            batch_questions = questions[batch_start:batch_end]
            batch_metadata = metadata[batch_start:batch_end]

            for i in range(len(batch_image_paths)):
                image_path = batch_image_paths[i]
                question = batch_questions[i]
                meta = batch_metadata[i]

                if meta["image_path"] in processed_image_paths:
                    continue

                response = batch_predict(image_path, question, model, processor)
                result_item = {
                    "image_path": meta["image_path"],
                    "coordinates": meta["coordinates"],
                    "type": meta["type"],
                    "question": meta["question"],
                    "answer": meta["answer"],
                    "output": response
                }
                out_f.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                out_f.flush()
                processed_image_paths.add(meta["image_path"])
                
if __name__ == "__main__":
    main()