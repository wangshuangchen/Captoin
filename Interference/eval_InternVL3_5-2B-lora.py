import os
import json
import re
import jsonlines
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

torch_device = "auto"

model_checkpoint = "/home3/wangshuangchen/model/InternVL3_5-2B-HF"
processor = AutoProcessor.from_pretrained(model_checkpoint)
model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=torch_device, torch_dtype=torch.bfloat16)

# lora微调时打开
model = PeftModel.from_pretrained(model, "/home3/wangshuangchen/LLaMA-Factory-main/saves/InternVL3_5-2B-HF-Multimodal-dianwang_cot/lora/sft")

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

def extract_mcq_only(question: str) -> str:
    # Match complete multiple choice question (based on (A)...(E)...)
    match = re.search(r"(Which of the following.*?\(E\)[^)]+?\.)", question)
    if match:
        return match.group(1).strip()
    return question

def main():
    input_file = "/home3/wangshuangchen/LLaMA-Factory-main/data/test-1642.jsonl"    # Input dataset for prediction
    output_file = "/home3/wangshuangchen/LLaMA-Factory-main/results_lora/InternVL3_5-2B-test-1642.jsonl"    # Output file in JSON Lines format
    
    with jsonlines.open(input_file) as reader:
        data = list(reader)
    # Load processed records (determine if already predicted by image_path)
    processed_image_paths = load_processed_image_paths(output_file)
    
    # Iterate through dataset, process each sample and write to output file, skip if already exists
    with open(output_file, "a", encoding="utf-8") as out_f:
        for item in data:
            image_path = item["image_path"]
            if image_path in processed_image_paths:
                print(f"Skipping already processed sample: {image_path}")
                continue
            
            raw_question = item["conversation"]["Question"]
            question=extract_mcq_only(raw_question)
            messages = [
                    {
                        "role": "system",
                        "content": [
                        {   
                            "type":"text",
                            "text":"You are an AI model for violation detection. The format of the anomaly region coordinates should be <boxes>[[(?,?),(?,?)],...] </boxes>, that is, the violation area may be multiple or even one."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image", 
                            "url": image_path
                        },
                        {
                            "type": "text", 
                            "text": "<image>\n"+ question +"  Please provide your description and prediction. If there is no danger in the next step, only output [safe]; if not, immediately specify the coordinates of the violation area [(?, ?),(?, ?)] and immediately output [unsafe]. And output indicates which of the following six scenarios this image belongs to: <welding or cutting operations>, <work at heights>, <staying under the crane>, <crossing the fence>, <electrical inspection pole operation> or <others>."
                        },
                    ],
                }
            ]

            inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

            generate_ids = model.generate(**inputs, max_new_tokens=1024)
            decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            
            
            result_item = {
                "image_path": image_path,
                "coordinates": item["coordinates"],
                "type": item["type"],
                "question": question,
                "answer": item["conversation"]["Answer"],   # Ground truth answer
                "output": decoded_output    # Model prediction
            }
            
            # Write current result to output file (JSON Lines format)
            out_f.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            out_f.flush()  # Ensure data is written to disk
            processed_image_paths.add(image_path)
            print(f"Processing completed: {image_path}")

if __name__ == "__main__":
    main()
