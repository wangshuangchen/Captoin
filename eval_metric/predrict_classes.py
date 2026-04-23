# coding=utf-8
import json
import re

def evaluate_predictions(jsonl_file_path, output_file_path):
    # 定义场景类别列表 categories
    CATEGORIES = [
        "welding or cutting operations",
        "work at heights", 
        "staying under the crane",
        "crossing the fence",
        "electrical inspection pole operation",
        "others"
    ]
    
    # 编译正则表达式
    safety_pattern = re.compile(r'\[(safe|unsafe)\]', re.IGNORECASE)
    # 改进的类别模式，匹配多种格式：<category>、*category*、**category**等
    category_patterns = [
        re.compile(r'<([^>]+)>'),  # <category>
        re.compile(r'\*+([^*]+)\*+'),  # *category* 或 **category**
        re.compile(r'【([^】]+)】'),  # 【category】
        re.compile(r'「([^」]+)」'),  # 「category」
        re.compile(r'《([^》]+)》'),  # 《category》
    ]
    
    total_samples = 0
    safety_correct = 0
    category_correct = 0
    results = []
    
    def extract_category(text):
        """从文本中提取类别，支持多种格式"""
        if not text:
            return None
            
        # 首先尝试精确匹配预定义的类别
        for category in CATEGORIES:
            if category.lower() in text.lower():
                return f"<{category}>"
        
        # 然后尝试各种模式匹配
        for pattern in category_patterns:
            match = pattern.search(text)
            if match:
                extracted = match.group(1).strip()
                # 对提取的内容进行标准化
                extracted_lower = extracted.lower()
                for category in CATEGORIES:
                    if category.lower() in extracted_lower or extracted_lower in category.lower():
                        return f"<{category}>"
                # 如果没有精确匹配，但提取到了内容，也返回
                return f"<{extracted}>"
        
        return None
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            total_samples += 1
            sample_result = {
                'line_number': line_num,
                'output': None,
                'answer': None,
                'predicted_safety': None,
                'actual_safety': None,
                'safety_correct': False,
                'predicted_category': None,
                'actual_category': None,
                'category_correct': False,
                'error': None
            }
            
            try:
                data = json.loads(line)
                output = data.get('output', '')
                answer = data.get('answer', '')
                
                sample_result['output'] = output
                sample_result['answer'] = answer
                
                # 提取安全状态
                pred_safety_match = safety_pattern.search(output)
                actual_safety_match = safety_pattern.search(answer)
                
                pred_safety = pred_safety_match.group(1).lower() if pred_safety_match else None
                actual_safety = actual_safety_match.group(1).lower() if actual_safety_match else None
                
                sample_result['predicted_safety'] = pred_safety
                sample_result['actual_safety'] = actual_safety
                
                # 判断安全状态是否正确
                if pred_safety is not None and actual_safety is not None:
                    safety_correct_flag = (pred_safety == actual_safety)
                    sample_result['safety_correct'] = safety_correct_flag
                    if safety_correct_flag:
                        safety_correct += 1
                
                # 提取场景类别 - 使用改进的方法
                pred_category = extract_category(output)
                actual_category = extract_category(answer)
                
                sample_result['predicted_category'] = pred_category
                sample_result['actual_category'] = actual_category
                
                # 判断场景类别是否正确
                if pred_category is not None and actual_category is not None:
                    # 标准化比较（去掉<>进行比较）
                    pred_clean = pred_category.strip('<>').lower()
                    actual_clean = actual_category.strip('<>').lower()
                    
                    # 模糊匹配：检查是否有重叠的关键词
                    category_correct_flag = False
                    
                    # 精确匹配
                    if pred_clean == actual_clean:
                        category_correct_flag = True
                    else:
                        # 模糊匹配：检查关键词重叠
                        pred_words = set(pred_clean.split())
                        actual_words = set(actual_clean.split())
                        if pred_words & actual_words:  # 有共同词汇
                            category_correct_flag = True
                        else:
                            # 检查是否是同义词或相近类别
                            similar_categories = {
                                "welding": ["welding", "cutting", "weld"],
                                "heights": ["heights", "high", "elevated"],
                                "crane": ["crane", "lifting", "hoist"],
                                "fence": ["fence", "barrier", "enclosure"],
                                "electricity": ["electricity", "electrical", "power"],
                                "others": ["others", "other", "miscellaneous"]
                            }
                            
                            for key, synonyms in similar_categories.items():
                                if any(word in pred_clean for word in synonyms) and any(word in actual_clean for word in synonyms):
                                    category_correct_flag = True
                                    break
                    
                    sample_result['category_correct'] = category_correct_flag
                    if category_correct_flag:
                        category_correct += 1
                
            except Exception as e:
                sample_result['error'] = str(e)
            
            results.append(sample_result)
    
    # 计算正确率
    safety_accuracy = safety_correct / total_samples if total_samples > 0 else 0.0
    category_accuracy = category_correct / total_samples if total_samples > 0 else 0.0
    
    # 打印结果
    print("total: {}".format(total_samples))
    print("安全状态预测正确数: {}".format(safety_correct))
    print("安全状态预测正确率: {:.4f}".format(safety_accuracy))
    print("场景类别预测正确数: {}".format(category_correct))
    print("场景类别预测正确率: {:.4f}".format(category_accuracy))
    
    # 保存详细结果到文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
    
    print("\n详细结果已保存到: {}".format(output_file_path))

# 主程序
if __name__ == "__main__":
    # 输入和输出文件路径
    input_jsonl_path = "/home3/wangshuangchen/LLaMA-Factory-main/dataset/match-6-class/work-at-height/Llama-3.2-11B-lora-250.jsonl"
    output_jsonl_path = "/home3/wangshuangchen/LLaMA-Factory-main/caption_results/safe_classes/Llama-3.2-11B-lora-250.txt"
    
    # 执行评估
    evaluate_predictions(input_jsonl_path, output_jsonl_path)