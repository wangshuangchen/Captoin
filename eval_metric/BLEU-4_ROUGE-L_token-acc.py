import json
import numpy as np
from collections import Counter
import re

# def tokenize(text):
#     """简单的分词函数"""
#     return re.findall(r'\w+', text.lower())

def tokenize(text):
    """优化分词函数：保留连字符短语（如high-voltage），不拆分"""
    # 匹配规则：包含字母、数字、连字符（-）的连续字符视为一个Token
    return re.findall(r'[a-zA-Z0-9\-]+', text.lower())

def calculate_bleu(candidate, reference, n=4):
    """计算BLEU-n分数"""
    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)
    
    if len(candidate_tokens) == 0 or len(reference_tokens) == 0:
        return 0.0
    
    # 计算n-gram精度
    precision_scores = []
    for i in range(1, n + 1):
        candidate_ngrams = [tuple(candidate_tokens[j:j + i]) for j in range(len(candidate_tokens) - i + 1)]
        reference_ngrams = [tuple(reference_tokens[j:j + i]) for j in range(len(reference_tokens) - i + 1)]
        
        if not candidate_ngrams:
            precision_scores.append(0.0)
            continue
            
        candidate_ngram_counts = Counter(candidate_ngrams)
        reference_ngram_counts = Counter(reference_ngrams)
        
        matches = 0
        for ngram in candidate_ngram_counts:
            matches += min(candidate_ngram_counts[ngram], reference_ngram_counts.get(ngram, 0))
        
        precision_scores.append(matches / len(candidate_ngrams))
    
    # 几何平均精度
    if min(precision_scores) == 0:
        return 0.0
    
    geo_mean = np.exp(np.mean(np.log(precision_scores)))
    
    #  brevity penalty
    bp = 1.0 if len(candidate_tokens) > len(reference_tokens) else np.exp(1 - len(reference_tokens) / len(candidate_tokens))
    
    return bp * geo_mean

def calculate_rouge_l(candidate, reference):
    """计算ROUGE-L分数（基于最长公共子序列）"""
    def lcs_length(x, y):
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)
    
    if len(candidate_tokens) == 0 or len(reference_tokens) == 0:
        return 0.0
    
    lcs = lcs_length(candidate_tokens, reference_tokens)
    
    # ROUGE-L F1分数
    precision = lcs / len(candidate_tokens) if len(candidate_tokens) > 0 else 0
    recall = lcs / len(reference_tokens) if len(reference_tokens) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

# def calculate_token_acc(candidate, reference):
#     """计算token级别的准确率"""
#     candidate_tokens = tokenize(candidate)
#     reference_tokens = tokenize(reference)
    
#     if len(candidate_tokens) == 0:
#         return 0.0
    
#     # 找到匹配的token数量
#     min_len = min(len(candidate_tokens), len(reference_tokens))
#     matches = sum(1 for i in range(min_len) if candidate_tokens[i] == reference_tokens[i])
    
#     return matches / len(candidate_tokens)
def calculate_token_acc(candidate, reference):
    """修正后的Token-Acc计算：统计预测句中存在于参考句的Token数量（不考虑位置）"""
    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)
    
    if len(candidate_tokens) == 0:
        return 0.0
    
    # 处理参考句的Token计数（避免重复Token多算匹配数）
    reference_token_counts = Counter(reference_tokens)
    matches = 0
    
    # 遍历预测句的每个Token，检查是否在参考句中且有剩余计数
    for token in candidate_tokens:
        if reference_token_counts.get(token, 0) > 0:
            matches += 1
            reference_token_counts[token] -= 1  # 减少计数，避免重复Token重复匹配
    
    return matches / len(candidate_tokens)

def evaluate_jsonl_file(file_path):
    """评估JSONL文件中的所有样本"""
    results = {
        'bleu4_scores': [],
        'rouge_l_scores': [], 
        'token_acc_scores': [],
        'samples': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # 获取预测值和参考答案
                    candidate = data.get('output', '')
                    reference = data.get('answer', '')
                    
                    if not candidate or not reference:
                        print(f"警告: 第{line_num}行缺少output或answer字段")
                        continue
                    
                    # 计算各项指标
                    bleu4 = calculate_bleu(candidate, reference)
                    rouge_l = calculate_rouge_l(candidate, reference)
                    token_acc = calculate_token_acc(candidate, reference)
                    
                    # 保存结果
                    results['bleu4_scores'].append(bleu4)
                    results['rouge_l_scores'].append(rouge_l)
                    results['token_acc_scores'].append(token_acc)
                    results['samples'].append({
                        'line_num': line_num,
                        'candidate': candidate[:100] + '...' if len(candidate) > 100 else candidate,
                        'reference': reference[:100] + '...' if len(reference) > 100 else reference,
                        'bleu4': bleu4,
                        'rouge_l': rouge_l,
                        'token_acc': token_acc
                    })
                    
                    print(f"第{line_num}行评估完成: BLEU={bleu4:.4f}, ROUGE-L={rouge_l:.4f}, Token-Acc={token_acc:.4f}")
                    
                except json.JSONDecodeError as e:
                    print(f"错误: 第{line_num}行JSON解析失败: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        return None
    except Exception as e:
        print(f"错误: 读取文件时发生异常: {e}")
        return None
    
    # 计算平均分数
    if results['bleu4_scores']:
        results['avg_bleu4'] = np.mean(results['bleu4_scores'])
        results['avg_rouge_l'] = np.mean(results['rouge_l_scores'])
        results['avg_token_acc'] = np.mean(results['token_acc_scores'])
    else:
        results['avg_bleu4'] = 0.0
        results['avg_rouge_l'] = 0.0
        results['avg_token_acc'] = 0.0
    
    return results

def print_results(results):
    """打印评估结果"""
    if not results:
        print("没有可用的评估结果")
        return
    
    print("\n" + "="*80)
    print("评估结果汇总")
    print("="*80)
    print(f"总样本数: {len(results['bleu4_scores'])}")
    print(f"平均 BLEU: {results['avg_bleu4']:.4f}")
    print(f"平均 ROUGE-L: {results['avg_rouge_l']:.4f}")
    print(f"平均 Token-Acc: {results['avg_token_acc']:.4f}")
    print("="*80)
    
    print("\n前5个样本的详细结果:")
    print("-" * 80)
    for i, sample in enumerate(results['samples'][:5]):
        print(f"样本 {i+1} (第{sample['line_num']}行):")
        print(f"  预测: {sample['candidate']}")
        print(f"  参考: {sample['reference']}")
        print(f"  BLEU: {sample['bleu4']:.4f}")
        print(f"  ROUGE-L: {sample['rouge_l']:.4f}")
        print(f"  Token-Acc: {sample['token_acc']:.4f}")
        print()

# 主程序 - 直接运行
if __name__ == "__main__":
    # 在这里设置你的JSONL文件路径
    file_path = "/home3/wangshuangchen/LLaMA-Factory-main/dataset/match-6-class/cross-fence/InternVL3_5-2B-lora-250.jsonl"  # 请替换为你的实际文件路径
    
    print("开始评估JSONL文件...")
    print(f"文件路径: {file_path}")
    print("-" * 80)
    
    # 执行评估
    results = evaluate_jsonl_file(file_path)
    
    # 打印结果
    if results:
        print_results(results)
        
        # 可选：保存详细结果到文件
        output_file = "/home3/wangshuangchen/LLaMA-Factory-main/caption_results/BELU_ROUGE_token/InternVL3_5-2B-lora-250.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存到: {output_file}")
    else:
        print("评估失败，请检查文件路径和格式")