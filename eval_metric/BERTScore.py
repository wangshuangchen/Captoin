import json
import numpy as np
from bert_score import BERTScorer
import sys
import os
import torch  # 新增：用于显存管理和分批处理

class BERTScoreCalculator:
    def __init__(self, model_type="microsoft/deberta-xlarge-mnli", lang="en", num_layers=None):
        """初始化BERTScore计算器"""
        print("正在初始化BERTScore计算器...")
        
        # 关键优化1：指定FP16精度，减少50%显存占用
        torch_dtype = torch.float16
        
        self.scorer = BERTScorer(
            model_type=model_type,
            lang=lang,
            num_layers=num_layers,
            idf=False,
            device="cuda",
            batch_size=20,  # 关键优化2：批量可适当提高（FP16+分批下，20是安全值）
            nthreads=4,
            all_layers=False,
            rescale_with_baseline=False,
            # torch_dtype=torch_dtype,  # 启用FP16
            # device_map="auto"
        )
        print(f"成功加载BERTScore模型: {model_type} (FP16精度)")
        print(f"使用语言: {lang}")

    def calculate_bertscore(self, candidates, references):
        """计算BERTScore（支持分批处理）"""
        if self.scorer is None:
            print("BERTScore模型未加载")
            return None
        
        if len(candidates) != len(references):
            print(f"错误: 候选文本数量({len(candidates)})和参考文本数量({len(references)})不匹配")
            return None
        
        try:
            # 计算BERTScore（模型内部按batch_size分批推理）
            P, R, F1 = self.scorer.score(candidates, references)
            
            P = P.numpy()
            R = R.numpy()
            F1 = F1.numpy()
            
            return {
                'precision': P,
                'recall': R,
                'f1': F1
            }
            
        except Exception as e:
            print(f"计算BERTScore时出错: {e}")
            return None

    def calculate_bertscore_single(self, candidate, reference):
        """计算单个文本对的BERTScore"""
        return self.calculate_bertscore([candidate], [reference])

def calculate_bertscore_for_jsonl(jsonl_file_path, output_file, model_type, batch_size=20):
    """计算JSONL文件中所有样本的BERTScore（分批处理版本）"""
    
    # 初始化BERTScore计算器
    bert_calculator = BERTScoreCalculator(model_type=model_type)
    
    if bert_calculator.scorer is None:
        print("无法初始化BERTScore计算器，退出")
        return None
    
    results = {
        'precision_scores': [],
        'recall_scores': [],
        'f1_scores': [],
        'samples': [],
        'total_count': 0,
        'success_count': 0,
        'failed_count': 0
    }
    
    # 关键修改1：不一次性收集所有文本，而是按批次处理（减少内存/显存占用）
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            # 先读取所有有效样本（仅存储必要信息，不存储完整文本）
            valid_samples = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                results['total_count'] += 1
                
                try:
                    data = json.loads(line)
                    candidate = data.get('output', '').strip()[:200]  # 关键优化3：截断长文本（200词足够）
                    reference = data.get('answer', '').strip()[:200]
                    
                    if not candidate or not reference:
                        print(f"警告: 第{line_num}行缺少output/answer字段，跳过")
                        results['failed_count'] += 1
                        continue
                    
                    valid_samples.append({
                        'line_num': line_num,
                        'candidate': candidate,
                        'reference': reference,
                        'candidate_short': candidate[:100] + '...' if len(candidate) > 100 else candidate,
                        'reference_short': reference[:100] + '...' if len(reference) > 100 else reference
                    })
                    results['success_count'] += 1
                    
                    if line_num % 500 == 0:
                        print(f"已读取 {line_num} 行，有效样本 {len(valid_samples)} 个...")
                    
                except json.JSONDecodeError as e:
                    print(f"错误: 第{line_num}行JSON解析失败: {e}")
                    results['failed_count'] += 1
                    continue
                except Exception as e:
                    print(f"错误: 第{line_num}行处理失败: {e}")
                    results['failed_count'] += 1
                    continue
                    
    except FileNotFoundError:
        print(f"错误: 文件 {jsonl_file_path} 不存在")
        return None
    except Exception as e:
        print(f"错误: 读取文件时发生异常: {e}")
        return None
    
    if not valid_samples:
        print("没有有效的文本对可以计算BERTScore")
        return results
    
    print(f"\n开始计算BERTScore，共 {len(valid_samples)} 个有效样本，分批大小: {batch_size}")
    
    # 关键修改2：分批处理有效样本
    total_batches = (len(valid_samples) + batch_size - 1) // batch_size  # 总批次数
    for batch_idx in range(total_batches):
        # 计算当前批次的样本索引
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(valid_samples))
        batch_samples = valid_samples[start_idx:end_idx]
        
        print(f"\n处理批次 {batch_idx + 1}/{total_batches}（样本 {start_idx + 1}-{end_idx}）...")
        
        # 提取当前批次的文本对
        batch_candidates = [s['candidate'] for s in batch_samples]
        batch_references = [s['reference'] for s in batch_samples]
        
        # 清理显存碎片（关键！）
        torch.cuda.empty_cache()
        
        # 计算当前批次的BERTScore
        batch_results = bert_calculator.calculate_bertscore(batch_candidates, batch_references)
        
        if batch_results is None:
            print(f"批次 {batch_idx + 1} 计算失败，跳过")
            results['failed_count'] += len(batch_samples)
            results['success_count'] -= len(batch_samples)
            continue
        
        # 合并当前批次结果
        results['precision_scores'].extend(batch_results['precision'].tolist())
        results['recall_scores'].extend(batch_results['recall'].tolist())
        results['f1_scores'].extend(batch_results['f1'].tolist())
        
        # 构建当前批次的样本详细信息
        for i, (sample, p, r, f1) in enumerate(zip(
            batch_samples, batch_results['precision'], batch_results['recall'], batch_results['f1']
        )):
            results['samples'].append({
                'line_num': sample['line_num'],
                'candidate': sample['candidate_short'],
                'reference': sample['reference_short'],
                'precision': float(p),
                'recall': float(r),
                'f1': float(f1)
            })
        
        # 再次清理显存
        torch.cuda.empty_cache()
    
    # 计算整体统计信息
    if results['f1_scores']:
        results['avg_precision'] = np.mean(results['precision_scores'])
        results['avg_recall'] = np.mean(results['recall_scores'])
        results['avg_f1'] = np.mean(results['f1_scores'])
        results['max_f1'] = np.max(results['f1_scores'])
        results['min_f1'] = np.min(results['f1_scores'])
        results['std_f1'] = np.std(results['f1_scores'])
    else:
        results['avg_precision'] = 0.0
        results['avg_recall'] = 0.0
        results['avg_f1'] = 0.0
        results['max_f1'] = 0.0
        results['min_f1'] = 0.0
        results['std_f1'] = 0.0
    
    return results

# 其余函数（print_bertscore_results、save_bertscore_results、analyze_bertscore_distribution）不变...
def print_bertscore_results(results):
    if not results or not results['f1_scores']:
        print("没有可用的BERTScore结果")
        return
    
    print("\n" + "="*80)
    print("BERTScore评估结果汇总")
    print("="*80)
    print(f"总样本数: {results['total_count']}")
    print(f"成功计算: {results['success_count']}")
    print(f"失败数量: {results['failed_count']}")
    print(f"平均 Precision: {results['avg_precision']:.4f}")
    print(f"平均 Recall:    {results['avg_recall']:.4f}")
    print(f"平均 F1:        {results['avg_f1']:.4f}")
    print(f"最高 F1:        {results['max_f1']:.4f}")
    print(f"最低 F1:        {results['min_f1']:.4f}")
    print(f"F1 标准差:      {results['std_f1']:.4f}")
    print("="*80)
    
    print("\n前10个样本的BERTScore结果:")
    print("-" * 80)
    for i, sample in enumerate(results['samples'][:10]):
        print(f"样本 {i+1} (第{sample['line_num']}行):")
        print(f"  预测文本: {sample['candidate']}")
        print(f"  参考文本: {sample['reference']}")
        print(f"  Precision: {sample['precision']:.4f}")
        print(f"  Recall:    {sample['recall']:.4f}")
        print(f"  F1:        {sample['f1']:.4f}")
        print()

def save_bertscore_results(results, output_file="bertscore_results.json"):
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            serializable_results = {
                'statistics': {
                    'total_count': results['total_count'],
                    'success_count': results['success_count'],
                    'failed_count': results['failed_count'],
                    'avg_precision': float(results['avg_precision']),
                    'avg_recall': float(results['avg_recall']),
                    'avg_f1': float(results['avg_f1']),
                    'max_f1': float(results['max_f1']),
                    'min_f1': float(results['min_f1']),
                    'std_f1': float(results['std_f1'])
                },
                'samples': results['samples'],
                'all_precision_scores': results['precision_scores'],
                'all_recall_scores': results['recall_scores'],
                'all_f1_scores': results['f1_scores']
            }
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        print(f"BERTScore结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存结果失败: {e}")

def analyze_bertscore_distribution(results):
    if not results or not results['f1_scores']:
        return
    
    f1_scores = results['f1_scores']
    ranges = [
        (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
        (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)
    ]
    
    distribution = {}
    for range_start, range_end in ranges:
        count = len([s for s in f1_scores if range_start <= s < range_end])
        distribution[f"{range_start:.1f}-{range_end:.1f}"] = count
    
    print("\nBERTScore F1 分布:")
    print("-" * 40)
    total = len(f1_scores)
    for range_name, count in distribution.items():
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"  {range_name}: {count:4d} 个样本 ({percentage:5.1f}%)")

# 主程序
if __name__ == "__main__":
    jsonl_file_path = "/home3/wangshuangchen/LLaMA-Factory-main/dataset/match-6-class/cross-fence/Llama-3.2-11B-pre-250.jsonl"
    output_file = "/home3/wangshuangchen/LLaMA-Factory-main/caption_results/BERTScore/Llama-3.2-11B-pre-250.txt"
    model_type = "/home3/wangshuangchen/microsoft/deberta-xlarge-mnli"
    
    # 关键修改3：指定分批大小（20是32G显存+FP16的安全值，可根据实际情况调整）
    batch_size = 20
    
    print("开始计算BERTScore...")
    print(f"输入文件: {jsonl_file_path}")
    print(f"使用模型: {model_type}")
    print(f"分批大小: {batch_size}")
    print("-" * 80)
    
    # 传入分批大小
    results = calculate_bertscore_for_jsonl(jsonl_file_path, output_file, model_type, batch_size=batch_size)
    
    if results and results['f1_scores']:
        print_bertscore_results(results)
        analyze_bertscore_distribution(results)
        save_bertscore_results(results, output_file)
    else:
        print("BERTScore计算失败，请检查文件路径和格式")