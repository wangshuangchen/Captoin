import json
import re
import os
import numpy as np
from PIL import Image, ImageDraw
import datetime
import traceback

import re  


def get_image_size(image_path):
    """Dynamically obtain the image dimensions (width, height) from the image path."""
    try:
        with Image.open(image_path) as img:
            return img.size  #  (width, height)
    except Exception as e:
        print(f"get image size failed ({image_path}): {str(e)}")
        return (640, 480) 
def extract_coordinates(text):
    coordinates = []
    
    try:
        # --------------------------
        # 新增：格式4：[[(x1,y1)], [(x2,y2)]]（双层列表+单层元组）
        # --------------------------
        format4_pattern = r'\[\[<span data-type="inline-math" data-value="XHMqKFxkKylccyosXHMqKFxkKylccyo="></span>\]\s*,\s*\[<span data-type="inline-math" data-value="XHMqKFxkKylccyosXHMqKFxkKylccyo="></span>\]\]'
        format4_matches = re.findall(format4_pattern, text)
        for x1, y1, x2, y2 in format4_matches:
            coordinates.append((int(x1), int(y1)))  # 提取第一个坐标点
            coordinates.append((int(x2), int(y2)))  # 提取第二个坐标点
        if coordinates:  # 优先级：格式4结构化最强，优先返回
            return coordinates

        # --------------------------
        # 原有：格式1：[(x1,y1), (x2,y2)]（单层列表+元组）
        # --------------------------
        format1_pattern = r'\[<span data-type="inline-math" data-value="XHMqKFxkKylccyosXHMqKFxkKylccyo="></span>\s*,\s*<span data-type="inline-math" data-value="XHMqKFxkKylccyosXHMqKFxkKylccyo="></span>\]'
        format1_matches = re.findall(format1_pattern, text)
        for x1, y1, x2, y2 in format1_matches:
            coordinates.append((int(x1), int(y1)))
            coordinates.append((int(x2), int(y2)))
        if coordinates:
            return coordinates

        # --------------------------
        # 新增：格式3：(x1,y1) - (x2,y2)（连字符连接两个坐标）
        # --------------------------
        format3_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*-\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'
        format3_matches = re.findall(format3_pattern, text)
        for x1, y1, x2, y2 in format3_matches:
            coordinates.append((int(x1), int(y1)))
            coordinates.append((int(x2), int(y2)))
        if coordinates:
            return coordinates

        # --------------------------
        # 新增：格式2：(x1,y1) to (x2,y2)（"to"连接两个坐标）
        # --------------------------
        format2_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*to\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'
        format2_matches = re.findall(format2_pattern, text)
        for x1, y1, x2, y2 in format2_matches:
            coordinates.append((int(x1), int(y1)))
            coordinates.append((int(x2), int(y2)))
        if coordinates:
            return coordinates

        # --------------------------
        # 原有兜底：单个坐标 (x,y)（若上述格式均无匹配，提取独立坐标）
        # --------------------------
        standard_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'
        standard_matches = re.findall(standard_pattern, text)
        coordinates = [(int(x), int(y)) for x, y in standard_matches]

    except Exception as e:
        print(f"Error extracting coordinates: {str(e)}")
        return []

    # --------------------------
    # 修正：确保坐标为偶数个（成对出现，避免IOU计算错误）
    # --------------------------
    if len(coordinates) % 2 == 1:
        coordinates = coordinates[:-1]  # 删除最后一个孤立坐标

    print(coordinates)
    return coordinates

def parse_ground_truth_coordinates(coord_str):
    """Parse ground truth coordinate string, return list of regions"""
    if coord_str == "NULL" or coord_str == "null":
        return []
    
    regions = []
    
    try:
        # First try to manually parse complex nested lists
        # Format like: [[[415, 692], [554, 780]], [[171, 298], [260, 380]], ...]
        if '[[' in coord_str and ']]' in coord_str:
            # Extract all numbers
            all_numbers = re.findall(r'\d+', coord_str)
            # If the number count is a multiple of 4, it might be valid coordinate sets
            if len(all_numbers) % 4 == 0:
                # Every 4 numbers form a region (x1,y1,x2,y2)
                for i in range(0, len(all_numbers), 4):
                    if i + 3 < len(all_numbers):
                        x1, y1, x2, y2 = int(all_numbers[i]), int(all_numbers[i+1]), int(all_numbers[i+2]), int(all_numbers[i+3])
                        # Ensure coordinates are valid (top-left smaller than bottom-right)
                        if x1 > x2:
                            x1, x2 = x2, x1
                        if y1 > y2:
                            y1, y2 = y2, y1
                        regions.append(((x1, y1), (x2, y2)))
                
                # If at least one region was successfully parsed, return it
                if regions:
                    return regions
    
        # Try to parse JSON formatted coordinates (nested list form)
        if '[' in coord_str and ']' in coord_str:
            try:
                # Try to parse the string as JSON
                import json
                # Normalize the string, replace single quotes with double quotes
                normalized_str = coord_str.replace("'", '"')
                
                # Try to parse the complete string
                try:
                    nested_coords = json.loads(normalized_str)
                    
                    # Handle nested list format [[[x1,y1], [x2,y2]], ...]
                    if isinstance(nested_coords, list):
                        for region in nested_coords:
                            if isinstance(region, list) and len(region) == 2:
                                if all(isinstance(p, list) and len(p) == 2 for p in region):
                                    x1, y1 = region[0]
                                    x2, y2 = region[1]
                                    # Ensure coordinates are valid
                                    if x1 > x2:
                                        x1, x2 = x2, x1
                                    if y1 > y2:
                                        y1, y2 = y2, y1
                                    regions.append(((x1, y1), (x2, y2)))
                except json.JSONDecodeError:
                    # If the entire string is not valid JSON, try to extract each subregion
                    # Match [[number, number], [number, number]] format
                    region_pattern = r'\[\[(\d+),\s*(\d+)\],\s*\[(\d+),\s*(\d+)\]\]'
                    region_matches = re.findall(region_pattern, coord_str)
                    
                    for x1, y1, x2, y2 in region_matches:
                        # Ensure coordinates are valid
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        if x1 > x2:
                            x1, x2 = x2, x1
                        if y1 > y2:
                            y1, y2 = y2, y1
                        regions.append(((x1, y1), (x2, y2)))
                
                # If at least one region was successfully parsed, return it
                if regions:
                    return regions
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                # If not valid JSON, try other formats
                print(f"JSON parsing coordinates failed: {str(e)}, trying other formats")
        
        # Match [(number,number),(number,number)] format
        pattern = r'\[\((\d+),(\d+)\),\((\d+),(\d+)\)\]'
        matches = re.findall(pattern, coord_str)
        
        for x1, y1, x2, y2 in matches:
            # Ensure x1,y1 is top-left, x2,y2 is bottom-right
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            regions.append(((x1, y1), (x2, y2)))
        
        # If coordinates have been found, return results
        if regions:
            return regions
            
    except Exception as e:
        print(f"Error parsing coordinates: {str(e)}, trying other formats")
    
    try:
        # Finally try direct regex extraction of all numbers, 4 per group
        all_numbers = re.findall(r'\d+', coord_str)
        if len(all_numbers) % 4 == 0:
            for i in range(0, len(all_numbers), 4):
                if i + 3 < len(all_numbers):
                    x1, y1, x2, y2 = int(all_numbers[i]), int(all_numbers[i+1]), int(all_numbers[i+2]), int(all_numbers[i+3])
                    # Ensure coordinates are valid
                    if x1 > x2:
                        x1, x2 = x2, x1
                    if y1 > y2:
                        y1, y2 = y2, y1
                    regions.append(((x1, y1), (x2, y2)))
    except Exception as e:
        print(f"Error using fallback method to parse coordinates: {str(e)}")
    
    return regions

def create_mask_from_regions(regions, image_size=(640, 480)):
    """Create a binary mask image from a list of regions"""
    # Create black background
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    
    # Draw each region as white on the mask
    for region in regions:
        try:
            top_left, bottom_right = region
            # Ensure coordinates form a valid rectangle (top-left smaller than bottom-right)
            if top_left[0] <= bottom_right[0] and top_left[1] <= bottom_right[1]:
                draw.rectangle([top_left, bottom_right], fill=255)
            else:
                # If not valid, try to fix by swapping coordinates
                x1, y1 = top_left
                x2, y2 = bottom_right
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                draw.rectangle([(x1, y1), (x2, y2)], fill=255)
        except Exception as e:
            print(f"Error drawing region {region}: {str(e)}")
    
    return mask

def calculate_iou(pred_regions, gt_regions, image_path):  # 新增image_path参数
    """Calculate IOU between two sets of regions（动态获取图像尺寸）"""
    if not pred_regions or not gt_regions:
        return 0.0
    
    try:
        # 动态获取图像尺寸（替代固定尺寸）
        image_size = get_image_size(image_path)
        
        # 创建预测和真实区域的掩码
        pred_mask = create_mask_from_regions(pred_regions, image_size)
        gt_mask = create_mask_from_regions(gt_regions, image_size)
        
        # 计算交集和并集
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
    except Exception as e:
        print(f"Error calculating IOU: {str(e)}")
        return 0.0

def calculate_giou(pred_regions, gt_regions, image_path):  # 新增image_path参数
    """Calculate GIOU（动态获取图像尺寸）"""
    if not pred_regions or not gt_regions:
        return 0.0
    
    try:
        image_size = get_image_size(image_path)  # 动态获取尺寸
        pred_mask = create_mask_from_regions(pred_regions, image_size)
        gt_mask = create_mask_from_regions(gt_regions, image_size)
        
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        # 后续逻辑不变...
        pred_indices = np.where(pred_mask)
        gt_indices = np.where(gt_mask)
        
        if len(pred_indices[0]) == 0 or len(gt_indices[0]) == 0:
            return 0.0
            
        min_y = min(np.min(pred_indices[0]) if len(pred_indices[0]) > 0 else float('inf'), 
                    np.min(gt_indices[0]) if len(gt_indices[0]) > 0 else float('inf'))
        max_y = max(np.max(pred_indices[0]) if len(pred_indices[0]) > 0 else 0, 
                    np.max(gt_indices[0]) if len(gt_indices[0]) > 0 else 0)
        min_x = min(np.min(pred_indices[1]) if len(pred_indices[1]) > 0 else float('inf'), 
                    np.min(gt_indices[1]) if len(gt_indices[1]) > 0 else float('inf'))
        max_x = max(np.max(pred_indices[1]) if len(pred_indices[1]) > 0 else 0, 
                    np.max(gt_indices[1]) if len(gt_indices[1]) > 0 else 0)
        
        enclosing_area = (max_y - min_y + 1) * (max_x - min_x + 1)
        iou = intersection / union if union > 0 else 0.0
        giou = iou - (enclosing_area - union) / enclosing_area if enclosing_area > 0 else 0.0
        
        return giou
    except Exception as e:
        print(f"Error calculating GIOU: {str(e)}")
        return 0.0

def calculate_diou(pred_regions, gt_regions, image_path):  # 新增image_path参数
    """Calculate DIOU（动态获取图像尺寸）"""
    if not pred_regions or not gt_regions:
        return 0.0
    
    try:
        image_size = get_image_size(image_path)  # 动态获取尺寸
        pred_mask = create_mask_from_regions(pred_regions, image_size)
        gt_mask = create_mask_from_regions(gt_regions, image_size)
        
        # 补全：计算交集和并集
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        if union == 0:
            return 0.0
        iou = intersection / union  # 计算基础IOU
        
        # 补全：从掩码中提取预测框和真实框的边界坐标
        pred_indices = np.where(pred_mask)
        gt_indices = np.where(gt_mask)
        
        if len(pred_indices[0]) == 0 or len(gt_indices[0]) == 0:
            return 0.0
        
        # 预测框边界坐标（y轴对应行，x轴对应列）
        pred_min_y = np.min(pred_indices[0])
        pred_max_y = np.max(pred_indices[0])
        pred_min_x = np.min(pred_indices[1])
        pred_max_x = np.max(pred_indices[1])
        
        # 真实框边界坐标
        gt_min_y = np.min(gt_indices[0])
        gt_max_y = np.max(gt_indices[0])
        gt_min_x = np.min(gt_indices[1])
        gt_max_x = np.max(gt_indices[1])
        
        # 补全：计算中心点坐标
        pred_center_x = (pred_min_x + pred_max_x) / 2
        pred_center_y = (pred_min_y + pred_max_y) / 2
        gt_center_x = (gt_min_x + gt_max_x) / 2
        gt_center_y = (gt_min_y + gt_max_y) / 2
        
        # 补全：计算中心点距离的平方
        center_dist_squared = (pred_center_x - gt_center_x) ** 2 + (pred_center_y - gt_center_y) ** 2
        
        # 补全：计算最小包围框的对角线长度平方
        min_enclose_min_y = min(pred_min_y, gt_min_y)
        min_enclose_max_y = max(pred_max_y, gt_max_y)
        min_enclose_min_x = min(pred_min_x, gt_min_x)
        min_enclose_max_x = max(pred_max_x, gt_max_x)
        diagonal_length_squared = (min_enclose_max_x - min_enclose_min_x) ** 2 + (min_enclose_max_y - min_enclose_min_y) ** 2
        
        # 计算DIOU
        diou = iou - (center_dist_squared / diagonal_length_squared) if diagonal_length_squared > 0 else 0.0
        return diou
    except Exception as e:
        print(f"Error calculating DIOU: {str(e)}")
        return 0.0

def calculate_ciou(pred_regions, gt_regions, image_path):
    """Calculate CIOU (Complete IoU) between two sets of regions（动态获取图像尺寸）"""
    if not pred_regions or not gt_regions:
        return 0.0
    
    try:
        # 动态获取图像尺寸
        image_size = get_image_size(image_path)
        pred_mask = create_mask_from_regions(pred_regions, image_size)
        gt_mask = create_mask_from_regions(gt_regions, image_size)
        
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        # 防护：并集为0时直接返回0（避免后续除以0）
        if union == 0:
            return 0.0
        
        pred_indices = np.where(pred_mask)
        gt_indices = np.where(gt_mask)
        
        if len(pred_indices[0]) == 0 or len(gt_indices[0]) == 0:
            return 0.0
            
        # 获取边界框坐标
        pred_min_y = np.min(pred_indices[0])
        pred_max_y = np.max(pred_indices[0])
        pred_min_x = np.min(pred_indices[1])
        pred_max_x = np.max(pred_indices[1])
        
        gt_min_y = np.min(gt_indices[0])
        gt_max_y = np.max(gt_indices[0])
        gt_min_x = np.min(gt_indices[1])
        gt_max_x = np.max(gt_indices[1])
        
        # 计算中心点距离
        pred_center_x = (pred_min_x + pred_max_x) / 2
        pred_center_y = (pred_min_y + pred_max_y) / 2
        gt_center_x = (gt_min_x + gt_max_x) / 2
        gt_center_y = (gt_min_y + gt_max_y) / 2
        
        center_dist_squared = (pred_center_x - gt_center_x) ** 2 + (pred_center_y - gt_center_y) ** 2
        
        # 计算最小包围框对角线长度平方
        min_y = min(pred_min_y, gt_min_y)
        max_y = max(pred_max_y, gt_max_y)
        min_x = min(pred_min_x, gt_min_x)
        max_x = max(pred_max_x, gt_max_x)
        
        diagonal_length_squared = (max_x - min_x) ** 2 + (max_y - min_y) ** 2
        
        # 防护：对角线为0的情况
        if diagonal_length_squared == 0:
            return 0.0
        
        # 计算宽高比相关参数
        pred_width = pred_max_x - pred_min_x
        pred_height = pred_max_y - pred_min_y
        gt_width = gt_max_x - gt_min_x
        gt_height = gt_max_y - gt_min_y
        
        # 防护：避免宽高为0导致的除零错误
        if pred_height <= 0 or gt_height <= 0:
            return 0.0
            
        # 计算宽高比的反正切值
        pred_arctan = np.arctan(pred_width / pred_height)
        gt_arctan = np.arctan(gt_width / gt_height)
        
        # 计算宽高比一致性项v
        v = 4 / (np.pi ** 2) * (pred_arctan - gt_arctan) ** 2
        # 确保v是非负数
        v = max(0.0, v)
        
        # 计算alpha
        iou = intersection / union
        denominator = (1 - iou) + v
        # 防护：避免分母为0
        if denominator == 0:
            alpha = 0.0
        else:
            alpha = v / denominator
        
        # 计算CIOU
        ciou = iou - (center_dist_squared / diagonal_length_squared + alpha * v)
        
        # 确保结果在合理范围内且不是nan
        if np.isnan(ciou):
            return 0.0
        # CIOU理论范围在[-1, 1]之间
        return max(-1.0, min(1.0, ciou))
        
    except Exception as e:
        print(f"Error calculating CIOU: {str(e)}")
        return 0.0


def calculate_precision_recall_f1(true_positives, false_positives, false_negatives):
    """Calculate precision, recall, and F1 score"""
    try:
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1
    except Exception as e:
        print(f"Error calculating precision, recall, and F1 score: {str(e)}")
        return 0.0, 0.0, 0.0

def extract_answer_option(text):
    """Extract option X from "The answer is (X)" in text"""
    try:
        match = re.search(r'The answer is \(([A-E])\)', text)
        if match:
            return match.group(1)
            
        # Match format with suffix, e.g., "The answer is (D) no defect."
        match = re.search(r'The answer is \(([A-E])\)[\s\w\.]*', text)
        if match:
            return match.group(1)
            
        # Match "The answer is (X = A) break" or "The answer is (X = B) chip" format
        match = re.search(r'The answer is \([A-Z]\s*=\s*([A-E])[^\)]*\)', text)
        if match:
            return match.group(1)
            
        # Try other possible formats
        match = re.search(r'The answer is \(([A-E])', text)
        if match:
            return match.group(1)
            
        match = re.search(r'answer is \(([A-E])\)', text)
        if match:
            return match.group(1)
            
        match = re.search(r'answer is \(([A-E])\)[\s\w\.]*', text)
        if match:
            return match.group(1)
            
        match = re.search(r'answer is ([A-E])', text)
        if match:
            return match.group(1)
        
        match = re.search(r'option is \(([A-E])\)', text)
        if match:
            return match.group(1)
            
        match = re.search(r'option is \(([A-E])\)[\s\w\.]*', text)
        if match:
            return match.group(1)
            
        match = re.search(r'option is ([A-E])', text)
        if match:
            return match.group(1)
        
        match = re.search(r'[Oo]ption is ([A-E])[\.\s]', text)
        if match:
            return match.group(1)
            
        # Directly find possible options
        match = re.search(r'The answer is ([A-E])', text)
        if match:
            return match.group(1)
            
        # Look for patterns like "answer is A."
        match = re.search(r'[Aa]nswer is ([A-E])[\.\s]', text)
        if match:
            return match.group(1)
            
        # Finally try to match any mentioned options
        match = re.search(r'[Tt]he answer is \(?([A-E])', text)
        if match:
            return match.group(1)
            
        return None
    except Exception as e:
        print(f"Error extracting answer option: {str(e)}")
        return None

def extract_ground_truth_option(text):
    """Extract option from ground truth answer"""
    try:
        # A-E: re.search(r'\(([A-E])\)', text)
        match = re.search(r'\(([A-E])\s*=.*?\)', text)
        if match:
            return match.group(1)
        return None
    except Exception as e:
        print(f"Error extracting ground truth option: {str(e)}")
        return None

def evaluate_results(jsonl_file):
    """Evaluate results file, calculate IOU and accuracy"""
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_items = 0  # Total sample count
    valid_items = 0  # Valid sample count (excluding Error)
    valid_iou_items = 0
    total_iou = 0.0
    total_giou = 0.0
    total_diou = 0.0
    total_ciou = 0.0
    correct_answers = 0
    valid_answer_items = 0
    total_answer_items = 0
    extraction_failures = []
    error_samples = []  # Record Error samples
    
    # Used to calculate total number of samples that need to participate in IOU calculation (coordinates not NULL)
    iou_eligible_samples = 0
    
    # Used to calculate Precision, Recall, F1
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    results = []
    
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
            
        item_result = {
            "line_number": i,
            "extraction_errors": []
        }
        
        try:
            # Parse JSON
            item = json.loads(line)
            total_items += 1
            item_result["image_path"] = item['image_path']
            
            # Check if output is "Error"
            if item.get('output') == "Error":
                item_result["is_error"] = True
                item_result["pred_option"] = "Error"
                item_result["gt_option"] = extract_ground_truth_option(item['answer']) or "N/A"
                item_result["correct_answer"] = "N/A"
                item_result["iou"] = "N/A"
                item_result["giou"] = "N/A"
                item_result["diou"] = "N/A"
                item_result["ciou"] = "N/A"
                item_result["iou_note"] = "Output is Error, not calculated"
                item_result["extraction_errors"].append("output is Error")
                error_samples.append({
                    "line_number": i,
                    "image_path": item['image_path'],
                    "error": "output is Error"
                })
                results.append(item_result)
                continue
            
            # Only non-Error samples count as valid samples
            valid_items += 1
            
            # Check if coordinates are "NULL"
            is_null_coordinates = item.get('coordinates', '') == "NULL" or item.get('coordinates', '') == "null"
            
            # If coordinates are not NULL, then the sample should participate in IOU calculation
            if not is_null_coordinates:
                iou_eligible_samples += 1
            
            # Extract predicted answer and ground truth answer
            pred_option = extract_answer_option(item['output'])
            gt_option = extract_ground_truth_option(item['answer'])
            
            # Check if answer extraction is successful
            if pred_option is None:
                error_msg = f"Cannot extract predicted answer option from output: {item['output'][:100]}..."
                item_result["extraction_errors"].append(error_msg)
                extraction_failures.append({
                    "line_number": i,
                    "image_path": item['image_path'],
                    "error": error_msg
                })
            
            if gt_option is None:
                error_msg = f"Cannot extract option from ground truth answer: {item['answer']}"
                item_result["extraction_errors"].append(error_msg)
                extraction_failures.append({
                    "line_number": i,
                    "image_path": item['image_path'],
                    "error": error_msg
                })
            
            # Record answer option, even if extraction fails
            item_result["pred_option"] = pred_option if pred_option is not None else "N/A"
            item_result["gt_option"] = gt_option if gt_option is not None else "N/A"
            
            # Check answer correctness
            if pred_option is not None and gt_option is not None:
                is_correct = pred_option == gt_option
                if is_correct:
                    correct_answers += 1
                    true_positives += 1
                else:
                    false_positives += 1
                    false_negatives += 1
                valid_answer_items += 1
                item_result["correct_answer"] = is_correct
            else:
                item_result["correct_answer"] = False  # If answer cannot be extracted, consider it incorrect
                false_negatives += 1  # Unsuccessful extraction is considered missed
                total_answer_items += 1
            
            # If coordinates are NULL, no IOU calculation is needed
            if is_null_coordinates:
                item_result["iou"] = "N/A"
                item_result["giou"] = "N/A"
                item_result["diou"] = "N/A"
                item_result["ciou"] = "N/A"
                item_result["iou_note"] = "Coordinates are NULL, not counted in IOU statistics"
                item_result["pred_regions"] = []
                item_result["gt_regions"] = []
                continue
            
            # Extract predicted coordinates and ground truth coordinates (only process non-NULL coordinates)
            try:
                pred_coords = extract_coordinates(item['output'])
                if not pred_coords:
                    error_msg = f"Cannot extract coordinates from output: {item['output'][:100]}..."
                    item_result["extraction_errors"].append(error_msg)
                    extraction_failures.append({
                        "line_number": i,
                        "image_path": item['image_path'],
                        "error": error_msg
                    })
            except Exception as e:
                error_msg = f"Error extracting predicted coordinates: {str(e)}"
                item_result["extraction_errors"].append(error_msg)
                extraction_failures.append({
                    "line_number": i,
                    "image_path": item.get('image_path', 'N/A'),
                    "error": error_msg
                })
                pred_coords = []
            
            try:
                gt_regions = parse_ground_truth_coordinates(item['coordinates'])
                if not gt_regions and item['coordinates'] != "NULL" and item['coordinates'] != "null":
                    error_msg = f"Cannot extract coordinate region from coordinates field: {item['coordinates']}"
                    item_result["extraction_errors"].append(error_msg)
                    extraction_failures.append({
                        "line_number": i,
                        "image_path": item['image_path'],
                        "error": error_msg
                    })
            except Exception as e:
                error_msg = f"Error parsing ground truth coordinates: {str(e)}"
                item_result["extraction_errors"].append(error_msg)
                extraction_failures.append({
                    "line_number": i,
                    "image_path": item.get('image_path', 'N/A'),
                    "error": error_msg
                })
                gt_regions = []
            
            # Convert predicted coordinates to regions
            pred_regions = []
            try:
                for j in range(0, len(pred_coords), 2):
                    if j + 1 < len(pred_coords):
                        # Ensure coordinates form a valid rectangle
                        x1, y1 = pred_coords[j]
                        x2, y2 = pred_coords[j+1]
                        if x1 > x2:
                            x1, x2 = x2, x1
                        if y1 > y2:
                            y1, y2 = y2, y1
                        pred_regions.append(((x1, y1), (x2, y2)))
            except Exception as e:
                error_msg = f"Error building predicted regions: {str(e)}"
                item_result["extraction_errors"].append(error_msg)
                pred_regions = []
            
            # Calculate IOU and other metrics (for non-NULL coordinates)
            iou = 0.0
            giou = 0.0
            diou = 0.0
            ciou = 0.0
            try:
                if gt_regions or pred_regions:  # Only try to calculate if at least one is not empty
                    # 动态传入图像路径，用于获取尺寸
                    iou = calculate_iou(pred_regions, gt_regions, item['image_path'])
                    giou = calculate_giou(pred_regions, gt_regions, item['image_path'])
                    diou = calculate_diou(pred_regions, gt_regions, item['image_path'])
                    ciou = calculate_ciou(pred_regions, gt_regions, item['image_path'])
                    if gt_regions and pred_regions:  # Only calculate valid IOU if both regions exist
                        valid_iou_items += 1
                total_iou += iou  # Add non-NULL samples' IOU to total
                total_giou += giou
                total_diou += diou
                total_ciou += ciou
                item_result["iou"] = iou
                item_result["giou"] = giou
                item_result["diou"] = diou
                item_result["ciou"] = ciou
            except Exception as e:
                error_msg = f"Error calculating IoU metrics: {str(e)}"
                item_result["extraction_errors"].append(error_msg)
                item_result["iou"] = 0.0
                item_result["giou"] = 0.0
                item_result["diou"] = 0.0
                item_result["ciou"] = 0.0
            
            # Add auxiliary information (for non-NULL coordinates)
            if not gt_regions:
                item_result["iou_note"] = "No ground truth region"
            elif not pred_regions:
                item_result["iou_note"] = "No predicted region"
            
            item_result["pred_regions"] = pred_regions
            item_result["gt_regions"] = gt_regions
                
        except Exception as e:
            error_msg = f"Error processing line {i}: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            extraction_failures.append({
                "line_number": i,
                "error": error_msg
            })
            
            # Even if an error occurs, add the result, using empty values
            item_result = {
                "line_number": i,
                "image_path": "Unknown",
                "pred_option": "N/A",
                "gt_option": "N/A",
                "correct_answer": False,
                "iou": 0.0,
                "giou": 0.0,
                "diou": 0.0,
                "ciou": 0.0,
                "pred_regions": [],
                "gt_regions": [],
                "extraction_errors": [error_msg]
            }
            
            total_items += 1
        
        # Add result regardless of processing success
        results.append(item_result)
    
    # Calculate average IOU and accuracy-related metrics
    avg_iou = total_iou / iou_eligible_samples if iou_eligible_samples > 0 else 0
    avg_giou = total_giou / iou_eligible_samples if iou_eligible_samples > 0 else 0
    avg_diou = total_diou / iou_eligible_samples if iou_eligible_samples > 0 else 0
    avg_ciou = total_ciou / iou_eligible_samples if iou_eligible_samples > 0 else 0
    
    accuracy = correct_answers / valid_answer_items if valid_answer_items > 0 else 0
    
    # Calculate precision, recall, and F1 score
    precision, recall, f1 = calculate_precision_recall_f1(true_positives, false_positives, false_negatives)
    
    # Summarize results
    summary = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_evaluated": jsonl_file,
        "total_items": total_items,
        "valid_items": valid_items,
        "error_samples": len(error_samples),
        "iou_eligible_samples": iou_eligible_samples,
        "valid_answer_items": valid_answer_items,
        "valid_iou_items": valid_iou_items,
        "correct_answers": correct_answers,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_iou": avg_iou,
        "avg_giou": avg_giou,
        "avg_diou": avg_diou,
        "avg_ciou": avg_ciou,
        "extraction_failures": extraction_failures,
        "error_samples": error_samples,
        "detailed_results": results
    }
    
    # Print detailed results
    print(f"Evaluation results:")
    print(f"Total sample count: {total_items}")
    print(f"Valid samples (excluding Error): {valid_items}")
    print(f"Error sample count: {len(error_samples)}")
    print(f"Valid answer samples: {valid_answer_items}")
    print(f"Samples needing IOU calculation (coordinates not NULL): {iou_eligible_samples}")
    print(f"Valid IOU samples: {valid_iou_items}")
    print(f"Average IOU: {avg_iou:.4f}")
    print(f"Average GIOU: {avg_giou:.4f}")
    print(f"Average DIOU: {avg_diou:.4f}")
    print(f"Average CIOU: {avg_ciou:.4f}")
    print(f"Accuracy: {accuracy:.4f} ({correct_answers}/{valid_answer_items})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")
    
    if error_samples:
        print(f"\nError samples ({len(error_samples)}):")
        for sample in error_samples:
            print(f"Line {sample['line_number']}: {sample.get('image_path', 'N/A')}")
    
    if extraction_failures:
        print(f"\nFailed samples ({len(extraction_failures)}):")
        for failure in extraction_failures:
            print(f"Line {failure['line_number']}: {failure.get('image_path', 'N/A')} - {failure['error']}")
    
    # Print results for each sample
    print("\nDetailed results:")
    for result in results:
        if result.get("is_error"):
            print(f"Image: {result.get('image_path', 'Unknown')}, Status: Error, Not calculated")
            continue
            
        iou_str = f"{result['iou']:.4f}" if isinstance(result['iou'], float) else result['iou']
        errors = ', '.join(result.get('extraction_errors', []))
        error_info = f" [Error: {errors}]" if errors else ""
        
        print(f"Image: {result.get('image_path', 'Unknown')}, IOU: {iou_str}, "
              f"Predicted/Ground Truth Option: {result['pred_option']}/{result['gt_option']}, "
              f"Correct: {result['correct_answer']}{error_info}")
    
    return summary

def save_results_to_txt(summary, output_file=None):
    """Save evaluation results to TXT file, in readable format"""
    if output_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"evaluation_results_{timestamp}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write title
        f.write("="*50 + "\n")
        f.write(f"Evaluation results summary - {summary['timestamp']}\n")
        f.write("="*50 + "\n\n")
        
        # Write basic information
        f.write(f"Evaluated file: {summary['file_evaluated']}\n")
        f.write(f"Total sample count: {summary['total_items']}\n")
        f.write(f"Valid samples (excluding Error): {summary['valid_items']}\n")
        f.write(f"Error sample count: {summary['error_samples']}\n")
        f.write(f"Valid answer samples: {summary['valid_answer_items']}\n")
        f.write(f"Samples needing IOU calculation (coordinates not NULL): {summary['iou_eligible_samples']}\n")
        f.write(f"Valid IOU samples: {summary['valid_iou_items']}\n")
        
        # Write IoU related metrics
        f.write(f"Average IOU: {summary['avg_iou']:.4f}\n")
        f.write(f"Average GIOU: {summary['avg_giou']:.4f}\n")
        f.write(f"Average DIOU: {summary['avg_diou']:.4f}\n")
        f.write(f"Average CIOU: {summary['avg_ciou']:.4f}\n")
        
        # Write classification metrics
        f.write(f"Accuracy: {summary['accuracy']:.4f} ({summary['correct_answers']}/{summary['valid_answer_items']})\n")
        f.write(f"Precision: {summary['precision']:.4f}\n")
        f.write(f"Recall: {summary['recall']:.4f}\n")
        f.write(f"F1 score: {summary['f1']:.4f}\n\n")
        
        # Write Error samples
        if summary.get('error_samples'):
            f.write("-"*50 + "\n")
            f.write(f"Error samples ({len(summary['error_samples'])} samples):\n")
            f.write("-"*50 + "\n")
            for sample in summary['error_samples']:
                f.write(f"Line {sample['line_number']}: {sample.get('image_path', 'N/A')}\n")
            f.write("\n")
        
        # Write failed samples
        if summary['extraction_failures']:
            f.write("-"*50 + "\n")
            f.write(f"Failed samples ({len(summary['extraction_failures'])} samples):\n")
            f.write("-"*50 + "\n")
            for failure in summary['extraction_failures']:
                f.write(f"Line {failure['line_number']}: {failure.get('image_path', 'N/A')} - {failure['error']}\n")
            f.write("\n")
        
        # Write detailed results
        f.write("-"*50 + "\n")
        f.write("Detailed evaluation results:\n")
        f.write("-"*50 + "\n")
        
        for i, result in enumerate(summary['detailed_results']):
            f.write(f"Sample {i+1}:   Line number {result.get('line_number', 'N/A')}: {result.get('image_path', 'Unknown')}\n")
            
            # Handle Error samples
            if result.get("is_error"):
                f.write(f"   Status: Error, Not calculated\n")
                f.write(f"   Ground truth option: {result['gt_option']}\n\n")
                continue
                
            # Write option information
            f.write(f"   Predicted option: {result['pred_option']}\n")
            f.write(f"   Ground truth option: {result['gt_option']}\n")
            f.write(f"   Option correct: {result['correct_answer']}\n")
            
            # Write IoU related information
            if isinstance(result['iou'], float):
                f.write(f"  IOU: {result['iou']:.4f}\n")
                f.write(f"  GIOU: {result['giou']:.4f}\n")
                f.write(f"  DIOU: {result['diou']:.4f}\n")
                f.write(f"  CIOU: {result['ciou']:.4f}\n")
            else:
                f.write(f"  IOU: {result['iou']}")
                if 'iou_note' in result:
                    f.write(f" ({result['iou_note']})")
                f.write("\n")
                f.write(f"  GIOU: {result['giou']}\n")
                f.write(f"  DIOU: {result['diou']}\n")
                f.write(f"  CIOU: {result['ciou']}\n")
            
            # Write error information
            if result.get('extraction_errors'):
                f.write("   Extraction error:\n")
                for error in result['extraction_errors']:
                    f.write(f"    - {error}\n")
            
            f.write("\n")
    
    print(f"\nResults saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    # Evaluate test file
    jsonl_file = "/home3/wangshuangchen/LLaMA-Factory-main/dataset/lora_jsonl/Llama-3.2-11B-lora.jsonl"

    summary = evaluate_results(jsonl_file)
    
    # Save results to TXT file
    output_file = "/home3/wangshuangchen/LLaMA-Factory-main/dataset/IOU/Qwen2.5-VL-7B-lora-2350.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Try using absolute path
    # if not os.path.exists(os.path.dirname(output_file)):
    #     output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "COT_0423/evaluation_results.txt")
    # if not os.path.exists(os.path.dirname(output_file)):
    #     output_file = "matao/LLaMA-Factory-beifen/COT_0423/evaluation_results.txt"
        
    save_results_to_txt(summary, output_file) 