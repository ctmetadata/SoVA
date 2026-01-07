#!/usr/bin/env python3
"""
Independent BLEU-2 and Rouge-L calculation script
"""

import json
import argparse
from typing import List, Dict, Tuple
from rouge_score import rouge_scorer
from collections import defaultdict


def load_jsonl_data(file_path: str) -> Tuple[List[str], List[str], List[Dict]]:
    """Load reference and prediction data from JSONL file"""
    references = []
    predictions = []
    items = []
    skipped = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                item = json.loads(line)
                
                if 'reference' not in item or 'prediction' not in item:
                    print(f"Warning: Line {line_num} missing reference or prediction field, skipped")
                    skipped += 1
                    continue
                
                ref = item['reference'].strip()
                pred = item['prediction'].strip()
                
                if not ref or not pred:
                    print(f"Warning: Line {line_num} reference or prediction is empty, skipped")
                    skipped += 1
                    continue
                
                references.append(ref)
                predictions.append(pred)
                items.append(item)
                
            except json.JSONDecodeError as e:
                print(f"Error: Line {line_num} JSON parsing failed: {e}")
                skipped += 1
            except Exception as e:
                print(f"Error: Line {line_num} processing failed: {e}")
                skipped += 1
    
    print(f"Data loading completed:")
    print(f"  Successfully loaded: {len(references)} items")
    print(f"  Skipped: {skipped} invalid items")
    
    return references, predictions, items


def tokenize_text(text: str) -> List[str]:
    """Simple space tokenizer"""
    return text.strip().split()


def calculate_bleu2(references: List[str], predictions: List[str]) -> float:
    """Calculate BLEU-2 score"""
    if not references or not predictions:
        return 0.0
    
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        smoothie = SmoothingFunction().method3
        weights = [0.5, 0.5]  # BLEU-2 weights
        
        scores = []
        for ref, pred in zip(references, predictions):
            ref_tokens = tokenize_text(ref)
            pred_tokens = tokenize_text(pred)
            
            if not ref_tokens or not pred_tokens:
                scores.append(0.0)
                continue
            
            score = sentence_bleu(
                [ref_tokens],
                pred_tokens,
                weights=weights,
                smoothing_function=smoothie
            )
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.0
        
    except ImportError:
        print("Warning: NLTK not installed, cannot calculate BLEU-2 score")
        print("Please install: pip install nltk")
        return 0.0


def calculate_rougel(references: List[str], predictions: List[str]) -> float:
    """Calculate Rouge-L score"""
    if not references or not predictions:
        return 0.0
    
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        scores = []
        for ref, pred in zip(references, predictions):
            score = scorer.score(ref.strip(), pred.strip())['rougeL'].fmeasure
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.0
        
    except ImportError:
        print("Warning: rouge-score not installed, cannot calculate Rouge-L score")
        print("Please install: pip install rouge-score")
        return 0.0


def calculate_by_style(items: List[Dict]) -> Dict[str, Dict]:
    """Calculate metrics grouped by style"""
    style_groups = defaultdict(list)
    for item in items:
        if 'style' in item:
            style = item['style']
            style_groups[style].append(item)
    
    style_metrics = {}
    
    for style, group_items in style_groups.items():
        references = [item['reference'].strip() for item in group_items]
        predictions = [item['prediction'].strip() for item in group_items]
        
        bleu2 = calculate_bleu2(references, predictions)
        rougel = calculate_rougel(references, predictions)
        
        style_metrics[style] = {
            'count': len(group_items),
            'BLEU-2': round(bleu2, 4),
            'Rouge-L': round(rougel, 4)
        }
    
    return style_metrics


def print_metrics_summary(bleu2_score: float, rougel_score: float, total_samples: int):
    """Print metrics summary"""
    print("\n" + "="*50)
    print("          Text Generation Quality Evaluation Results")
    print("="*50)
    print(f"Total samples: {total_samples}")
    print(f"BLEU-2 (B-2): {bleu2_score:.4f}")
    print(f"Rouge-L (R-L): {rougel_score:.4f}")
    print("-"*50)
    print("Explanation:")
    print("  BLEU-2: Measures 2-gram matching, range 0-1, higher is better")
    print("  Rouge-L: Measures longest common subsequence, range 0-1, higher is better")
    print("="*50)


def save_results_to_json(
    bleu2_score: float,
    rougel_score: float,
    total_samples: int,
    style_metrics: Dict,
    output_path: str
):
    """Save results to JSON file"""
    import time
    
    result = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_samples': total_samples,
        'overall_metrics': {
            'BLEU-2': round(bleu2_score, 4),
            'Rouge-L': round(rougel_score, 4)
        },
        'metrics_by_style': style_metrics,
        'note': 'Score range 0~1, closer to 1 indicates better match between generated and reference text'
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"\nComplete results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Calculate BLEU-2 and Rouge-L metrics for text in JSONL file')
    parser.add_argument('--input', '-i', required=True, help='Input JSONL file path')
    parser.add_argument('--output', '-o', help='Output JSON file path (optional)')
    parser.add_argument('--by-style', action='store_true', help='Calculate metrics grouped by style')
    
    args = parser.parse_args()
    
    print(f"Processing file: {args.input}")
    
    references, predictions, items = load_jsonl_data(args.input)
    
    if len(references) == 0:
        print("Error: No valid data found!")
        return
    
    try:
        from nltk.translate.bleu_score import sentence_bleu
    except ImportError:
        print("Warning: NLTK library not installed, BLEU-2 score will not be calculated")
        print("Please install: pip install nltk")
    
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("Warning: rouge-score library not installed, Rouge-L score will not be calculated")
        print("Please install: pip install rouge-score")
    
    bleu2_score = calculate_bleu2(references, predictions)
    rougel_score = calculate_rougel(references, predictions)
    
    print_metrics_summary(bleu2_score, rougel_score, len(references))
    
    if args.by_style:
        style_metrics = calculate_by_style(items)
        if style_metrics:
            print("\nStatistics by Style:")
            print("-"*40)
            for style, metrics in sorted(style_metrics.items()):
                print(f"Style: {style:<15} Samples: {metrics['count']:<5} "
                      f"BLEU-2: {metrics['BLEU-2']:.4f}  Rouge-L: {metrics['Rouge-L']:.4f}")
            print("-"*40)
    else:
        style_metrics = {}
    
    if args.output:
        save_results_to_json(
            bleu2_score, rougel_score, len(references),
            style_metrics, args.output
        )


if __name__ == "__main__":
    main()