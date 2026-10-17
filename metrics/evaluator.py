"""Comprehensive Evaluation Suite matching AI4Bharat IndicXlit standards.
Computes:
1. Top-1 Word Accuracy (1 - WER)
2. Top-5 Beam Word Accuracy
3. Character Error Rate (CER)
4. Phonetic Error Rate (PER)
5. Mean Reciprocal Rank (MRR)
6. Character Confusion Breakdown
"""

from typing import List, Dict, Any
from .cer import compute_cer, levenshtein_distance
from .per import compute_per, to_phonetic_class_string
from .confusion import analyze_dataset_confusion

def compute_mrr(references: List[str], beam_hypotheses: List[List[str]]) -> float:
    """Computes Mean Reciprocal Rank across Top-K beam candidate lists."""
    ranks = []
    for ref, candidates in zip(references, beam_hypotheses):
        rank = 0
        for idx, cand in enumerate(candidates):
            if cand == ref:
                rank = 1.0 / (idx + 1)
                break
        ranks.append(rank)
    return float(sum(ranks)) / float(max(1, len(ranks)))

def evaluate_predictions(references: List[str], 
                         greedy_hypotheses: List[str], 
                         beam_hypotheses: List[List[str]] = None) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics over dataset predictions."""
    total = len(references)
    if total == 0:
        return {}
        
    top1_correct = sum(1 for ref, hyp in zip(references, greedy_hypotheses) if ref == hyp)
    top1_acc = (top1_correct / total) * 100.0
    wer = 100.0 - top1_acc
    
    total_cer = sum(compute_cer(ref, hyp) for ref, hyp in zip(references, greedy_hypotheses))
    mean_cer = total_cer / total
    
    total_per = sum(compute_per(ref, hyp) for ref, hyp in zip(references, greedy_hypotheses))
    mean_per = total_per / total
    
    results = {
        "num_samples": total,
        "top1_accuracy": round(top1_acc, 2),
        "wer": round(wer, 2),
        "mean_cer": round(mean_cer, 4),
        "mean_per": round(mean_per, 4),
    }
    
    if beam_hypotheses:
        top5_correct = 0
        for ref, candidates in zip(references, beam_hypotheses):
            if ref in candidates[:5]:
                top5_correct += 1
        top5_acc = (top5_correct / total) * 100.0
        mrr = compute_mrr(references, beam_hypotheses)
        results["top5_accuracy"] = round(top5_acc, 2)
        results["mrr"] = round(mrr, 4)
        
    results["top_confusions"] = analyze_dataset_confusion(references, greedy_hypotheses, top_k=15)
    return results
