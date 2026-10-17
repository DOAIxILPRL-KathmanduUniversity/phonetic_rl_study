"""Character Confusion Matrix & Substitution Analysis.
Identifies which Devanagari character pairs are most frequently substituted.
"""

from collections import Counter
import difflib
from typing import List, Tuple, Dict

def extract_substitutions(ref: str, hyp: str) -> List[Tuple[str, str]]:
    """Extracts (ref_char, hyp_char) substitution pairs from alignment."""
    matcher = difflib.SequenceMatcher(None, ref, hyp)
    subs = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            ref_sub = ref[i1:i2]
            hyp_sub = hyp[j1:j2]
            # Match 1-to-1 characters as far as possible
            min_len = min(len(ref_sub), len(hyp_sub))
            for k in range(min_len):
                subs.append((ref_sub[k], hyp_sub[k]))
            if len(ref_sub) > min_len:
                for k in range(min_len, len(ref_sub)):
                    subs.append((ref_sub[k], '<DEL>'))
            elif len(hyp_sub) > min_len:
                for k in range(min_len, len(hyp_sub)):
                    subs.append(('<INS>', hyp_sub[k]))
        elif tag == 'delete':
            for k in range(i1, i2):
                subs.append((ref[k], '<DEL>'))
        elif tag == 'insert':
            for k in range(j1, j2):
                subs.append(('<INS>', hyp[k]))
    return subs

def analyze_dataset_confusion(references: List[str], hypotheses: List[str], top_k: int = 20) -> List[Dict]:
    """Analyzes character confusion across an entire evaluation split."""
    counter = Counter()
    for ref, hyp in zip(references, hypotheses):
        subs = extract_substitutions(ref, hyp)
        for s in subs:
            if s[0] != s[1]:
                counter[s] += 1
                
    total_errors = sum(counter.values())
    results = []
    for (ref_ch, hyp_ch), count in counter.most_common(top_k):
        results.append({
            "reference": ref_ch,
            "hypothesis": hyp_ch,
            "count": count,
            "percentage": round((count / max(1, total_errors)) * 100, 2)
        })
    return results
