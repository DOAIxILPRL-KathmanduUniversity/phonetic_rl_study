"""Reward Suite for Reinforcement Learning Transliteration.

Supports 5 distinct reward regimes:
1. sparse_exact_match (Sutton & Barto sparse reward hypothesis: 1.0 on exact match, 0.0 otherwise)
2. dense_cer (Continuous character distance feedback: 1.0 - CER)
3. composite_balanced (Multi-objective: 0.35 CER + 0.30 PER + 0.20 EM + 0.15 Prefix)
4. composite_with_penalties (Composite minus orthotactic penalties for consecutive halantas or mid-word vowels)
5. dominant_phonetic (Prioritizes acoustic / phonetic correctness: 0.70 PER + 0.20 CER + 0.10 EM)
"""

import torch
from typing import List, Dict
from metrics.cer import compute_cer
from metrics.per import compute_per

INDEPENDENT_VOWELS = set(['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ', 'ऋ'])


def check_orthotactic_penalties(word: str) -> float:
    """Calculates orthotactic legality penalties:
    - Consecutive halantas ('््')
    - Independent vowels placed immediately after a consonant without a halanta
    - Degenerate character repetition (3+ identical characters)
    """
    penalty = 0.0
    if '््' in word:
        penalty += 0.35
        
    # Check for degenerate repetition
    for i in range(len(word) - 2):
        if word[i] == word[i+1] == word[i+2]:
            penalty += 0.30
            break
            
    # Check for independent vowels placed illegally mid-word
    for i in range(1, len(word)):
        if word[i] in INDEPENDENT_VOWELS and word[i-1] not in [' ', '्', '-']:
            penalty += 0.25
            break
            
    return penalty


def compute_single_reward(hyp: str, ref: str, reward_type: str = "composite_balanced") -> float:
    """Computes scalar reward for a single (hypothesis, reference) word pair."""
    is_exact = 1.0 if hyp == ref else 0.0
    cer_score = max(0.0, 1.0 - compute_cer(ref, hyp))
    per_score = max(0.0, 1.0 - compute_per(ref, hyp))
    
    # Prefix match score
    common_prefix = 0
    min_len = min(len(ref), len(hyp))
    while common_prefix < min_len and ref[common_prefix] == hyp[common_prefix]:
        common_prefix += 1
    prefix_score = common_prefix / max(1, len(ref))
    
    if reward_type == "sparse_exact_match":
        return is_exact
        
    elif reward_type == "dense_cer":
        return cer_score
        
    elif reward_type == "composite_balanced":
        return (0.35 * cer_score + 
                0.30 * per_score + 
                0.20 * is_exact + 
                0.15 * prefix_score)
                
    elif reward_type == "composite_with_penalties":
        base_reward = (0.35 * cer_score + 
                       0.30 * per_score + 
                       0.20 * is_exact + 
                       0.15 * prefix_score)
        penalty = check_orthotactic_penalties(hyp)
        return max(0.0, base_reward - penalty)
        
    elif reward_type == "dominant_phonetic":
        return (0.70 * per_score + 
                0.20 * cer_score + 
                0.10 * is_exact)
                
    else:
        # Default fallback
        return 0.5 * cer_score + 0.5 * is_exact


def compute_batch_rewards(hypotheses: List[str], 
                          references: List[str], 
                          reward_type: str = "composite_balanced", 
                          device: torch.device = None) -> torch.Tensor:
    """Computes rewards for a batch of generated words."""
    rewards = []
    for hyp, ref in zip(hypotheses, references):
        r = compute_single_reward(hyp, ref, reward_type=reward_type)
        rewards.append(r)
    tensor = torch.tensor(rewards, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor
