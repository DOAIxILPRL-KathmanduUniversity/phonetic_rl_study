"""Phonetic Error Rate (PER) computation.
Maps Devanagari Unicode characters into phonetic equivalence classes
and calculates Levenshtein distance on the collapsed phonetic string.
This differentiates true transcription failures from minor acoustic homophones.
"""

from .cer import levenshtein_distance

# Phonetic Equivalence Class Mapping
PHONETIC_EQUIVALENCE_MAP = {
    # Sibilants collapsed to S
    'स': 'S', 'श': 'S', 'ष': 'S',
    
    # Hraswa / Dirgha Matras collapsed
    'ि': 'I', 'ी': 'I',
    'ु': 'U', 'ू': 'U',
    'े': 'E', 'ै': 'AI',
    'ो': 'O', 'ौ': 'AU',
    'ा': 'A',
    
    # Independent Vowels
    'अ': 'A', 'आ': 'A',
    'इ': 'I', 'ई': 'I',
    'उ': 'U', 'ऊ': 'U',
    'ए': 'E', 'ऐ': 'AI',
    'ओ': 'O', 'औ': 'AU',
    
    # Labials (va / ba) collapsed to V
    'व': 'V', 'ब': 'V',
    
    # Nasals collapsed
    'ण': 'N', 'न': 'N', 'ङ': 'NG', 'ञ': 'NY', 'ँ': 'M', 'ं': 'M',
    
    # Dental / Retroflex stops collapsed for colloquial speech
    'ट': 'T', 'त': 'T',
    'ठ': 'TH', 'थ': 'TH',
    'ड': 'D', 'द': 'D',
    'ढ': 'DH', 'ध': 'DH',
    
    # Halanta / Virama
    '्': 'HAL',
    
    # Nukta
    '़': ''
}

def to_phonetic_class_string(word: str) -> str:
    """Translates a Devanagari string to its phonetic equivalence representation."""
    out = []
    for ch in word:
        if ch in PHONETIC_EQUIVALENCE_MAP:
            mapped = PHONETIC_EQUIVALENCE_MAP[ch]
            if mapped:
                out.append(mapped)
        else:
            out.append(ch)
    return " ".join(out)

def compute_per(ref: str, hyp: str) -> float:
    """Computes Phonetic Error Rate (PER) between reference and hypothesis.
    Normalizes both into phonetic tokens and calculates Levenshtein distance.
    """
    ref_tokens = to_phonetic_class_string(ref).split()
    hyp_tokens = to_phonetic_class_string(hyp).split()
    
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
        
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
                
    dist = dp[m][n]
    return float(dist) / float(len(ref_tokens))
