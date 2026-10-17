"""Character Error Rate (CER) computation based on Levenshtein edit distance."""

def levenshtein_distance(ref: str, hyp: str):
    """Computes Levenshtein distance and counts of substitutions, insertions, deletions."""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],      # deletion
                                   dp[i][j - 1],      # insertion
                                   dp[i - 1][j - 1])  # substitution
                
    dist = dp[m][n]
    
    # Backtrace to count S, I, D
    i, j = m, n
    subs, ins, dels = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            break
            
    return dist, subs, ins, dels

def compute_cer(ref: str, hyp: str) -> float:
    """Computes normalized Character Error Rate (CER) = (S + I + D) / len(ref)."""
    if not ref:
        return 0.0 if not hyp else 1.0
    dist, _, _, _ = levenshtein_distance(ref, hyp)
    return float(dist) / float(len(ref))
