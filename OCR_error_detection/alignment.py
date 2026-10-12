"""Word-level alignment engine with multi-optimal path tracking.

Ported from ocr_metrics.py in the parent repo. Uses exact DP to align
two word sequences, minimizing edit count then maximizing exact matches.
Tracks ALL tied optimal paths via bitmasks so we know which positions
are certain matches vs ambiguous.
"""


def align_tokens(reference, hypothesis):
    """Align two word sequences using DP.

    Returns a list of alignment steps: tuples of (ref_idx, hyp_idx) where
    None means insertion or deletion.

    Returns:
        steps: list of (ref_idx|None, hyp_idx|None) pairs
        distance: edit distance (word level)
        insertions: number of inserted words (in hypothesis, not in reference)
        deletions: number of deleted words (in reference, not in hypothesis)
    """
    n = len(reference)
    m = len(hypothesis)

    # DP table: dp[i][j] = (distance, -matches) for best alignment
    # of reference[:i] to hypothesis[:j]
    width = m + 1
    directions = [bytearray(width) for _ in range(n + 1)]
    directions[0][1:] = bytes([4]) * (width - 1)  # insertion bits

    previous = [(j, 0) for j in range(width)]
    for i, word in enumerate(reference, 1):
        directions[i][0] = 2  # deletion bit
        current = [(i, 0)]
        for j, other in enumerate(hypothesis, 1):
            match = word == other
            prior = previous[j - 1]
            diagonal = (prior[0] + (not match), prior[1] - match)
            deletion = (previous[j][0] + 1, *previous[j][1:])
            insertion = (current[-1][0] + 1, *current[-1][1:])
            choices = (diagonal, deletion, insertion)
            direction = min(range(3), key=choices.__getitem__)
            best = choices[direction]
            directions[i][j] = sum(1 << k for k, choice in enumerate(choices) if choice == best)
            current.append(best)
        previous = current

    distance = previous[-1][0]

    # Traceback
    steps = []
    matches = set()
    insertions = deletions = 0
    i, j = n, m
    while i or j:
        direction = directions[i][j]
        if direction & 1:
            if reference[i - 1] == hypothesis[j - 1]:
                matches.add(i - 1)
            steps.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif direction & 2:
            steps.append((i - 1, None))
            deletions += 1
            i -= 1
        else:
            steps.append((None, j - 1))
            insertions += 1
            j -= 1

    steps.reverse()
    return steps, distance, insertions, deletions, matches