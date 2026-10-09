#!/usr/bin/env python3
"""Phase A: assert targeted repairs and report independent cold/warm timings.

Run once per backend (separate processes), using the same options. No model is
loaded at import time. Cold means application caches cleared, not cold OS pages
or cleared internal caches in the external segmenter/KenLM.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

from amharic_ocr_corrector import AmharicOCRCorrector, EthiopicPunctuationManager

KENLM_BIN = "/mnt/mintdata/Users/Des/Mint/Models/KenLM/word_model/amharic_5gram.binary"
DICT_FILE = "/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Projects/p16_morpheme_analysis/data/main_i7.txt"
CONFUSION_JSON = str(Path(__file__).with_name('confusion_matrix.json'))
TEST_SENTENCES = [
    "የኢትዮጵያ፣ ግብጽ አና ሱዳን ባለሙያዎች በግድቡ የቴክኒክ አና የሕግ ጉዳይ ዙሪያ ውይይት አንደሚያደርጉ ይጠበቃል",
    "ሶስተኛው ዙር የሕዳሴው ግድብ ሶስትዮሽቨ ውይይት በግብጽ ተጀመረ።",
    "ሶስተኛው ተመሳሳይ ውይይት ዛሬ በግብጽ ካይሮ በመካሄድ ላይ መሆኑን በሀዳሴው ግድብ የኢትዮጵያ",
]
TARGETS = {'አና': 'እና', 'አንደሚያደርጉ': 'እንደሚያደርጉ',
           'ሶስትዮሽቨ': 'ሶስትዮሽ', 'በሀዳሴው': 'በህዳሴው'}


def lexical_tokens(text):
    normalized = EthiopicPunctuationManager.normalize_sentence_text(text)
    return [EthiopicPunctuationManager.decompose_token(token)[1] for token in normalized.split()]


def check_targets(original, corrected):
    """Check each targeted position, not just presence anywhere in the output."""
    before, after = lexical_tokens(original), lexical_tokens(corrected)
    length_ok = len(before) == len(after)
    return [dict(position=i, observed=word, expected=TARGETS[word],
                 actual=after[i] if i < len(after) else None,
                 passed=length_ok and after[i] == TARGETS[word])
            for i, word in enumerate(before) if word in TARGETS]


def timed(call):
    start = time.perf_counter()
    value = call()
    return value, (time.perf_counter() - start) * 1000


def timed_sentence(corrector, sentence):
    before = corrector.morphology_stats()
    output, elapsed = timed(lambda: corrector.correct_sentence(sentence))
    after = corrector.morphology_stats()
    delta = {key: after[key] - before[key] for key in
             ('trusted_bypasses', 'segmenter_calls', 'segmenter_failures', 'segmenter_seconds')}
    delta['cache_hits'] = after['cache']['hits'] - before['cache']['hits']
    delta['cache_misses'] = after['cache']['misses'] - before['cache']['misses']
    return output, elapsed, delta


def peak_rss_mib():
    try:
        import resource
        import sys
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024 * 1024 if sys.platform == 'darwin' else 1024)
    except ImportError:
        return None


def run_benchmark(corrector, repeats=3):
    if repeats < 1:
        raise ValueError('repeats must be positive')
    rows = []
    for sentence in TEST_SENTENCES:
        cold, warm = [], []
        cold_morphology, warm_morphology = [], []
        checks = []
        outputs = []
        for _ in range(repeats):
            corrector.clear_caches()
            output, elapsed, stats = timed_sentence(corrector, sentence)
            cold.append(elapsed)
            cold_morphology.append(stats)
            checks.extend(check_targets(sentence, output))
            outputs.append(output)
            warm_output, elapsed, stats = timed_sentence(corrector, sentence)
            warm.append(elapsed)
            warm_morphology.append(stats)
            checks.extend(check_targets(sentence, warm_output))
            outputs.append(warm_output)
        rows.append(dict(input=sentence, output=outputs[0], cold_ms=cold, warm_ms=warm,
                         cold_morphology=cold_morphology, warm_morphology=warm_morphology,
                         cold_median_ms=statistics.median(cold), warm_median_ms=statistics.median(warm),
                         stable_output=len(set(outputs)) == 1, measured_outputs=outputs, checks=checks,
                         non_target_changes=[dict(position=i, before=before, after=after)
                             for i, (before, after) in enumerate(zip(lexical_tokens(sentence), lexical_tokens(outputs[0])))
                             if before not in TARGETS and before != after],
                         token_count_changed=len(lexical_tokens(sentence)) != len(lexical_tokens(outputs[0]))))

    # Separate diagnostic pass, never used to prime the cold sentence benchmark.
    token_rows = []
    for word in dict.fromkeys(lexical_tokens(' '.join(TEST_SENTENCES))):
        if not word:
            continue
        corrector.clear_caches()
        candidates, cold_ms = timed(lambda: corrector.generate_candidates(word))
        _, warm_ms = timed(lambda: corrector.generate_candidates(word))
        token_rows.append(dict(token=word, cold_ms=cold_ms, warm_ms=warm_ms,
                               candidates=candidates))
    target_recall = {row['token']: TARGETS[row['token']] in dict(row['candidates'])
                     for row in token_rows if row['token'] in TARGETS}
    passed = (set(target_recall) == set(TARGETS) and all(target_recall.values())
              and all(row['stable_output'] and all(check['passed'] for check in row['checks']) for row in rows))
    return dict(passed=passed, sentences=rows, tokens=token_rows, target_candidate_recall=target_recall)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kenlm', default=KENLM_BIN)
    parser.add_argument('--dictionary', default=DICT_FILE)
    parser.add_argument('--confusion', default=CONFUSION_JSON)
    parser.add_argument('--backend', choices=('trie', 'symspell'), default='symspell')
    parser.add_argument('--symspell-distance', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--max-optical-cost', type=float, default=4.0)
    parser.add_argument('--json-report', type=Path)
    parser.add_argument('--trusted-long-word-frequency', type=int, default=1000)
    parser.add_argument('--trusted-short-word-frequency', type=int, default=4000)
    parser.add_argument('--allow-no-morphology', action='store_true',
                        help='Allow a diagnostic run without morphology (not a full Phase A validation)')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    for path in (args.kenlm, args.dictionary, args.confusion):
        if not Path(path).is_file():
            parser.error(f'Missing model/data file: {path}')
    options = dict(beam_width=4, lambda_optical=1.0, lambda_lm=1.8, lambda_morph=1.5,
                   candidate_backend=args.backend, symspell_max_edit_distance=args.symspell_distance,
                   max_optical_cost=args.max_optical_cost,
                   trusted_long_word_frequency=args.trusted_long_word_frequency,
                   trusted_short_word_frequency=args.trusted_short_word_frequency)
    start = time.perf_counter()
    corrector = AmharicOCRCorrector(args.kenlm, args.confusion, args.dictionary, **options)
    initialization_seconds = time.perf_counter() - start
    if corrector.segmenter is None and not args.allow_no_morphology:
        parser.error('Morphology is unavailable; install/expose the segmenter or use --allow-no-morphology')
    report = run_benchmark(corrector, args.repeats)
    report.update(backend=args.backend, options=options, initialization_seconds=initialization_seconds,
                  vocabulary_words=len(corrector.vocab), morphology_available=corrector.segmenter is not None,
                  peak_process_rss_mib=peak_rss_mib(),
                  inputs=dict(kenlm=args.kenlm, dictionary=args.dictionary, confusion=args.confusion))
    print(f"Initialization: {initialization_seconds:.2f}s; vocabulary: {len(corrector.vocab):,}; "
          f"peak process RSS: {report['peak_process_rss_mib']} MiB")
    for i, row in enumerate(report['sentences'], 1):
        print(f"Sentence {i}: cold median {row['cold_median_ms']:.2f}ms; "
              f"warm median {row['warm_median_ms']:.2f}ms")
        print('Deep analyzer calls (cold/warm):',
              [stats['segmenter_calls'] for stats in row['cold_morphology']], '/',
              [stats['segmenter_calls'] for stats in row['warm_morphology']])
        print('Deep analyzer ms (cold/warm):',
              [round(stats['segmenter_seconds'] * 1000, 2) for stats in row['cold_morphology']], '/',
              [round(stats['segmenter_seconds'] * 1000, 2) for stats in row['warm_morphology']])
        print('Trusted dictionary bypasses (cold/warm):',
              [stats['trusted_bypasses'] for stats in row['cold_morphology']], '/',
              [stats['trusted_bypasses'] for stats in row['warm_morphology']])
        print('CORRECTED:', row['output'])
        if row['non_target_changes'] or row['token_count_changed']:
            print('REVIEW non-target changes:', row['non_target_changes'],
                  'token count changed:', row['token_count_changed'])
        for check in row['checks'][:len(check_targets(row['input'], row['output']))]:
            print('PASS' if check['passed'] else 'FAIL', check)
    for row in report['tokens']:
        print(f"{row['token']}: cold {row['cold_ms']:.2f}ms; warm {row['warm_ms']:.2f}ms; "
              f"candidates: {row['candidates'][:5]}")
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('TARGET CHECKS PASSED' if report['passed'] else 'TARGET CHECKS FAILED')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
