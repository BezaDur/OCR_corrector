import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('evaluator', ROOT / '04_evaluate_full_corrector.py')
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class IdentityCorrector:
    def correct_text_block(self, text):
        return text


class EvaluatorTests(unittest.TestCase):
    def test_distances(self):
        for left, right, expected in [('', '', 0), ('abc', '', 3), ('abc', 'axc', 1),
                                      ('በሸታ', 'በሽታ', 1), (['ቤት', 'ነው'], ['ቤት'], 1)]:
            self.assertEqual(evaluator.dp_edit_distance(left, right), expected)
            self.assertEqual(evaluator.dp_edit_distance(right, left), expected)

    def test_page_metrics_and_output(self):
        with tempfile.TemporaryDirectory() as folder:
            gt = Path(folder) / 'p.gt.txt'
            raw = Path(folder) / 'p.ocr.txt'
            gt.write_text('ቤት፡፡', encoding='utf-8')
            raw.write_text('ቤት።', encoding='utf-8')
            with patch.multiple(evaluator, RAW_OCR_DIR=folder, CORRECTED_OUT_DIR=folder,
                                _corrector=IdentityCorrector(), CANONICAL_METRICS=True), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = evaluator.process_single_page(str(gt))
            self.assertEqual(result[:6], (3, 0, 0, 1, 0, 0))
            self.assertGreaterEqual(result[6], 0)
            self.assertEqual(result[7], 1)
            self.assertEqual((Path(folder) / 'p.corr.txt').read_text(), 'ቤት።\n')

    def test_worker_uses_same_configuration(self):
        options = {'candidate_backend': 'symspell', 'max_optical_cost': 3.5}
        config = ('lm', 'matrix', 'dict', 'raw', 'out', options, True)
        with patch.multiple(evaluator, _corrector=None, KENLM_BIN=None, CONFUSION_JSON=None,
                            DICT_FILE=None, RAW_OCR_DIR=None, CORRECTED_OUT_DIR=None,
                            CORRECTOR_OPTIONS={}, CANONICAL_METRICS=False), \
                patch.object(evaluator, 'AmharicOCRCorrector') as constructor:
            evaluator.init_worker(config)
            constructor.assert_called_once_with(kenlm_path='lm', confusion_matrix_path='matrix',
                                                dict_vocab_path='dict', **options)
            self.assertTrue(evaluator.CANONICAL_METRICS)

    def test_missing_raw_fails(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(evaluator, 'RAW_OCR_DIR', folder):
            with self.assertRaises(FileNotFoundError):
                evaluator.process_single_page(str(Path(folder) / 'missing.gt.txt'))


if __name__ == '__main__':
    unittest.main()
