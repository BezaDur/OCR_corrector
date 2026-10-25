"""Tests for the pre-merge checking tools.

Covers ``scripts/make_eval_corpus.py`` (the labelling recipe that turns a new
Evaluation/-style folder into a scorable corpus), the ``--eval-all`` mode of
``scripts/evaluate_out_of_sample.py`` and ``document_groups`` in the trainer.
None of these touch the real corpus, so they stay fast.
"""
import os
import sys
import tempfile
import unittest

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                    # 4_ocr_correction/error_detection/
_STAGE = os.path.dirname(_PKG)                   # 4_ocr_correction/
_ROOT = os.path.dirname(_STAGE)                  # repo root
for _p in (_ROOT, _STAGE, _PKG, os.path.join(_PKG, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_eval_corpus as mec  # noqa: E402
import evaluate_out_of_sample as eos  # noqa: E402


class TestAlignmentRows(unittest.TestCase):
    def test_match_substitution_insertion_and_deletion(self):
        # GT:       a b c d
        # OCR:      a x c d e   -> b->x substitution, e insertion, nothing dropped
        rows, stats = mec.align_page("a b c d", "a x c d e")
        self.assertEqual([r[3] for r in rows], ["MATCH", "SUBSTITUTION", "MATCH", "MATCH", "INSERTION"])
        self.assertEqual(rows[1][1], "b")
        self.assertIsNone(rows[4][1])
        self.assertEqual([r[2] for r in rows], [0, 1, 0, 0, 1])
        self.assertEqual(stats["deletions"], 0)

    def test_deleted_gt_word_is_not_scored(self):
        rows, stats = mec.align_page("a b c", "a c")
        self.assertEqual(stats["deletions"], 1)
        self.assertEqual([r[0] for r in rows], ["a", "c"])
        self.assertTrue(all(r[2] == 0 for r in rows))

    def test_punctuation_and_digits_are_stripped(self):
        rows, _ = mec.align_page("ሰላም፣ 2019 ዓመት", "ሰላም፣ 2019 ዓመት")
        self.assertEqual([r[0] for r in rows], ["ሰላም", "ዓመት"])


class TestBuildCorpus(unittest.TestCase):
    def _make_dir(self, pages):
        """pages: {stem: (gt_text, ocr_text, conf_or_None)}"""
        tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmp, "GroundTruth"))
        os.makedirs(os.path.join(tmp, "OCRed"))
        for stem, (gt, ocr, conf) in pages.items():
            with open(os.path.join(tmp, "GroundTruth", f"{stem}.gt.txt"), "w", encoding="utf-8") as fh:
                fh.write(gt)
            with open(os.path.join(tmp, "OCRed", f"{stem}.ocr.txt"), "w", encoding="utf-8") as fh:
                fh.write(ocr)
            if conf:
                with open(os.path.join(tmp, "OCRed", f"{stem}.conf.tsv"), "w", encoding="utf-8") as fh:
                    fh.write("word\tconfidence\n" + conf)
        return tmp

    def test_rows_doc_freq_and_pages(self):
        d = self._make_dir({
            "a": ("እና እና ሰላም", "እና እና ሰላም", None),
            "b": ("አንድ ሁለት", "አንድ ሶስት", None),
        })
        df, info = mec.build(d, quiet=True)
        self.assertEqual(info["pages"], 2)
        self.assertEqual(len(df), 5)
        self.assertEqual(info["errors"], 1)
        # doc_freq is the within-page count of the OCR word, not a corpus frequency
        first = df[df.page == "page_0001"]
        self.assertEqual(first[first.ocr_word == "እና"].doc_freq.tolist(), [2, 2])
        self.assertEqual(first[first.ocr_word == "ሰላም"].doc_freq.tolist(), [1])

    def test_confidence_read_from_conf_tsv(self):
        d = self._make_dir({"a": ("ሰላም", "ሰላም", "ሰላም፣\t91.5\n")})
        df, _ = mec.build(d, quiet=True)
        # the conf file keys on the punctuated token; it must still match the clean word
        self.assertAlmostEqual(float(df.tesseract_conf.iloc[0]), 91.5)

    def test_novelty_detects_reused_text(self):
        d = self._make_dir({
            "a": ("አንድ ሁለት ሶስት አራት አምስት ስድስት", "አንድ ሁለት ሶስት አራት አምስት ስድስት", None),
        })
        df, _ = mec.build(d, quiet=True)
        ref = os.path.join(tempfile.mkdtemp(), "ref.tsv")
        pd.DataFrame({
            "page": ["page_0001"] * 6 + ["page_0999"] * 6,
            "ocr_word": (["አንድ", "ሁለት", "ሶስት", "አራት", "አምስት", "ስድስት"] * 2),
            "gt_word": (["አንድ", "ሁለት", "ሶስት", "አራት", "አምስት", "ስድስት"] * 2),
            "is_error": [0] * 12,
            "match_type": ["MATCH"] * 12,
            "tesseract_conf": [75.0] * 12,
            "doc_freq": [1] * 12,
        }).to_csv(ref, sep="\t", index=False)
        out = mec.novelty(df, reference=ref, dev_pages=0, quiet=True)
        # everything here is also on the reference page -> no new vocabulary
        self.assertEqual(out["share_tokens_unseen"], 0.0)


class TestEvalAll(unittest.TestCase):
    def test_eval_all_returns_every_row(self):
        path = os.path.join(tempfile.mkdtemp(), "c.tsv")
        pd.DataFrame({
            "page": ["page_0001", "page_0001", "page_0500"],
            "ocr_word": ["a", "b", "c"],
            "gt_word": ["a", "x", "c"],
            "is_error": [0, 1, 0],
            "match_type": ["MATCH", "SUBSTITUTION", "MATCH"],
            "tesseract_conf": [75.0] * 3,
            "doc_freq": [1, 1, 1],
        }).to_csv(path, sep="\t", index=False)
        dev, prod = eos.load_corpus(path, 100, eval_all=True)
        self.assertEqual(len(dev), 3)
        self.assertEqual(len(prod), 0)
        dev2, prod2 = eos.load_corpus(path, 100)
        self.assertEqual(len(dev2), 2)
        self.assertEqual(len(prod2), 1)


class TestDocumentGroups(unittest.TestCase):
    def test_one_document_and_boundary(self):
        from Model.train_LightGBM import document_groups
        words_a = [f"w{i}" for i in range(30)]
        words_b = [f"v{i}" for i in range(30)]
        rows = []
        for page, seq in (("page_0001", words_a[:20]), ("page_0002", words_a[10:30]),
                          ("page_0003", words_b)):
            for w in seq:
                rows.append({"page": page, "ocr_word": w, "gt_word": w})
        df = pd.DataFrame(rows)
        g = document_groups(df)
        self.assertEqual(g["page_0001"], g["page_0002"])
        self.assertNotEqual(g["page_0002"], g["page_0003"])


if __name__ == "__main__":
    unittest.main()
