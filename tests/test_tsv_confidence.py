"""TSV word confidences: parsing, alignment to text lines, mismatch handling."""
import os
import tempfile
import unittest

from tsv_confidence import align_confidences, read_tsv_lines

HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
ROWS = [
    "1\t1\t0\t0\t0\t0\t0\t0\t1654\t2338\t-1\t",
    "2\t1\t1\t0\t0\t0\t123\t132\t1311\t154\t-1\t",
    "4\t1\t1\t1\t1\t0\t124\t132\t1118\t28\t-1\t",
    "5\t1\t1\t1\t1\t1\t124\t140\t117\t20\t91.658699\tየኢትዮጵያ፣",
    "5\t1\t1\t1\t1\t2\t253\t140\t51\t17\t92.111366\tግብጽ",
    "5\t1\t1\t1\t1\t3\t314\t140\t29\t16\t86.271027\tእና",
    "5\t1\t1\t1\t2\t1\t124\t160\t60\t17\t40.000000\tሱዳን",
    "5\t1\t1\t1\t2\t2\t200\t160\t50\t17\t0.000000\t \t",
]
LINE1 = "የኢትዮጵያ፣ ግብጽ እና"
LINE2 = "ሱዳን"


class TestTsvConfidence(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".tsv")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(HEADER + "\n".join(ROWS) + "\n")

    def tearDown(self):
        os.remove(self.path)

    def test_containers_and_blank_tokens_are_dropped(self):
        groups = read_tsv_lines(self.path)
        self.assertEqual([len(g) for g in groups], [3, 1])
        self.assertEqual([w.text for w in groups[0]], ["የኢትዮጵያ፣", "ግብጽ", "እና"])

    def test_confidences_are_kept_on_0_100_scale(self):
        groups = read_tsv_lines(self.path)
        self.assertAlmostEqual(groups[0][0].conf, 91.658699)
        self.assertAlmostEqual(groups[1][0].conf, 40.0)

    def test_matching_lines_get_their_confidences(self):
        res, stats = align_confidences(self.path, [LINE1, LINE2])
        self.assertEqual(len(res[0]), 3)
        self.assertEqual(len(res[1]), 1)
        self.assertEqual(stats, {"matched": 2, "mismatched": 0, "empty": 0})

    def test_word_count_mismatch_gives_none_not_a_guess(self):
        res, stats = align_confidences(self.path, ["የኢትዮጵያ ግብጽ", LINE2])
        self.assertIsNone(res[0])
        self.assertEqual(stats["mismatched"], 1)

    def test_empty_text_line_is_skipped(self):
        res, stats = align_confidences(self.path, ["", LINE1])
        self.assertIsNone(res[0])
        self.assertEqual(stats["empty"], 1)
        self.assertEqual(len(res[1]), 3)


if __name__ == "__main__":
    unittest.main()
