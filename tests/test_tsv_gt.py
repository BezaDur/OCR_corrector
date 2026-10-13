import unittest

from tsv_gt import annotate_rows


def word(text, conf, n):
    return {"level": "5", "block_num": "1", "par_num": "1", "line_num": "1", "word_num": str(n),
            "conf": str(conf), "text": text}


class TestTsvGt(unittest.TestCase):
    def test_correct_and_wrong_words_get_gt_columns(self):
        rows = [{"level": "2", "conf": "-1", "text": ""},
                word("ሰላም", 90, 1), word("ቤቱ", 20, 2),
                {"level": "5", "conf": "-1", "text": "", "word_num": "3"}]
        out = annotate_rows(rows, "ሰላም ቤት")
        self.assertEqual(out[1]["core"], "ሰላም")
        self.assertEqual((out[1]["gt_word"], out[1]["gt_status"]), ("ሰላም", "clean"))
        self.assertEqual((out[2]["gt_word"], out[2]["gt_status"]), ("ቤት", "error"))
        self.assertEqual(out[0]["gt_status"], "")
        self.assertEqual(out[3]["gt_status"], "")

    def test_wrong_word_at_a_clean_word_position_is_error(self):
        # OCR "X B" against gold "A B C": X matches nothing, B matches gold B.
        rows = [word("XXX", 40, 1), word("ቤት", 95, 2)]
        out = annotate_rows(rows, "ሰላም ቤት ሰላም")
        self.assertEqual(out[0]["gt_status"], "error")
        # The wrong word is paired with the gold word it replaced (ሰላም) in one traceback.
        self.assertEqual(out[0]["gt_word"], "ሰላም")
        self.assertEqual((out[1]["gt_word"], out[1]["gt_status"]), ("ቤት", "clean"))


if __name__ == "__main__":
    unittest.main()
