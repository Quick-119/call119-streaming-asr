"""評価で採用した限定補正の境界テスト。"""
import unittest

from postprocess import correct_stream_parts, correct_transcript


class PostprocessTest(unittest.TestCase):
    def test_medical_context(self):
        self.assertEqual(correct_transcript("起用歴はなく転換の症状がある"),
                         "既往歴はなくてんかんの症状がある")
        self.assertEqual(correct_transcript("全解除で生活している"), "全介助で生活している")

    def test_mock_error_context(self):
        self.assertEqual(correct_transcript("建物の2体から建物の回数を教える"),
                         "建物の2階から建物の階数を教える")
        self.assertEqual(correct_transcript("統計道でお腹を抑えている"),
                         "峠道でお腹を押さえている")
        self.assertEqual(correct_transcript("出血していて糖尿が突き沿っている"),
                         "出血していて同僚が付き添っている")

    def test_no_unrelated_changes(self):
        text = "もしもし、怪我をした子供が別れ道にいます。転換点です。"
        self.assertEqual(correct_transcript(text), text)

    def test_streaming_boundary(self):
        self.assertEqual(correct_stream_parts("起", "用歴はない"), ("起", "用歴はない"))
        self.assertEqual(correct_stream_parts("起用歴", "はない"), ("既往歴", "はない"))
        self.assertEqual(correct_stream_parts("出血していて糖尿が", "突き沿っている"),
                         ("出血していて糖尿が", "付き添っている"))

    def test_idempotent(self):
        text = "起用歴、建物の2体から、統計道"
        self.assertEqual(correct_transcript(correct_transcript(text)),
                         correct_transcript(text))


if __name__ == "__main__":
    unittest.main()
