"""CER で通話単位の悪化を確認した、限定的な ASR 文字起こし補正。"""
from __future__ import annotations

import re


# 正規表現は実通報の既存評価と公開模擬通報で個別検証する。
# 表記揺れの一括置換は、実通報の正解表記を壊すため含めない。
_RULES = (
    (re.compile(r"起用歴"), "既往歴"),
    (re.compile(r"転換(?=の症状)"), "てんかん"),
    (re.compile(r"(?<=全)解除(?=で|が|に)"), "介助"),
    (re.compile(r"(?<=建物の)2体(?=から)"), "2階"),
    (re.compile(r"(?<=建物の)回数(?=を)"), "階数"),
    (re.compile(r"先時期(?=の枯草)"), "河川敷"),
    (re.compile(r"統計道"), "峠道"),
    (re.compile(r"(?<=お腹を)抑え"), "押さえ"),
    (re.compile(r"(?<=出血していて)糖尿(?=が突き沿)"), "同僚"),
    (re.compile(r"突き沿(?=って)"), "付き添"),
)


def correct_transcript(text: str) -> str:
    """累積文字起こしを補正する。確定・部分本文の結合後にも適用する。"""
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text


def correct_stream_parts(confirmed: str, partial: str) -> tuple[str, str]:
    """累積表示用の確定・部分本文を整合したまま補正する。"""
    fixed_confirmed = correct_transcript(confirmed)
    fixed_whole = correct_transcript(confirmed + partial)
    if fixed_whole.startswith(fixed_confirmed):
        return fixed_confirmed, fixed_whole[len(fixed_confirmed):]
    # 規則が境界をまたぎ、確定部分まで遡る場合は次回の確定を待つ。
    return fixed_confirmed, correct_transcript(partial)
