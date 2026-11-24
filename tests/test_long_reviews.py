"""Long reviews must be tagged in full, never silently truncated. Run: python -m pytest tests -q"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infer import LOCAL_MODEL_DIR, check, get_highlighter, sentence_spans  # noqa: E402


@pytest.mark.parametrize("text, expected", [
    ("Great screen. Bad battery!", ["Great screen.", "Bad battery!"]),
    ("Line one\nLine two", ["Line one", "Line two"]),
    ("It costs 4.5 dollars. Fine.", ["It costs 4.5 dollars.", "Fine."]),
    ("  Wow!!  Really?  ", ["Wow!!", "Really?"]),
    ("No punctuation at all", ["No punctuation at all"]),
    ("", []),
])
def test_sentence_spans(text, expected):
    assert [text[s:e] for s, e in sentence_spans(text)] == expected


def test_sentence_spans_cover_every_word():
    text = "First part. Second part!\n\nThird part? Fourth"
    covered = "".join(text[s:e] for s, e in sentence_spans(text))
    assert covered.replace(" ", "") == text.replace(" ", "").replace("\n", "")


@pytest.fixture(scope="module")
def highlighter():
    if not (LOCAL_MODEL_DIR / "config.json").is_file():
        pytest.skip("no local model in artifacts/absa-model (see README: Quick start)")
    return get_highlighter(str(LOCAL_MODEL_DIR))


def tokens(h, text):
    return len(h.tokenizer(text)["input_ids"])


def test_aspect_after_token_512_is_found(highlighter):
    review = "Filler sentence about nothing in particular. " * 120 + "The battery is terrible."
    assert tokens(highlighter, review) > 900
    result = highlighter.highlight_reviews([review])
    check(result, [review])
    spans = [(s.text, s.sentiment) for s in result.reviews[0].spans]
    assert ("battery", "NEG") in spans
    assert len(spans) <= 3, f"filler text should not produce many aspects: {spans}"


def test_short_review_is_one_chunk(highlighter):
    review = "Great screen. The keyboard is awful! Battery life is amazing."
    assert highlighter._segments(review, 512) == [(0, len(review))]


def test_single_sentence_longer_than_model_limit(highlighter):
    review = "and then " * 400 + "the battery is terrible"  # one sentence, no punctuation
    assert tokens(highlighter, review) > 600
    result = highlighter.highlight_reviews([review])
    check(result, [review])
    assert any(s.text == "battery" for s in result.reviews[0].spans)


def test_offsets_point_into_original_review(highlighter):
    reviews = ["Great screen.  The keyboard is awful!\nBattery life is amazing.", "", "Shipping was slow."]
    result = highlighter.highlight_reviews(reviews)
    check(result, reviews)
    got = {s.text for r in result.reviews for s in r.spans}
    assert {"screen", "keyboard", "Battery life", "Shipping"} <= got
