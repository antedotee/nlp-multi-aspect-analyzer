"""Regression tests for aspect-chip grouping. Run: python -m pytest tests -q"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infer import AspectSpan, cluster_chips, get_grouper, normalize_aspect  # noqa: E402


def mentions(*phrases, sentiment="POS"):
    """One mention per phrase, each in its own review."""
    return [(i, AspectSpan(p, 0, len(p), sentiment, 0.9)) for i, p in enumerate(phrases)]


def chip_of(chips, phrase):
    key = phrase.strip().lower()
    return next(c for c in chips if c.aspect == key or key in c.aspect)


@pytest.mark.parametrize("raw, key", [
    ("Battery life", "battery"),
    ("the battery", "battery"),
    ("Sound Quality", "sound"),
    ("speakers", "speaker"),
    ("batteries", "battery"),
    ("glasses", "glass"),
    ("bass", "bass"),
    ("Windows", "windows"),
    ("series", "series"),
    ("life", "life"),                       # a lone attribute word is kept
    ("screen protector", "screen protector"),
    ("  “Ear cushions” ", "ear cushion"),
])
def test_normalize_aspect(raw, key):
    assert normalize_aspect(raw) == key


def test_fallback_groups_exact_normalized_keys_only():
    chips = cluster_chips(mentions("Battery life", "battery", "price", "cost"), grouper=None)
    assert sorted((c.aspect, c.total) for c in chips) == [("battery", 2), ("cost", 1), ("price", 1)]


def test_counts_and_naming_are_consistent():
    ms = mentions("battery", "Battery", "battery life", sentiment="POS") + [
        (7, AspectSpan("battery", 0, 7, "NEG", 0.8)),
        (8, AspectSpan("battery", 0, 7, "NEU", 0.8)),
    ]
    (chip,) = cluster_chips(ms, grouper=None)
    assert chip.aspect == "battery"           # most frequent surface form names the chip
    assert (chip.pos, chip.neg, chip.neu, chip.total) == (3, 1, 1, 5)
    assert chip.review_ids == [0, 1, 2, 7, 8]


@pytest.fixture(scope="module")
def grouper():
    g = get_grouper()
    if g is None:
        pytest.skip("grouping model unavailable (offline)")
    return g


@pytest.mark.parametrize("a, b", [
    ("battery", "battery life"),
    ("price", "cost"),
    ("audio", "sound"),
    ("sound", "sound quality"),
    ("screen", "display"),
    ("customer service", "customer support"),
    ("shipping", "delivery"),
    ("noise cancelling", "noise cancellation"),
])
def test_synonyms_merge(grouper, a, b):
    chips = cluster_chips(mentions(a, b), grouper=grouper)
    assert len(chips) == 1, f"{a!r} and {b!r} should share a chip: {chips}"


@pytest.mark.parametrize("a, b", [
    ("screen", "screen protector"),
    ("battery", "battery case"),
    ("battery", "charger"),
    ("sound", "microphone"),
    ("price", "shipping"),
    ("customer service", "warranty"),
    ("trackpad", "keyboard"),
])
def test_distinct_aspects_stay_apart(grouper, a, b):
    chips = cluster_chips(mentions(a, b), grouper=grouper)
    assert len(chips) == 2, f"{a!r} and {b!r} must stay separate: {chips}"


def test_mixed_review_set(grouper):
    ms = mentions("battery life", "Battery", "price", "cost", "screen", "screen protector", "audio", "sound")
    chips = cluster_chips(ms, grouper=grouper)
    assert sum(c.total for c in chips) == len(ms)
    assert chip_of(chips, "battery").total == 2
    assert len(chips) == 5  # battery, price/cost, screen, screen protector, sound/audio
