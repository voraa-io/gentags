"""
Unit tests for tag normalization.
No API keys required.
"""

import pytest
from gentags.normalize import normalize_tag, normalize_tag_eval


def test_normalize_tag_lowercase():
    """Test basic lowercase normalization."""
    assert normalize_tag("Great Food") == "great food"
    assert normalize_tag("GREAT FOOD") == "great food"


def test_normalize_tag_strip_punctuation():
    """Test that leading/trailing punctuation is stripped."""
    assert normalize_tag("great food!") == "great food"
    assert normalize_tag("...great food...") == "great food"
    assert normalize_tag("'great food'") == "great food"


def test_normalize_tag_collapse_whitespace():
    """Test that multiple spaces are collapsed."""
    assert normalize_tag("great   food") == "great food"
    assert normalize_tag("great\tfood") == "great food"


def test_normalize_tag_preserves_internal_hyphens():
    """Test that internal hyphens are preserved."""
    assert normalize_tag("family-friendly") == "family-friendly"
    assert normalize_tag("coffee-shop") == "coffee-shop"


def test_normalize_tag_eval_removes_prefixes():
    """Test that common prefixes are removed."""
    assert normalize_tag_eval("very good") == "good"
    assert normalize_tag_eval("really nice") == "nice"
    assert normalize_tag_eval("quite cozy") == "cozy"
    assert normalize_tag_eval("super fast") == "fast"
    assert normalize_tag_eval("pretty good") == "good"
    assert normalize_tag_eval("gets crowded") == "crowded"


def test_normalize_tag_eval_plural_to_singular():
    """Test simple plural to singular conversion."""
    # Regular -s
    assert normalize_tag_eval("dishes") == "dish"
    assert normalize_tag_eval("tables") == "table"
    
    # -ies -> -y
    assert normalize_tag_eval("pastries") == "pastry"
    assert normalize_tag_eval("fries") == "fry"
    
    # -es after s/x/z
    assert normalize_tag_eval("boxes") == "box"
    assert normalize_tag_eval("dishes") == "dish"
    
    # Don't change -ss
    assert normalize_tag_eval("glass") == "glass"


def test_normalize_tag_eval_combined():
    """Test that prefix removal and plural handling work together."""
    assert normalize_tag_eval("very nice dishes") == "nice dish"
    assert normalize_tag_eval("really good pastries") == "good pastry"


def test_normalize_tag_eval_preserves_non_plural():
    """Test that non-plural words are preserved."""
    assert normalize_tag_eval("great food") == "great food"
    assert normalize_tag_eval("cozy atmosphere") == "cozy atmosphere"

