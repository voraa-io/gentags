"""
Unit tests for tag filtering (word count constraints).
No API keys required.
"""

import pytest
from gentags.normalize import filter_valid_tags


def test_filter_valid_tags_within_limit():
    """Test that tags within word limit pass."""
    tags = ["food", "great food", "really great food", "very really great food"]
    valid, filtered = filter_valid_tags(tags, max_words=4)
    assert valid == tags
    assert filtered == []


def test_filter_valid_tags_exceeds_limit():
    """Test that tags exceeding word limit are filtered."""
    tags = ["food", "great food", "this is way too many words for a tag"]
    valid, filtered = filter_valid_tags(tags, max_words=4)
    assert valid == ["food", "great food"]
    assert filtered == ["this is way too many words for a tag"]


def test_filter_valid_tags_empty():
    """Test filtering empty list."""
    tags = []
    valid, filtered = filter_valid_tags(tags, max_words=4)
    assert valid == []
    assert filtered == []


def test_filter_valid_tags_exactly_at_limit():
    """Test tags exactly at the limit."""
    tags = ["one", "two words", "three word tag", "four word tag here"]
    valid, filtered = filter_valid_tags(tags, max_words=4)
    assert valid == tags
    assert filtered == []


def test_filter_valid_tags_one_over_limit():
    """Test tags one word over limit."""
    tags = ["one", "two words", "this is five words here now"]
    valid, filtered = filter_valid_tags(tags, max_words=4)
    assert valid == ["one", "two words"]
    assert filtered == ["this is five words here now"]


def test_filter_valid_tags_custom_max():
    """Test with custom max_words."""
    tags = ["one", "two words", "three word tag"]
    valid, filtered = filter_valid_tags(tags, max_words=2)
    assert valid == ["one", "two words"]
    assert filtered == ["three word tag"]

