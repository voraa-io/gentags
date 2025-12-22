"""
Unit tests for JSON parsing (extract_json_list).
No API keys required.
"""

import pytest
from gentags.parsing import extract_json_list


def test_parse_simple_list():
    """Test parsing a simple JSON list."""
    text = '["tag1", "tag2", "tag3"]'
    tags, status = extract_json_list(text)
    assert status == "success"
    assert tags == ["tag1", "tag2", "tag3"]


def test_parse_with_markdown():
    """Test parsing JSON wrapped in markdown code blocks."""
    text = '```json\n["tag1", "tag2"]\n```'
    tags, status = extract_json_list(text)
    assert status == "success"
    assert tags == ["tag1", "tag2"]


def test_parse_with_text_before():
    """Test parsing JSON list when there's text before it."""
    text = 'Here are the tags:\n["tag1", "tag2"]\nThat\'s all.'
    tags, status = extract_json_list(text)
    assert status == "success"
    assert tags == ["tag1", "tag2"]


def test_parse_empty_list():
    """Test parsing empty list."""
    text = "[]"
    tags, status = extract_json_list(text)
    assert status == "success"
    assert tags == []


def test_parse_error_invalid_json():
    """Test parse error for invalid JSON."""
    text = "not json at all"
    tags, status = extract_json_list(text)
    assert status == "parse_error"
    assert tags is None


def test_parse_error_empty_string():
    """Test parse error for empty string."""
    text = ""
    tags, status = extract_json_list(text)
    assert status == "parse_error"
    assert tags is None


def test_parse_strips_whitespace():
    """Test that tags are stripped of whitespace."""
    text = '["  tag1  ", " tag2 ", "tag3"]'
    tags, status = extract_json_list(text)
    assert status == "success"
    assert tags == ["tag1", "tag2", "tag3"]


def test_parse_filters_empty_strings():
    """Test that empty strings in list are filtered out."""
    text = '["tag1", "", "tag2", "   "]'
    tags, status = extract_json_list(text)
    assert status == "success"
    assert tags == ["tag1", "tag2"]

