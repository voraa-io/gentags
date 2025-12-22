"""
JSON parsing: extract_json_list().
"""

import json
import re
from typing import Optional, List, Tuple


def extract_json_list(text: str) -> Tuple[Optional[List[str]], str]:
    """
    Extract JSON list from model output.
    
    Returns:
        (tags, status): tags is list or None, status is "success" or "parse_error"
    """
    if not text:
        return None, "parse_error"
    
    text_stripped = text.strip()
    
    # Strategy 1: Try parsing entire text as JSON list
    try:
        parsed = json.loads(text_stripped)
        if isinstance(parsed, list):
            tags = [str(t).strip() for t in parsed if t and str(t).strip()]
            return tags, "success"
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Strip markdown code blocks and try again
    content = re.sub(r"^```(?:json)?\s*", "", text_stripped, flags=re.MULTILINE)
    content = re.sub(r"\s*```$", "", content, flags=re.MULTILINE)
    try:
        parsed = json.loads(content.strip())
        if isinstance(parsed, list):
            tags = [str(t).strip() for t in parsed if t and str(t).strip()]
            return tags, "success"
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Find the FIRST complete JSON array (greedy match for balanced brackets)
    # This is more robust than non-greedy regex
    bracket_start = text_stripped.find('[')
    if bracket_start != -1:
        depth = 0
        for i, char in enumerate(text_stripped[bracket_start:], start=bracket_start):
            if char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
                if depth == 0:
                    candidate = text_stripped[bracket_start:i+1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, list):
                            tags = [str(t).strip() for t in parsed if t and str(t).strip()]
                            return tags, "success"
                    except json.JSONDecodeError:
                        pass
                    break
    
    # All strategies failed
    return None, "parse_error"

