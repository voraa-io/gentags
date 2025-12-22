"""
Tag normalization: normalize_tag(), normalize_tag_eval(), filter_valid_tags().
"""

import re
from typing import List, Tuple
from .config import MAX_TAG_WORDS


def normalize_tag(tag: str) -> str:
    """
    Light normalization for evaluation (tag_norm).
    - Lowercase
    - Strip punctuation (except internal hyphens/apostrophes)
    - Collapse whitespace
    """
    tag = tag.lower().strip()
    # Remove leading/trailing punctuation
    tag = re.sub(r'^[^\w]+|[^\w]+$', '', tag)
    # Collapse internal whitespace
    tag = re.sub(r'\s+', ' ', tag)
    return tag


def normalize_tag_eval(tag: str) -> str:
    """
    Stricter normalization for stability metrics (tag_norm_eval).
    - All of normalize_tag() plus:
    - Simple plural → singular
    - Remove common prefixes like "gets", "very", "really"
    """
    tag = normalize_tag(tag)
    
    # Remove common prefixes that don't change meaning
    prefixes = ['gets ', 'very ', 'really ', 'quite ', 'super ', 'pretty ']
    for prefix in prefixes:
        if tag.startswith(prefix):
            tag = tag[len(prefix):]
    
    # Simple plural handling
    words = tag.split()
    normalized_words = []
    for word in words:
        if len(word) > 3:
            # Handle -ies → -y (pastries → pastry, fries → fry)
            if word.endswith('ies'):
                normalized_words.append(word[:-3] + 'y')
            # Handle -es after s/x/z/ch/sh (dishes → dish, boxes → box)
            elif word.endswith('es') and len(word) > 4 and word[-3] in 'sxz':
                normalized_words.append(word[:-2])
            elif word.endswith('ches') or word.endswith('shes'):
                normalized_words.append(word[:-2])
            # Handle regular -s (but not -ss like "glass")
            elif word.endswith('s') and not word.endswith('ss'):
                normalized_words.append(word[:-1])
            else:
                normalized_words.append(word)
        else:
            normalized_words.append(word)
    
    return ' '.join(normalized_words)


def filter_valid_tags(tags: List[str], max_words: int = MAX_TAG_WORDS) -> Tuple[List[str], List[str]]:
    """
    Filter tags by word count constraint.
    
    Returns:
        (valid_tags, filtered_out_tags)
    """
    valid = []
    filtered = []
    for tag in tags:
        word_count = len(tag.split())
        if 1 <= word_count <= max_words:
            valid.append(tag)
        else:
            filtered.append(tag)
    return valid, filtered

