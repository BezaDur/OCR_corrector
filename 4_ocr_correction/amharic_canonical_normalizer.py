#!/usr/bin/env python3
"""
Centralized Canonical Normalizer for Amharic OCR & NLP Evaluation.
Standardizes punctuation equivalences (full stops, colons, semicolons, commas, quotes)
so neither evaluation metrics nor post-processors penalize keyboard approximations.
"""

import re
import unicodedata

# 1. Full Stop / Sentence Terminators: ::, ፡፡, .., etc. -> ።
RE_FULL_STOP = re.compile(r'([:፡]{2,}|\.{2,}|።+)')

# 2. Preface Colon: :-, :- -> ፦
RE_PREFACE_COLON = re.compile(r'([:፡]\s*[-–—]|፦+)')

# 3. Comma / Semicolon keyboard duplicates
RE_COMMA = re.compile(r'(,{2,}|፣{2,}|፣|,)')
RE_SEMICOLON = re.compile(r'(;{2,}|፤{2,}|፤|;)')

# 4. Quotes canonicalization: curly/guillemets -> "
RE_QUOTES = re.compile(r'[“”‘’«»‹›„‟\']')

# 5. Question marks
RE_QUESTION = re.compile(r'[?፧]+')

# 6. Exclamations
RE_EXCLAMATION = re.compile(r'[!‼]+')


def canonicalize_amharic_punctuation(text: str) -> str:
    """
    Applies deterministic canonical punctuation equivalence:
      - Full stops (::, ፡፡, .., ።) -> ።
      - Semicolons (;, ፤) -> ፤
      - Commas (,, ፣) -> ፣
      - Preface Colons (:-, ፦) -> ፦
      - Colons (፥, single :) -> ፥
      - Quotes -> "
      - Inter-word stray single '፡' -> whitespace
    """
    if not text:
        return ""

    # Ensure Unicode NFC standard
    t = unicodedata.normalize("NFC", text)

    # Multi-char full-stops & keyboard approximations first
    t = RE_FULL_STOP.sub('።', t)
    t = RE_PREFACE_COLON.sub('፦', t)

    # Standardize commas and semicolons
    t = RE_COMMA.sub('፣', t)
    t = RE_SEMICOLON.sub('፤', t)

    # Standardize quotes
    t = RE_QUOTES.sub('"', t)

    # Standardize questions and exclamations
    t = RE_QUESTION.sub('?', t)
    t = RE_EXCLAMATION.sub('!', t)

    # Trailing colons/hulet-neteb at boundary -> ።
    t = re.sub(r'[:፡]\s*$', '።', t, flags=re.MULTILINE)
    t = re.sub(r'[:፡]\s*(["\n])', r'።\1', t)

    # Single colon between numbers or words -> ፥
    t = re.sub(r'([0-9])\s*[:፡]\s*([0-9])', r'\1፥\2', t)

    # Inter-word isolated single '፡' -> clean space
    t = re.sub(r'[:፡]', ' ', t)

    # Collapse multiple whitespace
    t = re.sub(r'[ ]+', ' ', t)

    return t.strip()


def normalize_for_metric(text: str) -> str:
    """
    Prepares text for CER / WER evaluation:
    Canonicalizes punctuation and strips boundary whitespace.
    """
    return canonicalize_amharic_punctuation(text)