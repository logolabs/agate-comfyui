"""Prompt/caption normalisation for Agate, applied identically in training and at inference, plus the count
parser that feeds the count code (models/fcdm_thinker2_mr.CountCode). Measured on step 138,010
(tools/text_probe.py, 2026-09-26; thinker plan distance, 1.0 = a new noise seed, object swap 1.17):

  ALL CAPS prompt 0.89 and typos 0.80 move the plan like a colour change; Title Case 0.43; count changes
  only 0.57 and quoted-text changes 0.37-0.54. So:
  * whitespace is collapsed (harmless; robust already)
  * casing outside quotes is normalised ONLY when a prompt is shouted (mostly ALL CAPS) or Title Cased; a
    normal prompt keeps its proper nouns. Text inside double quotes is never touched: it is what gets drawn
  * number words and small digits become one canonical lowercase word ("THREE", "Three", "3" -> "three"),
    and their value + character span are returned for the count code
  * typo augmentation (training only) never touches quotes or number words
  * "without X" / "no X" (inference only) move to the negative prompt: the thinker barely reacts to
    negation (0.45) while CFG pushes away from a negative prompt reliably
"""
from __future__ import annotations

import random
import re

NUM_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
NUM_WORDS.update({"dozen": 12})
WORD_OF = {v: k for k, v in NUM_WORDS.items() if k != "dozen"}
# a number followed by one of these is not an object count ("2 pm", "24 hours", "3 of the")
NOT_COUNT_NEXT = {"pm", "am", "o'clock", "percent", "%", "years", "year", "hours", "hour", "minutes", "minute",
                  "seconds", "second", "times", "px", "cm", "mm", "m", "km", "kg", "g", "inch", "inches", "feet",
                  "degrees", "x", "d", "k", "th", "st", "nd", "rd", "of", "o"}
# "one" used as a pronoun or ordinal context ("the one", "no one", "each one")
ONE_PRONOUN_PREV = {"the", "this", "that", "no", "each", "every", "any", "some", "which", "another", "someone",
                    "everyone", "anyone", "only"}
QUOTE_RE = re.compile(r'"[^"]*"|“[^”]*”')
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+")


def _outside_quotes(text: str) -> list[tuple[int, int]]:
    """Character ranges NOT inside double (or curly) quotes."""
    spans, last = [], 0
    for m in QUOTE_RE.finditer(text):
        spans.append((last, m.start()))
        last = m.end()
    spans.append((last, len(text)))
    return [(a, b) for a, b in spans if b > a]


def _map_outside(text: str, fn) -> str:
    out, last = [], 0
    for a, b in _outside_quotes(text):
        out.append(text[last:a])
        out.append(fn(text[a:b]))
        last = b
    out.append(text[last:])
    return "".join(out)


def _fix_case(seg_words: list[str]) -> str | None:
    """'shout' when >= 60% of the alphabetic words (2+ letters) are ALL CAPS, 'title' when >= 70% of the
    3+ letter words are Titlecased (and there are at least 3); None otherwise."""
    long2 = [w for w in seg_words if len(w) >= 2 and w.isalpha()]
    if long2 and sum(w.isupper() for w in long2) / len(long2) >= 0.6:
        return "shout"
    long3 = [w for w in seg_words if len(w) >= 3 and w.isalpha()]
    if len(long3) >= 3 and sum(w[0].isupper() and w[1:].islower() for w in long3) / len(long3) >= 0.7:
        return "title"
    return None


def is_json_caption(text: str) -> bool:
    """A schema-JSON caption (the LogoLabs logo set's long captions: one JSON object). Every function here
    leaves these untouched: in JSON every key and value sits in double quotes, so the quote-based rules
    misfire -- measured 2026-09-28 on 3,994 logo captions: add_spelling appended the spelled KEYS
    ("h i g h l e v e l d e s c r i p t i o n ; ...") to 100% of them, and escaped quotes (\\"FONTS\\") flipped
    the in/out-of-quotes parity so normalize lowercased the text to draw in 7% of them."""
    s = text.split(SPELL_SEP)[0].strip()
    if not (s.startswith("{") and s.endswith("}")):
        return False
    try:
        import json
        return isinstance(json.loads(s), dict)
    except ValueError:
        return False


def normalize(text: str) -> str:
    """Whitespace, shouted/Title casing outside quotes, canonical number words. Idempotent. JSON captions
    are returned unchanged (is_json_caption)."""
    if is_json_caption(text):
        return text
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for a, b in _outside_quotes(text) for w in WORD_RE.findall(text[a:b])]
    mode = _fix_case(words)
    if mode == "shout":
        text = _map_outside(text, lambda s: s.lower())
    elif mode == "title":                          # every Titlecased word but the first -> lowercase
        first = re.match(r"\s*\S+", text)
        head = first.end() if first else 0
        text = text[:head] + _map_outside(text[head:], lambda s: re.sub(r"\b[A-Z][a-z]*\b", lambda m: m.group(0).lower(), s))
    return _map_outside(text, _canon_numbers)


def _canon_numbers(seg: str) -> str:
    def rep(m):
        w = m.group(0)
        lw = w.lower()
        if lw in NUM_WORDS:
            return lw
        if w.isdigit() and 1 <= int(w) <= 20 and _is_count_context(seg, m.end()):
            return WORD_OF[int(w)]
        return w
    return re.sub(r"\b[A-Za-z]+\b|\b\d+\b", rep, seg)


def _is_count_context(seg: str, end: int) -> bool:
    nxt = re.match(r"\s*([A-Za-z%']+)", seg[end:])
    return bool(nxt) and nxt.group(1).lower() not in NOT_COUNT_NEXT


def find_counts(text: str) -> list[tuple[int, int, int]]:
    """(start, end, n) of every object count outside quotes: number words / 1-20 digits followed by a
    word that is not a unit or 'of', 'one' not used as a pronoun, 'a dozen' / 'a pair of' included. The
    spelled-out suffix (after SPELL_SEP) is never parsed: a spelled '7' is a character, not a count."""
    out = []
    if is_json_caption(text):
        return out
    text = text.split(SPELL_SEP)[0]
    for a, b in _outside_quotes(text):
        seg = text[a:b]
        for m in re.finditer(r"\b(?:a\s+)?(dozen|pair(?=\s+of\b))\b|\b([A-Za-z]+|\d+)\b", seg):
            if m.group(1):                                        # "a dozen", "a pair of"
                n = 12 if m.group(1).lower() == "dozen" else 2
                out.append((a + m.start(1), a + m.end(1), n))
                continue
            w = m.group(2)
            lw = w.lower()
            if lw.isdigit():
                n = int(lw)
                if not 1 <= n <= 99:
                    continue
            elif lw in NUM_WORDS and lw not in ("zero", "dozen"):
                n = NUM_WORDS[lw]
            else:
                continue
            if not _is_count_context(seg, m.end()):
                continue
            prev = re.findall(r"[A-Za-z]+", seg[:m.start()])
            if lw == "one" and prev and prev[-1].lower() in ONE_PRONOUN_PREV:
                continue
            out.append((a + m.start(2), a + m.end(2), n))
    return out


SPELL_SEP = " || spell: "
SPELL_MAX_FRAC, SPELL_MAX_CHARS, SPELL_MAX_WORDS, SPELL_MAX_SPANS = 0.6, 40, 6, 3


def text_spans(text: str) -> list[str]:
    """The quoted spans that are text to DRAW (a sign, a wordmark), not quoted prose: properly closed straight
    or curly quotes (an unclosed quote is ignored), content <= 60% of the prompt, <= 40 characters, <= 6
    words, with at least one letter or digit; at most 3 per prompt, in order."""
    body = text.split(SPELL_SEP)[0]
    out = []
    for m in QUOTE_RE.finditer(body):
        c = m.group(0)[1:-1].strip()
        if (c and any(ch.isalnum() for ch in c) and len(c) <= SPELL_MAX_FRAC * len(body)
                and len(c) <= SPELL_MAX_CHARS and len(c.split()) <= SPELL_MAX_WORDS):
            out.append(c)
    return out[:SPELL_MAX_SPANS]


def spell(content: str) -> str:
    """'Luma Studio' -> 'L u m a / S t u d i o': one token per letter (Ettin's BPE keeps a spaced capital or
    lowercase letter whole), case kept, '/' between words, punctuation other than & ! ? ' - dropped."""
    words = [[ch for ch in w if ch.isalnum() or ch in "&!?'-"] for w in content.split()]
    return " / ".join(" ".join(w) for w in words if w)


def add_spelling(text: str) -> str:
    """Append the spelled-out letters of every drawable quoted span after SPELL_SEP (idempotent). Why: Ettin
    reads "OPEN" as ONE token and a one-letter change moved the thinker's plan only 0.37 (a synonym: 0.36),
    tools/text_probe.py 2026-09-26; spelled letters give the thinker the characters themselves."""
    if SPELL_SEP in text or is_json_caption(text):
        return text
    spans = text_spans(text)
    if not spans:
        return text
    return text + SPELL_SEP + " ; ".join(spell(c) for c in spans)


def split_negatives(text: str) -> tuple[str, list[str]]:
    """Inference only: move 'without X' / 'with no X' / 'no X' (outside quotes) to a negative list.
    X runs to the next comma, semicolon, full stop, ' and ' or the end."""
    negs = []
    if is_json_caption(text):                      # "no cast shadow" is part of a JSON caption's content
        return text, negs
    pat = re.compile(r"\s*,?\s*\b(?:with\s+)?(?:without|no)\s+(?:any\s+)?(?:a\s+|an\s+|the\s+)?([^,.;\"“]+?)"
                     r"(?=\s+and\s+|[,.;]|$)", re.I)

    def cut(seg):
        def f(m):
            negs.append(m.group(1).strip())
            return ""
        return pat.sub(f, seg)
    pos = _map_outside(text, cut)
    pos = re.sub(r"\s+", " ", re.sub(r"\s+([,.;])", r"\1", pos)).strip(" ,;")
    return pos, negs


def typo(text: str, rng: random.Random, p: float) -> str:
    """Training augmentation: with probability p, one typo (swap / drop / double a letter) in one word of
    4+ letters outside quotes that is not a number word."""
    if p <= 0 or rng.random() >= p or is_json_caption(text):
        return text
    cands = [(a + m.start(), a + m.end()) for a, b in _outside_quotes(text)
             for m in re.finditer(r"[A-Za-z]{4,}", text[a:b]) if m.group(0).lower() not in NUM_WORDS]
    if not cands:
        return text
    s, e = rng.choice(cands)
    w = list(text[s:e])
    i = rng.randrange(1, len(w) - 1)
    op = rng.choice(("swap", "drop", "double"))
    if op == "swap":
        w[i], w[i + 1] = w[i + 1], w[i]
    elif op == "drop":
        del w[i]
    else:
        w.insert(i, w[i])
    return text[:s] + "".join(w) + text[e:]


def count_tensor(encoder, texts: list[str], length: int):
    """(B, length) float tensor: the count value at the tokens that spell a count, 0 elsewhere -- aligned
    with encoder.tokenize(texts) (HF fast tokenizer offsets; the byte-level HashTextEncoder: token j = byte
    j - 1). Truncated / zero-padded to `length` (the bucketed token length)."""
    import torch
    out = torch.zeros(len(texts), length)
    spans = [find_counts(t) for t in texts]
    if not any(spans):
        return out
    tok = getattr(encoder, "tok", None)
    if tok is not None:
        import contextlib
        with getattr(encoder, "tok_lock", None) or contextlib.nullcontext():   # shared with tokenize(): see TOK_LOCK
            offs = tok(texts, truncation=True, max_length=encoder.max_len,
                       return_offsets_mapping=True)["offset_mapping"]
    else:
        offs = [[(0, 0)] + [(j, j + 1) for j in range(len(t.encode("utf-8")))] for t in texts]
    for r, (sp, off) in enumerate(zip(spans, offs)):
        for j, (a, b) in enumerate(off[:length]):
            if b <= a:
                continue
            for s, e, n in sp:
                if a < e and b > s:
                    out[r, j] = n
    return out
