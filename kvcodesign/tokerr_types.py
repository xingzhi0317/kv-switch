# Token-type classifier shared by tokerr.py and dirloss.py (our own heuristic, no POS tagger).
# First match wins: first (position 0) | digit | punct (punctuation / whitespace only) | cap (word starting with an
# upper-case letter, not at a sentence start: a rough proxy for proper nouns) | cont (sub-word continuation) | word.
import os, string
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/kvswitch/models/Qwen3-4B"))
PUNCT = set(string.punctuation) | set("“”‘’—–…·")
TYPES = ["first", "digit", "punct", "cap", "cont", "word"]


def token_types(ids):
    out, prev_end = [], True
    for j, i in enumerate(ids.tolist()):
        s = tok.decode([i]); t = s.strip()
        if j == 0:
            ty = "first"
        elif any(c.isdigit() for c in t):
            ty = "digit"
        elif not t or all(c in PUNCT for c in t):
            ty = "punct"
        elif t[0].isupper() and not prev_end and s[:1].isspace():
            ty = "cap"
        elif not s[:1].isspace() and t[0].isalpha():
            ty = "cont"
        else:
            ty = "word"
        if t:
            prev_end = t[-1] in ".!?:\"”" or "\n" in s
        elif "\n" in s:
            prev_end = True
        out.append(ty)
    return out
