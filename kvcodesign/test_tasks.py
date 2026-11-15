# CPU-only checks for tasks.py (scoring and the synthetic tasks).  No model, no tokenizer download:
# a whitespace "tokenizer" stands in for the real one.   python test_tasks.py
import tasks as T

enc = lambda t: t.split()

# ---- GSM8K extraction: must be insensitive to markdown / LaTeX / thousands separators
for pred, gold, ok in [
    ("... so 9 * 2 = 18.\nThe answer is **18**.", "18", True),
    ("The answer is $1,234.", "1234", True),
    ("Thus the total is \\boxed{42}", "42", True),
    ("\\[\\boxed{\\$42}\\]", "42", True),
    ("Answer: 7.5", "7.5", True),
    ("3 apples and 4 pears, 7 in total", "7", True),
    ("The answer is 19.", "18", False),
    ("no number here", "18", False),
]:
    assert T.gsm_eq(T.gsm_extract(pred), gold) == ok, pred

# ---- F1 / EM
assert T.f1("**Denver Broncos**", ["Denver Broncos"]) == 1.0 and T.em("The Denver Broncos.", ["Denver Broncos"]) == 1.0
assert T.f1("\\boxed{Paris}", ["Paris"]) == 1.0
assert 0 < T.f1("the Broncos of Denver", ["Denver Broncos"]) < 1 and T.em("Broncos", ["Denver Broncos"]) == 0.0
assert T.f1("", ["x"]) == 0.0

# ---- code scoring
g = "482913"
assert T.code_score("482913", g) == dict(score=1.0, digit_acc=1.0)
assert T.code_score("The code is **482,913**.", g)["score"] == 1.0
assert T.code_score("482 913", g)["score"] == 1.0
assert T.code_score("\\boxed{482913}", g)["score"] == 1.0
r = T.code_score("482931", g); assert r["score"] == 0.0 and abs(r["digit_acc"] - 4 / 6) < 1e-9
r = T.code_score("4829130", g); assert r["score"] == 0.0 and r["digit_acc"] < 1
assert T.code_score("I don't know", g) == dict(score=0.0, digit_acc=0.0)
assert T.code_score("In 2024 the code was 482913", g)["score"] == 1.0

# ---- passkey: deterministic, right length, code stated exactly once at the requested depth
arts = [" ".join(f"Sentence number {i} of article {a} talks about nothing in particular." for i in range(400)) for a in range(7)]
for L in (2000, 8000):
    a = T.passkey_samples(enc, 10, L, arts); b = T.passkey_samples(enc, 10, L, arts)
    assert [x["h"] for x in a] == [x["h"] for x in b]
    for i, s in enumerate(a):
        h = " ".join(s["h"])
        assert h.count(s["gold"]) == 1 and len(s["gold"]) == 6 and s["kind"] == "code"
        assert abs(len(s["h"]) - L) < 0.03 * L, (len(s["h"]), L)
        pos = h.index(s["gold"]) / len(h)
        assert abs(pos - s["meta"]["depth"]) < 0.05, (pos, s["meta"])
        assert h.rstrip().endswith("Reply with the number only.") and s["q"][0] == "Answer:<|im_end|>"
    assert len({s["gold"] for s in a}) == 10 and [s["meta"]["depth"] for s in a[:5]] == T.DEPTHS

# ---- key-value retrieval
a = T.kvret_samples(enc, 20); b = T.kvret_samples(enc, 20)
assert [x["h"] for x in a] == [x["h"] for x in b]
for s in a:
    h = " ".join(s["h"])
    assert h.count(s["gold"]) == 1
    name = h.split("ID number of ")[1].split("?")[0]
    assert f"{name}: {s['gold']}" in h and h.count(name) == 2 and h.count(": ") == 41

# ---- score() dispatch
assert T.score(dict(kind="mc", gold=2), 2) == dict(score=1.0)
assert T.score(dict(kind="gsm", gold="18"), "The answer is 18")["score"] == 1.0
assert set(T.score(dict(kind="gen", gold=["x"]), "x")) == {"score", "em"}
assert set(T.score(dict(kind="code", gold=g), g)) == {"score", "digit_acc"}
print("TASKS OK")
