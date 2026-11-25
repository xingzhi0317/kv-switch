# Training / validation conversations for the KL-trained translator, following KV-Lingo (arXiv 2609.32610, §3.6, §4.1,
# App. E) and Hybrid-KV Part 1 (arXiv 2611.12062, §3.6, §5):
#   source  : nvidia/Nemotron-SFT-Instruction-Following-Chat-v2, data/reasoning_off.jsonl (rows >= 200000, never the
#             conversations used by our klchat evaluation)
#   prefix  : the prompt = every turn before the last assistant answer, Qwen3 chat template, ending with
#             "<|im_start|>assistant\n<think>\n\n</think>\n\n" (reasoning-off keeps the empty think placeholder),
#             at most 8K tokens (Hybrid-KV)
#   cont    : the last assistant answer + <|im_end|>, capped at 2048 tokens (KV-Lingo m = 2048)
#   val     : 64 held-out conversations (KV-Lingo App. E)
# Cached as token ids in data/kl2_convs.json.
import os, json, random

CACHE = os.path.expanduser("~/kvcodesign/data/kl2_convs.json")
MAXP, MAXC, NVAL, NTRAIN = 8192, 2048, 64, 20000


def load(tok):
    if os.path.exists(CACHE):
        d = json.load(open(CACHE))
        return d["train"], d["val"]
    from huggingface_hub import hf_hub_download
    evalu = {x["uuid"] for v in json.load(open(os.path.expanduser("~/kvcodesign/data/klchat.json"))).values() for x in v}
    path = hf_hub_download("nvidia/Nemotron-SFT-Instruction-Following-Chat-v2", "data/reasoning_off.jsonl", repo_type="dataset")
    enc = lambda t: tok(t, add_special_tokens=False).input_ids
    out = []
    for i, line in enumerate(open(path)):
        if i < 200000:
            continue
        d = json.loads(line)
        if d.get("uuid") in evalu:
            continue
        m = [x for x in d["messages"] if x.get("content")]
        if len(m) < 2 or m[-1]["role"] != "assistant":
            continue
        p = enc("".join(f"<|im_start|>{x['role']}\n{x['content']}<|im_end|>\n" for x in m[:-1])
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        c = enc(m[-1]["content"] + "<|im_end|>")[:MAXC]
        if len(p) > MAXP or len(c) < 2:
            continue
        out.append(dict(uuid=d.get("uuid"), p=p, c=c))
        if len(out) >= NTRAIN + NVAL:
            break
    random.Random(11).shuffle(out)
    val, train = out[:NVAL], out[NVAL:]
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(dict(train=train, val=val), open(CACHE, "w"))
    return train, val
