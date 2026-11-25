# KV-Lingo-style training stream for the KL-trained translator (KV-Lingo arXiv 2609.32610, §4.1 and App. C):
#   - reasoning-on and reasoning-off samples alternate (Nemotron-SFT-Instruction-Following-Chat-v2, revision 1a9454ed);
#   - half natural conversations split at the start of a random assistant turn, prefix about 10 to 10K tokens and
#     roughly uniform (we resample over 1K-token bins);
#   - half concatenated conversations, prefix 12K-16K tokens: one question/answer pair is removed and its question is
#     placed at the end of the prefix, its answer is the continuation;
#   - reasoning-on: the prefix ends at the bare assistant header and the continuation opens with the <think> block;
#     reasoning-off: the prefix ends with the empty thinking placeholder (Qwen3 template). Earlier assistant turns keep
#     only their final content (as the Qwen3 template renders history). Continuations are capped at 2048 tokens.
#   - reasoning_off rows < 200000 and the klchat evaluation conversations are never used.
# Output: data/kvl2_ids.npy (flat int32 token ids) and data/kvl2_index.json (per sample: split, kind, mode, offsets).
import os, json, glob, random, hashlib, time
import numpy as np
from transformers import AutoTokenizer

HOME = os.path.expanduser("~")
NAME = os.environ.get("KVL2", "kvl2")
OUT_IDS, OUT_IDX = f"{HOME}/kvcodesign/data/{NAME}_ids.npy", f"{HOME}/kvcodesign/data/{NAME}_index.json"
N = dict(val=64, fit=400, train=int(os.environ.get("NTRAIN", 40000)))
MAXC, NAT_MAX, CAT_LO, CAT_HI, MAXP_TOT = 2048, 10000, 12000, 16000, 16384
tok = AutoTokenizer.from_pretrained(f"{HOME}/kvswitch/models/Qwen3-4B")
HDR = {"off": "<|im_start|>assistant\n<think>\n\n</think>\n\n", "on": "<|im_start|>assistant\n"}
t0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{(time.time() - t0) / 60:.1f} min]", *a, flush=True)


def enc_many(texts):
    return tok(texts, add_special_tokens=False).input_ids


def answer_text(m, mode):
    if mode == "on":
        return f"<think>\n{(m.get('reasoning_content') or '').strip()}\n</think>\n\n{m['content'].lstrip()}<|im_end|>"
    return m["content"] + "<|im_end|>"


def stream(mode, r=0, W=1):
    """Conversations of one reasoning mode as lists of messages with content, in file order; worker r of W reads only
    lines i with i % W == r, so workers never share a conversation."""
    snap = glob.glob(f"{HOME}/.cache/huggingface/hub/datasets--nvidia--Nemotron-SFT-Instruction-Following-Chat-v2/snapshots/*/data")[0]
    evalu = {x["uuid"] for v in json.load(open(f"{HOME}/kvcodesign/data/klchat.json")).values() for x in v}
    for i, line in enumerate(open(f"{snap}/reasoning_{mode}.jsonl")):
        if (mode == "off" and i < 200000) or i % W != r:
            continue
        d = json.loads(line)
        if d.get("uuid") in evalu:
            continue
        m = [x for x in d["messages"] if x.get("content")]
        if any(x["role"] == "assistant" for x in m[1:]):
            yield d.get("uuid", str(i)), m


class Source:
    """Per mode: a pool of natural samples binned by prefix length, and a conversation stream for concatenation."""
    def __init__(self, mode, r=0, W=1):
        self.mode, self.r, self.W = mode, r, W
        self.it, self.bins = stream(mode, r, W), [[] for _ in range(NAT_MAX // 1000)]
        self.used, self.restarts = 0, 0

    def next_conv(self):
        self.used += 1
        try:
            return next(self.it)
        except StopIteration:   # this worker's share is used up: start over (counted, reported)
            self.restarts += 1
            self.it = stream(self.mode, self.r, self.W)
            return next(self.it)

    def natural_candidate(self):
        uid, m = self.next_conv()
        turns = [j for j, x in enumerate(m) if x["role"] == "assistant" and j > 0]
        j = turns[int(hashlib.md5(uid.encode()).hexdigest(), 16) % len(turns)]
        segs = enc_many([f"<|im_start|>{x['role']}\n{x['content']}<|im_end|>\n" for x in m[:j]] + [HDR[self.mode], answer_text(m[j], self.mode)])
        p = [t for s in segs[:-1] for t in s]
        c = segs[-1][:MAXC]
        b = min(len(p) // 1000, len(self.bins) - 1)
        if 10 <= len(p) <= NAT_MAX and len(c) >= 2 and len(self.bins[b]) < 5000:   # cap memory per bin
            self.bins[b].append((np.asarray(p, dtype=np.int32), np.asarray(c, dtype=np.int32)))

    def natural(self, rng):
        b = rng.randrange(len(self.bins))   # uniform over 1K bins
        tries = 0
        while not self.bins[b] and tries < 50:
            self.natural_candidate(); tries += 1
        if not self.bins[b]:   # this length range is too rare in the data: nearest non-empty bin
            b = min((k for k in range(len(self.bins)) if self.bins[k]), key=lambda k: abs(k - b))
        p, c = self.bins[b].pop()
        return p.tolist(), c.tolist()

    def concat(self, rng):
        while True:   # a single huge question can exceed the cap on its own: draw a new set of conversations
            p, c = self._concat(rng)
            if len(p) <= MAXP_TOT:
                return p, c

    def _concat(self, rng):
        L = rng.randint(CAT_LO, CAT_HI)
        pairs, tot = [], 0
        while tot < L:
            uid, m = self.next_conv()
            m = [x for x in m if x["role"] in ("user", "assistant")]
            for a in range(len(m) - 1):
                if m[a]["role"] == "user" and m[a + 1]["role"] == "assistant":
                    u, h, ans = enc_many([f"<|im_start|>user\n{m[a]['content']}<|im_end|>\n",
                                          f"<|im_start|>assistant\n{m[a + 1]['content']}<|im_end|>\n",
                                          answer_text(m[a + 1], self.mode)])
                    pairs.append((u, h, ans)); tot += len(u) + len(h)
        k = rng.randrange(len(pairs))
        hdr = enc_many([HDR[self.mode]])[0]
        keep = [i for i in range(len(pairs)) if i != k]
        while keep and sum(len(pairs[i][0]) + len(pairs[i][1]) for i in keep) + len(pairs[k][0]) + len(hdr) > MAXP_TOT:
            keep.pop(0)   # the last conversation overshot: drop whole leading pairs, never cut inside a message
        p = [t for i in keep for t in pairs[i][0] + pairs[i][1]] + pairs[k][0] + hdr
        return p, pairs[k][2][:MAXC]


def work(args):
    """Worker r of W: samples i with i % W == r of every split, from its own share of the conversations."""
    r, W = args
    rng = random.Random(r)
    src = {"off": Source("off", r, W), "on": Source("on", r, W)}
    ids, rows, off = [], [], 0
    for split in ("val", "fit", "train"):
        for i in [i for i in range(N[split]) if (i // 4) % W == r]:   # blocks of 4: every worker gets all four kinds
            mode = ("off", "on")[i % 2]
            kind = ("natural", "concat")[(i // 2) % 2]
            p, c = src[mode].natural(rng) if kind == "natural" else src[mode].concat(rng)
            ids.append(np.asarray(p + c, dtype=np.int32))
            rows.append(dict(split=split, i=i, kind=kind, mode=mode, so=off, np=len(p), nc=len(c)))
            off += len(p) + len(c)
    np.save(f"{OUT_IDS}.shard{r}.npy", np.concatenate(ids))
    return r, rows, {k: (v.used, v.restarts) for k, v in src.items()}


def main():
    from multiprocessing import Pool
    W = int(os.environ.get("WORKERS", 32))
    with Pool(W) as pool:
        res = pool.map(work, [(r, W) for r in range(W)])
    log("workers done")
    shards = {r: np.load(f"{OUT_IDS}.shard{r}.npy", mmap_mode="r") for r, _, _ in res}
    log("conversations used / restarts per mode, summed over workers",
        {m: (sum(u[m][0] for _, _, u in res), sum(u[m][1] for _, _, u in res)) for m in ("off", "on")})
    rows = sorted(((("val", "fit", "train").index(x["split"]), x["i"], r, x) for r, rs, _ in res for x in rs), key=lambda t: t[:2])
    ids, index, off = [], [], 0
    for _, _, r, x in rows:
        ids.append(np.asarray(shards[r][x["so"]:x["so"] + x["np"] + x["nc"]]))
        index.append(dict(split=x["split"], kind=x["kind"], mode=x["mode"], o=off, np=x["np"], nc=x["nc"]))
        off += x["np"] + x["nc"]
    np.save(OUT_IDS, np.concatenate(ids))
    json.dump(index, open(OUT_IDX, "w"))
    for r in shards:
        os.remove(f"{OUT_IDS}.shard{r}.npy")
    for split in ("val", "fit", "train"):
        for kind in ("natural", "concat"):
            ps = [x["np"] for x in index if x["split"] == split and x["kind"] == kind]
            log(split, kind, len(ps), "prefix mean", int(np.mean(ps)), "p10/p50/p90/max", [int(np.percentile(ps, q)) for q in (10, 50, 90, 100)])
        cs = [x["nc"] for x in index if x["split"] == split]
        log(split, "continuation mean", int(np.mean(cs)))
    log("KVL2DATADONE", off, "tokens")


if __name__ == "__main__":
    main()
