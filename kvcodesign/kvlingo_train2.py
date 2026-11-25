# Stage 1 + stage 2 of the KL-trained translator (kvlingo.py) on the KV-Lingo-style stream (kvlingo_data2.py), with the
# KV-Lingo learning-rate selection (App. E: several rates, lowest final validation KL on the 64 held-out conversations).
# Speed-ups over kvlingo.py (same arithmetic per conversation):
#   - several learning rates in one process: per conversation the source pre-norm K/V and the teacher's hidden states
#     are computed once and shared; each rate has its own translator and AdamW;
#   - data parallel over GPUs (torchrun): each step's 8 conversations are split across ranks, translator gradients are
#     summed across ranks before clipping (clip 1.0 per translator) and the optimiser step.
# Everything else as kvlingo.py: pre-norm capture, nu=1 relative depth, no bias, KL(teacher || student) averaged over the
# continuation then over the 8 conversations, AdamW wd 0, cosine with 5% warmup, translator fp32, models bf16.
# usage:
#   MODE=fit   PAIR=17to4 CUDA_VISIBLE_DEVICES=a python kvlingo_train2.py           (closed form, FIT_TOK positions)
#   MODE=train PAIR=17to4 LRS=3e-6,1e-5,3e-5 TAG=kvl2_17to4 CUDA_VISIBLE_DEVICES=a,b,c \
#       torchrun --nnodes=1 --nproc_per_node=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:29511 kvlingo_train2.py
import os, sys, json, math, time, random, datetime
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

sys.argv = sys.argv[:1]
import kvlingo as K

E = os.environ.get
MODE = E("MODE", "train")
LRS = [float(x) for x in E("LRS", "1e-5").split(",")]
LRS_S = E("LRS", "1e-5").split(",")
STEPS, BATCH, VAL_EVERY = int(E("STEPS", 5000)), 8, int(E("VAL_EVERY", 500))
MAXP = int(E("MAXP", 16384))
TAG = E("TAG", f"kvl2_{K.PAIR}")
HOME = os.path.expanduser("~")
CF2 = E("CF2", f"{K.OUT}/cf2_{K.PAIR}_nu{K.NU}_b{K.BIAS}.pt")
RES = f"{K.OUT}/res_{TAG}.json"
DATA = E("KVL2", "kvl2")   # data/{DATA}_ids.npy + data/{DATA}_index.json (kvlingo_data2.py)
IDS = np.load(f"{HOME}/kvcodesign/data/{DATA}_ids.npy", mmap_mode="r")
IDX = json.load(open(f"{HOME}/kvcodesign/data/{DATA}_index.json"))
t0 = time.time()


def sample(e):
    o, n_p, n_c = e["o"], e["np"], e["nc"]
    return dict(p=IDS[o:o + n_p].tolist(), c=IDS[o + n_p:o + n_p + n_c].tolist())


def fit_mode():
    dev = "cuda:0"
    K.SDEV = K.TDEV = dev
    K.CF_PATH = CF2
    S, T = K.load(K.SRC, dev), K.load(K.TGT, dev)
    pool, fitset, ntok = [e for e in IDX if e["split"] == "fit"] + [e for e in IDX if e["split"] == "train"], [], 0
    fmax = int(E("FIT_MAXP", 12288))   # two models' captures at 16K do not fit in 24GB: stage-1 statistics use the first
    for e in pool:                     # FIT_MAXP prefix tokens of each conversation (more conversations fill FIT_TOK)
        if ntok >= int(E("FIT_TOK", 2800000)):
            break
        x = sample(e)
        if len(x["p"]) > fmax:
            x["p"] = x["p"][:fmax]
        fitset.append(x); ntok += len(x["p"]) - 1
    K.log("fit conversations", len(fitset), "prefix positions", ntok)
    K.fit(S, T, fitset)


def train_mode():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=1))
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    K.SDEV = K.TDEV = dev
    log = (lambda *a: print(time.strftime("%H:%M:%S"), f"[{(time.time() - t0) / 60:.1f} min]", *a, flush=True)) if rank == 0 else (lambda *a: None)
    S, T = K.load(K.SRC, dev), K.load(K.TGT, dev)
    trs = [K.load_tr(S, T, CF2) for _ in LRS]
    params = [[p for p in tr.parameters() if p.requires_grad] for tr in trs]
    opts = [torch.optim.AdamW(ps, lr=lr, betas=(0.9, 0.999), weight_decay=0.0) for ps, lr in zip(params, LRS)]

    def ck(h_, r_):
        lp = T.lm_head(h_).float().log_softmax(-1); lr_ = T.lm_head(r_).float().log_softmax(-1)
        return (lr_.exp() * (lr_ - lp)).sum()

    def shared(x):
        p, c = x["p"], x["c"]
        n = len(p) - 1
        pi = torch.tensor([p], device=dev)
        ci = torch.tensor([[p[-1]] + c[:-1]], device=dev)   # held-back last prefix token, then the continuation
        with torch.no_grad():
            nkv = K.native_kv(T, pi[:, :n])
            ref = K.run_target(T, nkv, n, ci)[0]
            del nkv
            sk, sv = K.prenorm_kv(S, pi, last=n)
        return n, ci, ref, sk, sv, len(c)

    def student(tr, n, ci, ref, sk, sv, m, grad=True):
        with torch.set_grad_enabled(grad):
            h = K.run_target(T, tr.translate(T, sk, sv, grad), n, ci, grad)[0]
            tot = 0
            for a in range(0, m, 512):
                tot = tot + (checkpoint(ck, h[a:a + 512], ref[a:a + 512], use_reentrant=False) if grad else ck(h[a:a + 512], ref[a:a + 512]))
            return tot / m

    val = [e for e in IDX if e["split"] == "val"]

    def validate(only_first=False):
        tot = torch.zeros(len(trs), device=dev)
        with torch.no_grad():
            for e in val[rank::world]:
                sh = shared(sample(e))
                for k, tr in enumerate(trs[:1] if only_first else trs):
                    tot[k] += float(student(tr, *sh, grad=False))
                del sh
        dist.all_reduce(tot)
        v = (tot / len(val)).tolist()
        return [round(v[0], 4)] * len(trs) if only_first else [round(x, 4) for x in v]

    hist = dict(config=dict(pair=K.PAIR, lrs=LRS_S, steps=STEPS, batch=BATCH, maxp=MAXP, world=world, data="kvlingo_data2.py",
                            cf=CF2), val=[], train=[], oom=0)
    hist["val"].append((0, validate(only_first=True)))
    log("step 0 (closed form) val KL", hist["val"][-1][1])
    train = [e for e in IDX if e["split"] == "train" and e["np"] <= MAXP]
    order = list(range(len(train))); random.Random(0).shuffle(order)
    warm, oom, run, ts = max(1, int(0.05 * STEPS)), 0, [], time.time()
    for step in range(1, STEPS + 1):
        f = min(1.0, step / warm) * 0.5 * (1 + math.cos(math.pi * max(0, step - warm) / max(1, STEPS - warm)))
        for o, lr in zip(opts, LRS):
            for g in o.param_groups:
                g["lr"] = lr * f
        mine = [order[((step - 1) * BATCH + j) % len(order)] for j in range(BATCH)][rank::world]
        tots = torch.zeros(len(trs) + 1, device=dev)
        for j in mine:
            try:
                sh = shared(sample(train[j]))
                for k, tr in enumerate(trs):
                    loss = student(tr, *sh) / BATCH
                    loss.backward(); tots[k] += float(loss)
                del sh
            except torch.OutOfMemoryError:
                tots[-1] += 1; torch.cuda.empty_cache()
        for k in range(len(trs)):   # sum gradients over ranks, then clip and step each translator
            gs = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params[k]]
            flat = torch._utils._flatten_dense_tensors(gs)
            dist.all_reduce(flat)
            for p, g in zip(params[k], torch._utils._unflatten_dense_tensors(flat, gs)):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(params[k], 1.0)
            opts[k].step(); opts[k].zero_grad(set_to_none=True)
        dist.all_reduce(tots)
        oom += int(tots[-1]); run.append(tots[:-1].tolist())
        if step % 25 == 0:
            m = [round(sum(r[k] for r in run) / len(run), 4) for k in range(len(trs))]
            hist["train"].append((step, m))
            log("step", step, "train KL", m, "s/step", round((time.time() - ts) / step, 1), "oom", oom); run = []
        if step % VAL_EVERY == 0 or step == STEPS:
            hist["val"].append((step, validate())); hist["oom"] = oom
            log("step", step, "val KL", hist["val"][-1][1])
            if rank == 0:
                for tr, s in zip(trs, LRS_S):
                    torch.save(dict(state=tr.state_dict(), ch=tr.ch, step=step), f"{K.OUT}/sd2_{TAG}_lr{s}.pt")
                json.dump(hist, open(RES, "w"), indent=1)
    if rank == 0:
        fin = hist["val"][-1][1]
        best = min(range(len(LRS)), key=lambda k: fin[k])
        hist["best_lr"] = LRS_S[best]
        hist["best_at_edge"] = best in (0, len(LRS) - 1)
        json.dump(hist, open(RES, "w"), indent=1)
        os.system(f"ln -sfn sd2_{TAG}_lr{LRS_S[best]}.pt {K.OUT}/sd2_{TAG}_best.pt")
        log("TRAIN2DONE best LR", LRS_S[best], "final val KL", fin, "at edge" if hist["best_at_edge"] else "")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    fit_mode() if MODE == "fit" else train_mode()
