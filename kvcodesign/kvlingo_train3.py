# kvlingo_train2.py plus checkpointing, resume and learning-rate pruning (same arithmetic per step).
#   - checkpoint every CKPT_EVERY steps (rank 0, atomic): for every active learning rate the translator and its AdamW
#     state, plus step, history and OOM count -> kvlingo/ckpt3_{TAG}.pt.  The data order and the cosine schedule are
#     functions of the step, so a resumed run continues exactly where it stopped.
#   - RESUME=1 (default): continue from the checkpoint if it exists.  LRS on resume may be a subset of the saved rates
#     (manual pruning); the others are dropped.
#   - PRUNE_AT=n (0 = off): after the validation at step n, drop every rate whose validation KL exceeds the best by more
#     than PRUNE_MARGIN (default 5%); the survivors run to the end.  Closer rates all stay, so the final choice is still
#     made on the final validation KL (KV-Lingo App. E) among them.  Recorded in the history ("pruned").
# Other settings as kvlingo_train2.py.  Data: data/{KVL2}_ids.npy + _index.json (kvlingo_data2.py).
# usage (training only; the closed form is kvlingo_train2.py MODE=fit):
#   PAIR=17to4 LRS=3e-6,1e-5,3e-5 TAG=... PRUNE_AT=1000 CUDA_VISIBLE_DEVICES=a,b,c \
#     torchrun --nnodes=1 --nproc_per_node=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:PORT kvlingo_train3.py
import os, sys, json, math, time, random, datetime
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

sys.argv = sys.argv[:1]
import kvlingo as K

E = os.environ.get
LRS_S = E("LRS", "1e-5").split(",")
STEPS, BATCH, VAL_EVERY = int(E("STEPS", 5000)), 8, int(E("VAL_EVERY", 500))
CKPT_EVERY = int(E("CKPT_EVERY", 250))
RESUME = E("RESUME", "1") == "1"
PRUNE_AT, PRUNE_MARGIN = int(E("PRUNE_AT", 0)), float(E("PRUNE_MARGIN", 0.05))
MAXP = int(E("MAXP", 16384))
TAG = E("TAG", f"kvl3_{K.PAIR}")
HOME = os.path.expanduser("~")
CF2 = E("CF2", f"{K.OUT}/cf2_{K.PAIR}_nu{K.NU}_b{K.BIAS}.pt")
RES, CKPT = f"{K.OUT}/res_{TAG}.json", f"{K.OUT}/ckpt3_{TAG}.pt"
DATA = E("KVL2", "kvl2")
IDS = np.load(f"{HOME}/kvcodesign/data/{DATA}_ids.npy", mmap_mode="r")
IDX = json.load(open(f"{HOME}/kvcodesign/data/{DATA}_index.json"))
t0 = time.time()


def sample(e):
    o, n_p, n_c = e["o"], e["np"], e["nc"]
    return dict(p=IDS[o:o + n_p].tolist(), c=IDS[o + n_p:o + n_p + n_c].tolist())


def main():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=1))
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    K.SDEV = K.TDEV = dev
    log = (lambda *a: print(time.strftime("%H:%M:%S"), f"[{(time.time() - t0) / 60:.1f} min]", *a, flush=True)) if rank == 0 else (lambda *a: None)
    S, T = K.load(K.SRC, dev), K.load(K.TGT, dev)

    ck_state = torch.load(CKPT, map_location=dev, weights_only=False) if RESUME and os.path.exists(CKPT) else None
    if ck_state is not None:
        names = [s for s in LRS_S if s in ck_state["lrs"]] or list(ck_state["lrs"])
        dropped = [s for s in ck_state["lrs"] if s not in names]
        start, hist, oom = ck_state["step"] + 1, ck_state["hist"], ck_state["oom"]
        if dropped:
            hist.setdefault("pruned", []).append(dict(step=ck_state["step"], dropped=dropped, how="manual on resume"))
    else:
        names, start, oom = list(LRS_S), 1, 0
        hist = dict(config=dict(pair=K.PAIR, lrs=LRS_S, steps=STEPS, batch=BATCH, maxp=MAXP, world=world, data=DATA, cf=CF2,
                                prune_at=PRUNE_AT, prune_margin=PRUNE_MARGIN), val=[], train=[], oom=0, pruned=[])
    trs = {s: K.load_tr(S, T, CF2) for s in names}
    opts = {s: torch.optim.AdamW([p for p in trs[s].parameters() if p.requires_grad], lr=float(s), betas=(0.9, 0.999), weight_decay=0.0)
            for s in names}
    if ck_state is not None:
        for s in names:
            trs[s].load_state_dict(ck_state["states"][s]["tr"])
            opts[s].load_state_dict(ck_state["states"][s]["opt"])
        log("resumed from step", ck_state["step"], "rates", names, "dropped", dropped)
        del ck_state
    params = {s: [p for p in trs[s].parameters() if p.requires_grad] for s in names}

    def ck(h_, r_):
        lp = T.lm_head(h_).float().log_softmax(-1); lr_ = T.lm_head(r_).float().log_softmax(-1)
        return (lr_.exp() * (lr_ - lp)).sum()

    def shared(x):
        p, c = x["p"], x["c"]
        n = len(p) - 1
        pi = torch.tensor([p], device=dev)
        ci = torch.tensor([[p[-1]] + c[:-1]], device=dev)
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

    def validate(which):
        tot = torch.zeros(len(which), device=dev)
        with torch.no_grad():
            for e in val[rank::world]:
                sh = shared(sample(e))
                for k, s in enumerate(which):
                    tot[k] += float(student(trs[s], *sh, grad=False))
                del sh
        dist.all_reduce(tot)
        return {s: round(v, 4) for s, v in zip(which, (tot / len(val)).tolist())}

    def save_ckpt(step):
        if rank == 0:
            st = dict(step=step, lrs=list(names), hist=hist, oom=oom,
                      states={s: dict(tr=trs[s].state_dict(), opt=opts[s].state_dict()) for s in names})
            torch.save(st, CKPT + ".tmp"); os.replace(CKPT + ".tmp", CKPT)
        dist.barrier()

    if start == 1:
        v0 = validate(names[:1])[names[0]]   # all rates start from the same closed form
        hist["val"].append((0, {s: v0 for s in names}))
        log("step 0 (closed form) val KL", v0)
    train = [e for e in IDX if e["split"] == "train" and e["np"] <= MAXP]
    order = list(range(len(train))); random.Random(0).shuffle(order)
    warm, run, ts = max(1, int(0.05 * STEPS)), [], time.time()
    for step in range(start, STEPS + 1):
        f = min(1.0, step / warm) * 0.5 * (1 + math.cos(math.pi * max(0, step - warm) / max(1, STEPS - warm)))
        for s in names:
            for g in opts[s].param_groups:
                g["lr"] = float(s) * f
        mine = [order[((step - 1) * BATCH + j) % len(order)] for j in range(BATCH)][rank::world]
        tots = torch.zeros(len(names) + 1, device=dev)
        for j in mine:
            try:
                sh = shared(sample(train[j]))
                for k, s in enumerate(names):
                    loss = student(trs[s], *sh) / BATCH
                    loss.backward(); tots[k] += float(loss)
                del sh
            except torch.OutOfMemoryError:
                tots[-1] += 1; torch.cuda.empty_cache()
        for s in names:   # sum gradients over ranks, then clip and step each translator
            gs = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params[s]]
            flat = torch._utils._flatten_dense_tensors(gs)
            dist.all_reduce(flat)
            for p, g in zip(params[s], torch._utils._unflatten_dense_tensors(flat, gs)):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(params[s], 1.0)
            opts[s].step(); opts[s].zero_grad(set_to_none=True)
        dist.all_reduce(tots)
        oom += int(tots[-1]); run.append(dict(zip(names, tots[:-1].tolist())))
        if step % 25 == 0:
            m = {s: round(sum(r[s] for r in run) / len(run), 4) for s in names}
            hist["train"].append((step, m))
            log("step", step, "train KL", m, "s/step", round((time.time() - ts) / (step - start + 1), 1), "oom", oom); run = []
        if step % VAL_EVERY == 0 or step == STEPS or step == PRUNE_AT:
            hist["val"].append((step, validate(names))); hist["oom"] = oom
            log("step", step, "val KL", hist["val"][-1][1])
            if rank == 0:
                for s in names:
                    torch.save(dict(state=trs[s].state_dict(), ch=trs[s].ch, step=step), f"{K.OUT}/sd3_{TAG}_lr{s}.pt")
            if step == PRUNE_AT and len(names) > 1:
                v = hist["val"][-1][1]
                best = min(v.values())
                drop = [s for s in names if v[s] > best * (1 + PRUNE_MARGIN)]
                if drop:
                    hist["pruned"].append(dict(step=step, dropped=drop, val=v, margin=PRUNE_MARGIN))
                    names = [s for s in names if s not in drop]
                    for s in drop:
                        del trs[s], opts[s], params[s]
                    torch.cuda.empty_cache()
                    log("pruned", drop, "keep", names)
            if rank == 0:
                json.dump(hist, open(RES, "w"), indent=1)
        if step % CKPT_EVERY == 0 and step < STEPS:
            save_ckpt(step)
            if step == int(E("STOP_AFTER", 0)):   # testing only: simulate an interruption right after a checkpoint
                log("STOP_AFTER", step); dist.barrier(); dist.destroy_process_group(); return
    if rank == 0:
        fin = hist["val"][-1][1]
        best = min(fin, key=fin.get)
        allr = hist["config"]["lrs"]
        hist["best_lr"], hist["best_at_edge"] = best, best in (allr[0], allr[-1])
        json.dump(hist, open(RES, "w"), indent=1)
        os.system(f"ln -sfn sd3_{TAG}_lr{best}.pt {K.OUT}/sd3_{TAG}_best.pt")
        log("TRAIN3DONE best LR", best, "final val KL", fin, "at edge" if hist["best_at_edge"] else "")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
