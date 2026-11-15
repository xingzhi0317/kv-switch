# What does the rank truncation throw away, measured in the directions attention actually reads?
# (PLAN_v5 stage 2, item 3.)  CPU only, our own analysis code.  Same data and token types as tokerr.py.
# For every held-out token: dense prediction yd = x W, low-rank prediction yr = x P D, target y (all centred).
#   cut   = ||yd - yr||^2 / ||yd||^2        share of the dense prediction removed by the truncation
#   err_d = ||yd - y||^2 / ||y||^2,  err_r = ||yr - y||^2 / ||y||^2     (ratios of sums over the tokens of a type)
# each computed twice per target layer and averaged over layers:
#   plain    : Euclidean norm
#   readout  : K weighted by the per-frequency query energy (qstats wk); V measured after the output projection
#              (qstats Mv = sum_j W_o,j^T W_o,j), i.e. the part of the error the target model can actually see.
# Output: results/dirloss.json     env: THREADS
import os, json, time, torch
torch.set_num_threads(int(os.environ.get("THREADS", 48)))
from tokerr_types import token_types, TYPES
HOME = os.path.expanduser("~")
LS, LT, H, D = 28, 36, 8, 128
PK, QK = LS * H * D, LT * H * D
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
st = torch.load(f"{HOME}/kvcodesign/data/stats_17to4.pt", mmap=True)
mx, my = st["mx"].float(), st["my"].float()
DOCS = torch.load(f"{HOME}/kvcodesign/data/val_docs.pt")
QS = torch.load(f"{HOME}/kvcodesign/data/qstats_4b.pt")
wk = QS["wk"].reshape(LT, H * D)                                    # [LT, H*D]
ev, V = torch.linalg.eigh(QS["Mv"].double())
Mh = ((V * ev.clamp_min(0).sqrt()[..., None, :]) @ V.transpose(-1, -2)).float()   # Mv^(1/2)  [LT, H, D, D]
M = lambda r: torch.load(f"{HOME}/kvcodesign/mappers/rrr_var_r{r}.pt")
Wk, Wv = torch.load(f"{HOME}/kvcodesign/data/wraw_a0.01.pt")
RANKS = [256, 512, 768, 1024, 1536, 2048]
maps = {r: M(r) for r in RANKS}


def norms(e, kind):   # e [n, QK] -> (plain, readout) squared norms per (token, layer)  [n, LT]
    n = e.shape[0]
    plain = e.view(n, LT, H * D).square().sum(2)
    if kind == "k":
        rd = (e.view(n, LT, H * D).square() * wk).sum(2)
    else:
        rd = torch.einsum("nlhd,lhde->nlhe", e.view(n, LT, H, D), Mh).square().sum((2, 3))
    return plain, rd


acc = {}   # (kind, rank, type, quantity, metric) -> [LT] sums
add = lambda key, v: acc.__setitem__(key, acc.get(key, 0) + v)
for di, d in enumerate(DOCS):
    n = d["sk"].shape[1]
    ty = token_types(d["ids"])
    sel = {t: torch.tensor([a == t for a in ty]) for t in TYPES}
    sel["all"] = torch.tensor([a != "first" for a in ty])
    for kind, W, xs, ys, off_x, off_y in (("k", Wk, d["sk"], d["tk"], 0, 0), ("v", Wv, d["sv"], d["tv"], PK, QK)):
        x = xs.float().permute(1, 0, 2, 3).reshape(n, PK) - mx[off_x:off_x + PK]
        y = ys.float().permute(1, 0, 2, 3).reshape(n, QK) - my[off_y:off_y + QK]
        yd = x @ W
        ny, nd, ed = norms(y, kind), norms(yd, kind), norms(yd - y, kind)
        for r in RANKS:
            yr = (x @ maps[r][f"P{kind}"]) @ maps[r][f"D{kind}"]
            cut, er = norms(yd - yr, kind), norms(yr - y, kind)
            for t, s in sel.items():
                for mi, mname in enumerate(("plain", "readout")):
                    for q, v in (("cut", cut), ("err_r", er), ("err_d", ed), ("ny", ny), ("nd", nd)):
                        add((kind, r, t, q, mname), v[mi][s].sum(0))
    log("doc", di)

out = {}
for kind in "kv":
    for r in RANKS:
        for t in TYPES + ["all"]:
            for mname in ("plain", "readout"):
                g = lambda q: acc[(kind, r, t, q, mname)]
                out[f"{kind}.r{r}.{t}.{mname}"] = dict(
                    cut=float((g("cut") / g("nd")).mean()), err_dense=float((g("err_d") / g("ny")).mean()),
                    err_lowrank=float((g("err_r") / g("ny")).mean()),
                    cut_by_layer=(g("cut") / g("nd")).tolist())
json.dump(out, open(f"{HOME}/kvcodesign/results/dirloss.json", "w"), indent=1)
for kind in "kv":
    for mname in ("plain", "readout"):
        print(f"== {kind.upper()} {mname}: share of the dense prediction removed by truncation | error low-rank / error dense")
        for r in RANKS:
            print(f"  r={r:4d} " + "  ".join(f"{t} {out[f'{kind}.r{r}.{t}.{mname}']['cut']:.3f}|{out[f'{kind}.r{r}.{t}.{mname}']['err_lowrank'] / out[f'{kind}.r{r}.{t}.{mname}']['err_dense']:.2f}"
                                             for t in ["all", "digit", "cap", "cont", "punct", "word"]))
        print("  dense error: " + "  ".join(f"{t} {out[f'{kind}.r768.{t}.{mname}']['err_dense']:.3f}" for t in ["all", "digit", "cap", "cont", "punct", "word"]))
log("DIRLOSSDONE")
