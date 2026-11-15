# Dependency graph of the dense ridge solution (PLAN_v5 stage 2, item 4).  CPU only, our own analysis code.
# Which source (layer, head) does each target (layer, head) of the dense 1.7B -> 4B map read from?
# W = full dense ridge solution (data/wraw_a0.01.pt, K->K and V->V, all 28 source layers), G = source covariance.
# For a target block t (128 columns) the prediction is  yhat_t = sum_s x_s W[s, t]  over 224 source blocks s.
#   share[s, t]   = cov(x_s W[s,t], yhat_t) summed over the block's columns = sum(W[s,t] * (G W)[s,t]).
#                   Sums over s exactly to the explained variance of block t (can be slightly negative).
#   partial[s, t] = variance of the partial prediction x_s W[s,t] alone = tr(W[s,t]^T G[s,s] W[s,t])  (>= 0).
# Both are divided by the total variance of the target block, so share.sum(0) is the in-sample R^2 of that block.
# Output: results/depgraph.npz (K/V x share/partial, [224, 288]), results/depgraph.json (summary), results/depgraph.png
# env: THREADS
import os, json, time, numpy as np, torch
torch.set_num_threads(int(os.environ.get("THREADS", 48)))
HOME = os.path.expanduser("~")
LS, LT, H, D = 28, 36, 8, 128
PK, QK = LS * H * D, LT * H * D
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

st = torch.load(f"{HOME}/kvcodesign/data/stats_17to4.pt", mmap=True)
ntr = st["ntr"]
yy = (st["yy"] / ntr).float()                                   # per-column target variance [2*QK]
Wk, Wv = torch.load(f"{HOME}/kvcodesign/data/wraw_a0.01.pt", mmap=True)
arr, summ = {}, {}
for kind, W, off_s, off_t in (("k", Wk, 0, 0), ("v", Wv, PK, QK)):
    W = W.float().contiguous()
    G = st["G"][off_s:off_s + PK, off_s:off_s + PK].float().contiguous()
    log(kind, "G W ...")
    GW = G @ W
    share = (W * GW).view(LS * H, D, LT * H, D).sum((1, 3))
    del GW
    Gb = torch.stack([G[s * D:(s + 1) * D, s * D:(s + 1) * D] for s in range(LS * H)])          # [224, D, D]
    Wb = W.view(LS * H, D, QK)
    partial = (torch.bmm(Gb, Wb) * Wb).view(LS * H, D, LT * H, D).sum((1, 3))
    tv = yy[off_t:off_t + QK].view(LT * H, D).sum(1) * ntr                                           # target block sum of squares (G is a sum too)
    share, partial = share / tv, partial / tv
    arr[f"{kind}_share"], arr[f"{kind}_partial"] = share.numpy(), partial.numpy()
    del G, W, Wb, Gb

    s4 = share.view(LS, H, LT, H)
    lay = s4.sum((1, 3)) / H                                    # [LS, LT]: mean over target heads of the layer-level share
    r2 = share.sum(0).view(LT, H).mean(1)                       # in-sample explained variance per target layer
    layn = lay / lay.sum(0, keepdim=True)                       # each target layer's distribution over source layers
    src = torch.arange(LS, dtype=torch.float32)
    com = (layn * src[:, None]).sum(0)                          # centre of mass (source layer) per target layer
    prop = torch.arange(LT, dtype=torch.float32) * (LS - 1) / (LT - 1)   # proportional-depth source layer
    near = torch.stack([layn[max(0, int(round(float(prop[t]))) - 2):int(round(float(prop[t]))) + 3, t].sum() for t in range(LT)])
    # same-index head: share coming from source heads with the same head index (chance = 1/8)
    same = torch.stack([s4[:, h, :, h].sum(0) for h in range(H)], 1) / s4.sum((0, 1)).clamp_min(1e-9)   # [LT, H]
    # concentration: number of source blocks (of 224) / source layers (of 28) needed for 50% / 90% of the positive share
    def need(x, frac):   # x [n_src, n_tgt] -> mean count
        x = x.clamp_min(0).sort(0, descending=True).values
        c = x.cumsum(0) / x.sum(0, keepdim=True)
        return float(((c < frac).sum(0) + 1).float().mean())
    tot_src = lay.sum(1) / lay.sum()                            # how much each source layer feeds all targets
    summ[kind] = dict(
        r2_in_sample_by_target_layer=r2.tolist(),
        layer_share=lay.tolist(),
        top_source_layer_by_target_layer=layn.argmax(0).tolist(),
        top_source_layer_weight=layn.max(0).values.tolist(),
        centre_of_mass_source_layer=com.tolist(),
        corr_centre_of_mass_vs_proportional_depth=float(np.corrcoef(com.numpy(), prop.numpy())[0, 1]),
        frac_within_2_layers_of_proportional_depth=dict(mean=float(near.mean()), by_target_layer=near.tolist(), chance=5 / LS),
        same_head_index_fraction=dict(mean=float(same.mean()), by_target_layer=same.mean(1).tolist(), chance=1 / H),
        source_blocks_for_50pct=need(share, 0.5), source_blocks_for_90pct=need(share, 0.9),
        source_layers_for_50pct=need(lay, 0.5), source_layers_for_90pct=need(lay, 0.9),
        source_layer_total_share=tot_src.tolist(),
        negative_share_mass=float(share.clamp_max(0).sum().abs() / share.clamp_min(0).sum()),
        partial_over_share=float(partial.sum() / share.sum()),   # >1: source blocks are redundant (correlated)
    )
    log(kind, "R2", round(float(r2.mean()), 4), "| within +-2 of proportional depth", round(float(near.mean()), 3), f"(chance {5 / LS:.3f})",
        "| same-head", round(float(same.mean()), 3), "(chance 0.125)", "| corr(com, depth)", round(summ[kind]["corr_centre_of_mass_vs_proportional_depth"], 3),
        "| blocks for 50/90%", summ[kind]["source_blocks_for_50pct"], summ[kind]["source_blocks_for_90pct"],
        "| layers for 50/90%", summ[kind]["source_layers_for_50pct"], summ[kind]["source_layers_for_90pct"])
    log(kind, "top source layer per target layer", summ[kind]["top_source_layer_by_target_layer"])
    log(kind, "source layer total share (%)", [round(100 * x, 1) for x in tot_src.tolist()])

np.savez_compressed(f"{HOME}/kvcodesign/results/depgraph.npz", **arr)
json.dump(summ, open(f"{HOME}/kvcodesign/results/depgraph.json", "w"), indent=1)
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    for i, kind in enumerate("kv"):
        lay = np.array(summ[kind]["layer_share"]); layn = lay / lay.sum(0, keepdims=True)
        im = ax[i, 0].imshow(layn, aspect="auto", origin="lower", cmap="viridis")
        ax[i, 0].plot(np.arange(LT), np.arange(LT) * (LS - 1) / (LT - 1), "w--", lw=0.8)
        ax[i, 0].set(title=f"{kind.upper()}: share of each source layer (columns sum to 1)", xlabel="target layer (Qwen3-4B)", ylabel="source layer (Qwen3-1.7B)")
        plt.colorbar(im, ax=ax[i, 0])
        b = np.clip(arr[f"{kind}_share"], 0, None)
        im = ax[i, 1].imshow(b / b.sum(0, keepdims=True), aspect="auto", origin="lower", cmap="viridis", vmax=0.1)
        ax[i, 1].set(title=f"{kind.upper()}: share of each source (layer, head) block", xlabel="target layer x head (288)", ylabel="source layer x head (224)")
        plt.colorbar(im, ax=ax[i, 1])
    plt.tight_layout(); plt.savefig(f"{HOME}/kvcodesign/results/depgraph.png", dpi=110)
except Exception as e:   # the figure is optional
    log("plot skipped:", repr(e))
log("DEPGRAPHDONE")
