# Figures for the stage-2 diagnosis (run locally on the synced results):  python plot_stage2.py
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
LS, LT = 28, 36
R = "results/"
dep, arr = json.load(open(R + "depgraph.json")), np.load(R + "depgraph.npz")
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
for i, kind in enumerate("kv"):
    lay = np.array(dep[kind]["layer_share"]); layn = lay / lay.sum(0, keepdims=True)
    im = ax[i, 0].imshow(layn, aspect="auto", origin="lower", cmap="viridis")
    ax[i, 0].plot(np.arange(LT), np.arange(LT) * (LS - 1) / (LT - 1), "w--", lw=0.8, label="proportional depth")
    ax[i, 0].plot(np.arange(LT), dep[kind]["centre_of_mass_source_layer"], "r-", lw=1, label="centre of mass")
    ax[i, 0].legend(loc="upper left", fontsize=8)
    ax[i, 0].set(title=f"{kind.upper()}: which source layer each target layer reads (columns sum to 1)",
                 xlabel="target layer (Qwen3-4B)", ylabel="source layer (Qwen3-1.7B)")
    plt.colorbar(im, ax=ax[i, 0])
    b = np.clip(arr[f"{kind}_share"], 0, None); b = b / b.sum(0, keepdims=True)
    im = ax[i, 1].imshow(b, aspect="auto", origin="lower", cmap="viridis", vmax=np.percentile(b, 99.5))
    ax[i, 1].set(title=f"{kind.upper()}: source (layer, head) -> target (layer, head), 224 x 288",
                 xlabel="target layer x head", ylabel="source layer x head")
    plt.colorbar(im, ax=ax[i, 1])
plt.tight_layout(); plt.savefig(R + "depgraph.png", dpi=110); plt.close()

te = json.load(open(R + "tokerr.json"))
T = ["digit", "cont", "cap", "punct", "word"]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for j, (x, name) in enumerate((("k", "K error"), ("v", "V error"), ("logit", "attention-logit error"))):
    for mi, (m, lab) in enumerate((("dense", "dense"), ("lowrank_k768_v1536", "low-rank K768+V1536"))):
        v = te["by_type"][f"{m}.{x}"]
        ax[j].bar(np.arange(len(T)) + (mi - 0.5) * 0.38, [v[t]["mean"] for t in T], 0.38, label=lab)
    ax[j].set_xticks(np.arange(len(T))); ax[j].set_xticklabels([f"{t}\n(n={te['counts'][t]})" for t in T])
    ax[j].set(title=f"{name} by token type (1.0 = predicting the mean)"); ax[j].legend(fontsize=8)
plt.tight_layout(); plt.savefig(R + "tokerr_types.png", dpi=110); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for j, x in enumerate(("k", "v")):
    for t in ("digit", "cont", "cap", "word"):
        ax[j].plot(te["by_type_layer"][f"lowrank_k768_v1536.{x}"][t], label=t)
    ax[j].set(title=f"{x.upper()} error by target layer and token type (low-rank K768+V1536)", xlabel="target layer", ylabel="relative error"); ax[j].legend()
plt.tight_layout(); plt.savefig(R + "tokerr_layers.png", dpi=110); plt.close()
print("PLOTS OK")
