# Is "same head index" special? R^2 of single (source layer, source head) -> (target layer, target head) regressions.
import torch, json
sk, sv, tk, tv = torch.load("/home/wxz/kvbase/mappers/calib_17to4.pt")
support, layers, R2L = torch.load("/home/wxz/kvbase/mappers/cb_init_17to4.pt")
R2L = torch.as_tensor(R2L)
dev = "cuda"
n = 32768
idx = torch.randperm(sk[0].shape[0], generator=torch.Generator().manual_seed(0))[:n]
H = 8


def r2(x, y):  # x [n,d], y [n,d] centered ridge
    x = x - x.mean(0); y = y - y.mean(0)
    W = torch.linalg.solve(x.T @ x + 1e-2 * torch.eye(x.shape[1], device=dev), x.T @ y)
    return 1 - ((y - x @ W) ** 2).sum() / (y ** 2).sum()


out = {}
for t in [4, 12, 20, 28, 35]:
    s = int(R2L[t].argmax())  # best single source layer for this target layer
    for kind, S, T in (("K", sk, tk), ("V", sv, tv)):
        X = S[s][idx].to(dev).float(); Y = T[t][idx].to(dev).float()
        M = torch.zeros(H, H)
        for ht in range(H):
            for hs in range(H):
                M[ht, hs] = r2(X[:, hs], Y[:, ht]).item()
        diag = M.diag().mean().item()
        best = M.max(1).values.mean().item()
        rank_of_diag = [(M[h] > M[h, h]).sum().item() for h in range(H)]
        allheads = r2(X.reshape(n, -1), Y.reshape(n, -1)).item()
        out[f"t{t}_{kind}"] = dict(src_layer=s, diag_mean=diag, best_match_mean=best, offdiag_mean=((M.sum() - M.diag().sum()) / (H * H - H)).item(),
                                   diag_rank=rank_of_diag, all_heads_one_layer=allheads)
        print(t, kind, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[f"t{t}_{kind}"].items()}, flush=True)
json.dump(out, open("/home/wxz/kvbase/results/head_diag.json", "w"), indent=1)
