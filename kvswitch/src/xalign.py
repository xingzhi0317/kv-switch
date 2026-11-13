# Case 2 (cross-tokenizer) alignment + translator.
# Conversation is built piecewise for each model: template pieces (role headers etc.) + content.
# Content tokens are aligned by characters: B token j <- the A token that covers the last character of j
# (KV-Lingo-style character-coverage alignment, simplified to one source token per target token).
# Template / special positions have no counterpart: B computes them itself (forward_at), as in KV-Lingo.
import torch
from ks import *
from translate import forward_at


def _pieces(M, role, text):
    if M.family == "qwen3":
        return f"<|im_start|>{role}\n", text, "<|im_end|>\n"
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n", text, "<|eot_id|>"


def conv_aligned(A, B, msgs):
    """Returns idsA, idsB, mapB2A (LongTensor, -1 = B-only template token)."""
    ia, ib, mp = list(A.bos()), list(B.bos()), [-1] * len(B.bos())
    for role, text in msgs:
        ha, c, fa = _pieces(A, role, text)
        hb, _, fb = _pieces(B, role, text)
        ia += A.tok(ha, add_special_tokens=False).input_ids
        hbi = B.tok(hb, add_special_tokens=False).input_ids
        ib += hbi; mp += [-1] * len(hbi)
        ea = A.tok(c, add_special_tokens=False, return_offsets_mapping=True)
        eb = B.tok(c, add_special_tokens=False, return_offsets_mapping=True)
        a0 = len(ia)
        ia += ea.input_ids
        # char -> A token index
        ends = [e for _, e in ea.offset_mapping]
        j = 0
        for tid, (s, e) in zip(eb.input_ids, eb.offset_mapping):
            last = max(e - 1, s)
            while j < len(ends) - 1 and ends[j] <= last:
                j += 1
            ib.append(tid); mp.append(a0 + j if ends else -1)
        ia += A.tok(fa, add_special_tokens=False).input_ids
        fbi = B.tok(fb, add_special_tokens=False).input_ids
        ib += fbi; mp += [-1] * len(fbi)
    return ia, ib, torch.tensor(mp)


class XTranslator:
    def __init__(self, A, B, maps, k=3, vkind="WV", lowrank=None):
        self.A, self.B = A, B
        m = maps[k]
        WK, WV = m["WK"].float(), m[vkind].float()
        if lowrank is not None:
            WK = WK + lowrank["UK"].float() @ lowrank["VK"].float()
            WV = WV + lowrank["UV"].float() @ lowrank["VV"].float()
        self.WK, self.WV, self.src = WK.to(torch.bfloat16).to(A.dev), WV.to(torch.bfloat16).to(A.dev), m["src"]

    @torch.no_grad()
    def translate(self, cA, cB, idsB, mp):
        A, B = self.A, self.B
        NB = len(mp)
        mp = mp.to(A.dev)
        posA = mp.clamp_min(0)
        cosA, sinA = A.cossin(0, int(posA.max()) + 1)
        cosA, sinA = cosA[posA], sinA[posA]
        cosB, sinB = B.cossin(0, NB)
        for l in range(B.L):
            if cB.k[l] is None:
                cB.alloc(l)
            xk = torch.cat([derope(cA.k[s][:, posA].float(), cosA, sinA).to(torch.bfloat16).transpose(0, 1).reshape(NB, -1) for s in self.src[l]], -1)
            yk = (xk @ self.WK[l][:-1] + self.WK[l][-1]).view(NB, B.Hk, B.D).transpose(0, 1)
            cB.k[l][:, :NB] = rope(yk.float(), cosB, sinB).to(torch.bfloat16)
            xv = torch.cat([cA.v[s][:, posA].transpose(0, 1).reshape(NB, -1) for s in self.src[l]], -1)
            cB.v[l][:, :NB] = (xv @ self.WV[l][:-1] + self.WV[l][-1]).view(NB, B.Hk, B.D).transpose(0, 1)
        cB.n = NB
        sp = (mp < 0).nonzero().flatten().to(B.dev)
        forward_at(B, idsB[:, sp], sp, cB, NB)  # B's own KV on template / special tokens (incl. sink)
