from transformers import AutoTokenizer
import ks, xalign
class M:
    def __init__(s, n, fam):
        s.tok = AutoTokenizer.from_pretrained(f"{ks.ROOT}/models/{n}"); s.family = fam
    bos = ks.Model.bos
A, B = M("Llama-3.2-3B-Instruct", "llama"), M("Qwen3-1.7B", "qwen3")
msgs = [("user", "Hello there, what's the capital of Côte d'Ivoire?  Answer:  12,345.67 tokens!"), ("assistant", "Yamoussoukro — officially.")]
ia, ib, mp = xalign.conv_aligned(A, B, msgs)
print(len(ia), len(ib), (mp >= 0).float().mean().item())
for j in range(len(ib)):
    print(repr(B.tok.decode([ib[j]])), "<-", repr(A.tok.decode([ia[mp[j]]])) if mp[j] >= 0 else "SPECIAL")
