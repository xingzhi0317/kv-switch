# Polite GPU job queue for a shared server: a GPU counts as free only if it has NO compute process and < 1 GB used
# for 3 consecutive polls (60 s apart), so we never grab a card another user's job is between phases on.
# One job per GPU. Jobs may depend on other jobs. Log: ~/kvcodesign/logs/queue.log
import os, subprocess, time, json
HOME = os.path.expanduser("~")
PY = f"{HOME}/kvbase/.venv/bin/python"
GPUS = [int(g) for g in os.environ.get("GPUS", "0,1,2,3,4,5,6,7").split(",")]
L = f"{HOME}/kvcodesign/logs"
RR = "rrr_k768_v1536,rrr_k768_v1536_bf16"
CB = f"{HOME}/kvbase/mappers/cachebridge_17to4"


def ev(task, methods, tag, n=200, extra=""):
    out = f"{HOME}/kvcodesign/results/quality_{task}_{tag}.json"
    return f"cd {HOME}/kvcodesign && TASK={task} N={n} METHODS={methods} OUT={out} {extra} {PY} eval_quality.py"


JOBS = [  # name, deps, command, log
    ("fit_cb", [], f"cd {HOME}/kvbase && {PY} fit_cb2.py", "fit_cb2.log"),
    ("q_quality", [], ev("quality", f"native,{RR},rrr_k256_v1024,rrr_k2048_v2048", "rrr"), "q_quality_rrr.log"),
    ("q_hotpot", [], ev("hotpotqa_e", f"native,{RR},rrr_k256_v1024,rrr_k2048_v2048", "rrr"), "q_hotpot_rrr.log"),
    ("q_gsm_a", [], ev("gsm8k", f"native,{RR}", "rrr_a"), "q_gsm_rrr_a.log"),
    ("q_gsm_b", [], ev("gsm8k", "rrr_k256_v1024,rrr_k2048_v2048", "rrr_b"), "q_gsm_rrr_b.log"),
    ("cb_quality", ["fit_cb"], ev("quality", "cb_official", "cbfull", extra=f"MAPPER_CB={CB}"), "q_quality_cbfull.log"),
    ("cb_hotpot", ["fit_cb"], ev("hotpotqa_e", "cb_official", "cbfull", extra=f"MAPPER_CB={CB}"), "q_hotpot_cbfull.log"),
    ("cb_gsm", ["fit_cb"], ev("gsm8k", "cb_official", "cbfull", extra=f"MAPPER_CB={CB}"), "q_gsm_cbfull.log"),
]
if os.environ.get("SET") == "stage1":   # PLAN_v5 stage 1: exact-information tasks, all baselines (two method groups per task)
    GA = "native,src_native,kvbridge,lrf_bf16"
    GB = f"{RR},rrr_k2048_v2048,cb_official"
    JOBS = []
    for task, n in (("squad", 200), ("kvret", 200), ("passkey_2k", 200), ("passkey_8k", 100), ("passkey_16k", 100)):
        JOBS.append((f"{task}_a", [], ev(task, GA, "a", n), f"s1_{task}_a.log"))
        JOBS.append((f"{task}_b", [], ev(task, GB, "b", n, extra=f"MAPPER_CB={CB}"), f"s1_{task}_b.log"))
    # gsm8k_h already has native / src_native / kvbridge / lrf_bf16 (quality_gsm8k_h_h.json); add the rest
    JOBS.append(("gsm8k_h_b", [], ev("gsm8k_h", GB, "b", extra=f"MAPPER_CB={CB}"), "s1_gsm8k_h_b.log"))
    # re-run of the first group with the new recorder (output length, truncation, score at 256 tokens)
    JOBS.append(("gsm8k_h_a", [], ev("gsm8k_h", GA, "a2"), "s1_gsm8k_h_a2.log"))
if os.environ.get("SET") == "stage3a":   # PLAN_v5 stage 3 step 1: token identity (exact layer 0, identity-augmented low rank)
    GT = "kvbridge_l0,rrr_k768_v1536_l0,tid_k768_v1536,tid_k2048_v2048"
    JOBS = [(f"{task}_tid", [], ev(task, GT, "tid", n), f"s3_{task}_tid.log")
            for task, n in (("kvret", 200), ("passkey_2k", 200), ("squad", 200), ("passkey_8k", 100), ("gsm8k_h", 200),
                            ("quality", 200), ("hotpotqa_e", 200), ("passkey_16k", 100))]
SEL = os.environ.get("JOBS")
if SEL:
    JOBS = [j for j in JOBS if j[0] in SEL.split(",")]
    JOBS.sort(key=lambda j: SEL.split(",").index(j[0]))
    JOBS = [(n, [("log:fit_cb2.log:FITCBDONE" if d == "fit_cb" else d) for d in deps], c, lg) for n, deps, c, lg in JOBS]
log = lambda *a: print(time.strftime("%m-%d %H:%M:%S"), *a, flush=True)


def gpu_state():
    used = {}
    for line in subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,uuid", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.strip().splitlines():
        i, m, u = [x.strip() for x in line.split(",")]
        used[int(i)] = (int(m), u)
    busy = set(x.strip() for x in subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                                                 capture_output=True, text=True).stdout.split())
    return {i: (m < 1000 and u not in busy) for i, (m, u) in used.items()}


streak = {g: 0 for g in GPUS}
running, done, failed = {}, set(), set()
pending = list(JOBS)
log("queue start", [j[0] for j in JOBS], "gpus", GPUS)
while pending or running:
    for name, (p, g) in list(running.items()):
        rc = p.poll()
        if rc is not None:
            (done if rc == 0 else failed).add(name)
            log("finished", name, "gpu", g, "rc", rc)
            del running[name]
    st = gpu_state()
    mine = {g for _, g in running.values()}
    for g in GPUS:
        streak[g] = streak[g] + 1 if (st.get(g) and g not in mine) else 0
    for job in list(pending):
        name, deps, cmd, lg = job
        if any(d in failed for d in deps):
            pending.remove(job); failed.add(name); log("skip (dep failed)", name); continue
        ok = lambda d: (d in done) or (d.startswith("log:") and os.path.exists(f"{L}/{d.split(':')[1]}")
                                         and d.split(":")[2] in open(f"{L}/{d.split(':')[1]}").read())
        if not all(ok(d) for d in deps):
            continue
        free = [g for g in GPUS if streak[g] >= 3 and g not in {x for _, x in running.values()}]
        if not free:
            break
        g = free[0]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), HF_ENDPOINT="https://hf-mirror.com", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        p = subprocess.Popen(cmd, shell=True, env=env, stdout=open(f"{L}/{lg}", "a"), stderr=subprocess.STDOUT)
        running[name] = (p, g); pending.remove(job); streak[g] = 0
        log("launched", name, "on gpu", g)
    time.sleep(60)
log("QUEUEDONE done", sorted(done), "failed", sorted(failed))
