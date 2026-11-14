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
        if not all(d in done for d in deps):
            continue
        free = [g for g in GPUS if streak[g] >= 3 and g not in {x for _, x in running.values()}]
        if not free:
            break
        g = free[0]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), HF_ENDPOINT="https://hf-mirror.com")
        p = subprocess.Popen(cmd, shell=True, env=env, stdout=open(f"{L}/{lg}", "a"), stderr=subprocess.STDOUT)
        running[name] = (p, g); pending.remove(job); streak[g] = 0
        log("launched", name, "on gpu", g)
    time.sleep(60)
log("QUEUEDONE done", sorted(done), "failed", sorted(failed))
