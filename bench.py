"""Time prompt reading (prefill) and generation via Ollama's own counters.
Usage: python bench.py <model> [label] [num_gpu]   (num_gpu 0 = force CPU, for comparison)"""
import json, sys, urllib.request

model = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else "default"
opts = {"num_ctx": 4096, "num_predict": 128, "temperature": 0, "seed": 1}
if len(sys.argv) > 3:
    opts["num_gpu"] = int(sys.argv[3])  # 0 = force CPU, 999 = offload all layers

filler = " ".join(f"Line {i}: the service retries the request with exponential backoff and logs the error code." for i in range(170))
prompt = filler + "\n\nIn three sentences, summarize what the lines above describe."

def run(p):
    body = json.dumps({"model": model, "prompt": p, "stream": False, "think": False, "options": opts}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))

run("hi")  # load model
r = run(prompt)
pp = r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9)
tg = r["eval_count"] / (r["eval_duration"] / 1e9)
print(f"{label:>12}: prefill {r['prompt_eval_count']} tok @ {pp:6.1f} tok/s ({r['prompt_eval_duration']/1e9:5.1f}s) | "
      f"generate {r['eval_count']} tok @ {tg:5.1f} tok/s")
