"""tinyagent: a Claude-Code-style agent with a tiny prompt, built for slow local GPUs (Intel iGPU + Ollama).

It explores on its own (list / find / grep / read), and asks before it writes files or runs commands.
Usage:  python tinyagent.py                 # interactive, in the current folder
        python tinyagent.py -p "question"   # one-shot
Env:    TINYAGENT_MODEL (default qwen3-coder:30b), TINYAGENT_CTX (default 16384), OLLAMA_HOST
Stdlib only.
"""
import argparse, fnmatch, json, os, re, subprocess, sys, threading, time, urllib.request
from pathlib import Path

MODEL = os.environ.get("TINYAGENT_MODEL", "qwen3-coder:30b")
CTX = int(os.environ.get("TINYAGENT_CTX", "16384"))
HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not HOST.startswith("http"):
    HOST = "http://" + HOST
ROOT = Path.cwd().resolve()
SKIP = {".git", "node_modules", ".venv", "venv", "site-packages", "__pycache__", ".obsidian", ".trash", "dist", "build"}
MAX_OUT = 6000  # chars per tool result; keeps the context (and prefill time) small

SYSTEM = f"""You are a coding and research agent working in the folder {ROOT}.
Use tools to explore before answering: find_files or grep to locate things, then read_file.
Never guess file contents. Prefer few, targeted tool calls. Paths are relative to the folder.
When you have enough information, answer concisely and name the files you used."""


def safe(path: str) -> Path:
    """Resolve a user/model path and refuse anything outside ROOT."""
    p = (ROOT / path).resolve()
    if p != ROOT and ROOT not in p.parents:
        raise ValueError(f"path outside workspace: {path}")
    return p


def walk():
    for dirpath, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for f in files:
            yield Path(dirpath, f)


def list_dir(path="."):
    p = safe(path)
    rows = [f"{c.name}/" if c.is_dir() else c.name for c in sorted(p.iterdir()) if c.name not in SKIP]
    return "\n".join(rows) or "(empty)"


def find_files(pattern):
    pat = pattern.lower()
    hits = [str(f.relative_to(ROOT)) for f in walk()
            if fnmatch.fnmatch(f.name.lower(), pat) or fnmatch.fnmatch(str(f.relative_to(ROOT)).lower(), pat)]
    return "\n".join(hits[:200]) or "no matches"


def grep(pattern, glob="*"):
    rx = re.compile(pattern, re.IGNORECASE)
    out = []
    for f in walk():
        if not fnmatch.fnmatch(f.name, glob) or f.stat().st_size > 2_000_000:
            continue
        try:
            for n, line in enumerate(f.open(encoding="utf-8", errors="ignore"), 1):
                if rx.search(line):
                    out.append(f"{f.relative_to(ROOT)}:{n}: {line.strip()[:200]}")
                    if len(out) >= 100:
                        return "\n".join(out) + "\n(truncated at 100 matches)"
        except OSError:
            pass
    return "\n".join(out) or "no matches"


def read_file(path, start=1, lines=300):
    text = safe(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    chunk = text[start - 1:start - 1 + lines]
    more = f"\n... ({len(text)} lines total; call again with start={start + lines})" if start - 1 + lines < len(text) else ""
    return "\n".join(chunk) + more


def confirm(what):
    return input(f"\n  allow {what}? [y/N] ").strip().lower() == "y"


def write_file(path, content):
    p = safe(path)
    if not confirm(f"write {p.relative_to(ROOT)} ({len(content)} chars)"):
        return "user denied the write"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {path}"


def run(command):
    if not confirm(f"run `{command}`"):
        return "user denied the command"
    r = subprocess.run(command, shell=True, cwd=ROOT, capture_output=True, text=True, timeout=300)
    return f"exit {r.returncode}\n{r.stdout}{r.stderr}"


TOOLS = {f.__name__: f for f in (list_dir, find_files, grep, read_file, write_file, run)}
SPECS = [
    ("list_dir", "List a folder.", {"path": "folder, default ."}),
    ("find_files", "Find files by name glob, e.g. *readme* or *.py (case-insensitive).", {"pattern": "glob"}),
    ("grep", "Search file contents with a regex (case-insensitive).", {"pattern": "regex", "glob": "file filter, e.g. *.md"}),
    ("read_file", "Read a text file, 300 lines at a time.", {"path": "file", "start": "first line, default 1"}),
    ("write_file", "Create or overwrite a file (user must approve).", {"path": "file", "content": "full new content"}),
    ("run", "Run a shell command in the folder (user must approve).", {"command": "command line"}),
]
REQUIRED = {"list_dir": [], "find_files": ["pattern"], "grep": ["pattern"], "read_file": ["path"],
            "write_file": ["path", "content"], "run": ["command"]}
TOOL_SCHEMA = [{"type": "function", "function": {
    "name": n, "description": d,
    "parameters": {"type": "object", "required": REQUIRED[n],
                   "properties": {k: {"type": "integer" if k == "start" else "string", "description": v} for k, v in p.items()}}}}
    for n, d, p in SPECS]


TEXT_CALL = re.compile(r"<function=(\w+)>(.*?)</function>", re.S)
PARAM = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.S)


def text_tool_calls(text):
    """qwen3-coder sometimes writes its tool call as raw XML text that Ollama doesn't parse. Recover it."""
    return [{"function": {"name": n, "arguments": {k: (int(v) if k == "start" and v.isdigit() else v)
                                                   for k, v in PARAM.findall(body)}}}
            for n, body in TEXT_CALL.findall(text) if n in TOOLS]


def ticker(stop, step, t0):
    """Live status on stderr until the first token arrives (prompt reading / model load is the silent part)."""
    while not stop.wait(0.5):
        print(f"\r  [step {step}] reading context... {time.time() - t0:4.0f}s", end="", file=sys.stderr, flush=True)


def chat(messages, step):
    """Stream one model call: show a timer, then print answer text as it is generated."""
    body = json.dumps({"model": MODEL, "messages": messages, "tools": TOOL_SCHEMA, "stream": True,
                       "options": {"num_ctx": CTX, "temperature": 0.2}}).encode()
    req = urllib.request.Request(f"{HOST}/api/chat", body, {"Content-Type": "application/json"})
    t0, stop = time.time(), threading.Event()
    threading.Thread(target=ticker, args=(stop, step, t0), daemon=True).start()
    content, calls, last, first = [], [], {}, None
    try:
        for line in urllib.request.urlopen(req, timeout=1800):
            last = json.loads(line)
            m = last.get("message", {})
            if first is None and (m.get("content") or m.get("tool_calls")):
                first = time.time()
                stop.set()
                print(f"\r  [step {step}] read context in {first - t0:.0f}s" + " " * 10, file=sys.stderr, flush=True)
            if m.get("content"):
                content.append(m["content"])
                so_far = "".join(content)
                if "<function" not in so_far and "<tool_call" not in so_far:  # hide raw tool-call markup
                    print(m["content"], end="", flush=True)  # live answer
            calls += m.get("tool_calls") or []
    finally:
        stop.set()
    if content:
        print()
    text = "".join(content)
    if not calls:
        calls = text_tool_calls(text)
        if calls:
            content = [text[:text.find("<")].strip()]  # keep the preamble, drop the markup
    gen = last.get("eval_count", 0) / max(last.get("eval_duration", 1) / 1e9, 1e-9)
    print(f"  [step {step}] done in {time.time() - t0:.0f}s | {last.get('prompt_eval_count', 0)} tok in, "
          f"{gen:.1f} tok/s out", file=sys.stderr)
    return {"role": "assistant", "content": "".join(content), "tool_calls": calls} if calls else \
           {"role": "assistant", "content": "".join(content)}


def turn(messages, max_steps=15):
    for step in range(1, max_steps + 1):
        msg = chat(messages, step)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            return ""  # answer was already streamed to the screen
        for c in calls:
            name, args = c["function"]["name"], c["function"].get("arguments") or {}
            print(f"  -> {name}({json.dumps(args)[:120]})", file=sys.stderr)
            try:
                out = str(TOOLS[name](**args))
            except Exception as e:  # bad args / missing file: tell the model, let it recover
                out = f"error: {e}"
            if len(out) > MAX_OUT:
                out = out[:MAX_OUT] + f"\n... (truncated, {len(out)} chars total)"
            print(f"     {len(out)} chars back", file=sys.stderr)
            messages.append({"role": "tool", "content": out, "tool_name": name})
    return "(stopped: too many steps)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prompt")
    a = ap.parse_args()
    messages = [{"role": "system", "content": SYSTEM}]
    if a.prompt:
        messages.append({"role": "user", "content": a.prompt})
        turn(messages)
        return
    print(f"tinyagent | {MODEL} | {ROOT} | Ctrl+C to quit")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if q:
            messages.append({"role": "user", "content": q})
            print()
            turn(messages)


if __name__ == "__main__":
    main()
