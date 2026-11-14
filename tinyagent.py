"""tinyagent: a Claude-Code-style coding agent with a tiny prompt, built for slow local GPUs.

Runs on a local model through Ollama (default) or any OpenAI-compatible API (OpenRouter, OpenAI, LM Studio...).
It explores on its own, edits files with a diff preview, and asks before it changes anything.

    python tinyagent.py                          # interactive, in the current folder
    python tinyagent.py -p "fix the failing test" # one-shot
    python tinyagent.py -c                        # resume the last session in this folder
    python tinyagent.py --provider openrouter     # same agent, cloud model (needs OPENROUTER_API_KEY)

Type /help inside for commands. Stdlib only.
"""
import argparse, difflib, fnmatch, html.parser, json, os, re, subprocess, sys, threading, time
import urllib.error, urllib.request
from pathlib import Path

VERSION = "0.3.0"
ROOT = Path.cwd().resolve()
SKIP = {".git", "node_modules", ".venv", "venv", "site-packages", "__pycache__", ".obsidian", ".trash",
        "dist", "build", ".tinyagent"}
MAX_OUT = int(os.environ.get("TINYAGENT_MAX_OUT", "6000"))  # chars per tool result: keeps prefill cheap
HOME = Path(os.environ.get("TINYAGENT_HOME", str(Path.home() / ".tinyagent")))
WIN = sys.platform == "win32"
SHELL = "PowerShell" if WIN else "bash"

PROVIDERS = {  # name -> (base url, default model, api-key env var)
    "ollama": (os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
               os.environ.get("TINYAGENT_MODEL", "qwen3-coder:30b"), None),
    "openrouter": ("https://openrouter.ai/api/v1", os.environ.get("TINYAGENT_OPENROUTER_MODEL", "qwen/qwen3-coder"),
                   "OPENROUTER_API_KEY"),
    "openai": (os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
               os.environ.get("TINYAGENT_OPENAI_MODEL", "gpt-4.1-mini"), "OPENAI_API_KEY"),
}
MODES = ("ask", "auto", "yolo", "readonly")  # auto = edits auto-approved; yolo = everything auto-approved


class State:
    provider = "ollama"
    model = PROVIDERS["ollama"][1]
    ctx = int(os.environ.get("TINYAGENT_CTX", "16384"))
    mode = os.environ.get("TINYAGENT_MODE", "ask")
    messages: list = []
    undo: list = []        # stack of (Path, previous text or None, eol)
    allow: set = set()     # tools the user said "always" to, this session
    last_in = 0            # prompt tokens of the last call (from the server)
    session: Path = None
    calls_made = 0


S = State()

# ---------------------------------------------------------------- terminal helpers
if WIN:
    os.system("")  # enables ANSI colors in the Windows console
TTY = sys.stderr.isatty()


def c(code, s):
    return f"\033[{code}m{s}\033[0m" if TTY else s


def dim(s): return c("2", s)
def info(s): print(dim(s), file=sys.stderr, flush=True)


# ---------------------------------------------------------------- files
def safe(path: str) -> Path:
    """Resolve a path from the model or user and refuse anything outside ROOT."""
    p = (ROOT / str(path).strip().strip('"')).resolve()
    if p != ROOT and ROOT not in p.parents:
        raise ValueError(f"path outside workspace: {path}")
    return p


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/") or "."


def read_text(p: Path):
    """Read as LF text and remember the file's line ending, so edits don't flip CRLF/LF."""
    raw = p.read_bytes().decode("utf-8", errors="replace")
    eol = "\r\n" if "\r\n" in raw else "\n"
    return raw.replace("\r\n", "\n"), eol


def write_text(p: Path, text: str, eol="\n"):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\r\n", "\n").replace("\n", eol))


def walk(base: Path = None):
    for dirpath, dirs, files in os.walk(base or ROOT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP)
        for f in sorted(files):
            yield Path(dirpath, f)


def show_diff(old: str, new: str, name: str, limit=60):
    lines = list(difflib.unified_diff(old.splitlines(), new.splitlines(), f"a/{name}", f"b/{name}", lineterm="", n=2))
    for ln in lines[:limit]:
        color = "32" if ln.startswith("+") and not ln.startswith("+++") else \
                "31" if ln.startswith("-") and not ln.startswith("---") else "36" if ln.startswith("@@") else "2"
        print("    " + c(color, ln), file=sys.stderr)
    if len(lines) > limit:
        info(f"    ... {len(lines) - limit} more diff lines")


def approve(tool, summary, preview=None):
    """Permission gate. Returns (allowed, reason)."""
    risky = tool in ("run", "fetch_url")
    if S.mode == "readonly" and tool in ("edit_file", "write_file", "run"):
        return False, "blocked: read-only mode (user can switch with /mode)"
    if S.mode == "yolo" or tool in S.allow or (S.mode == "auto" and not risky):
        if preview:
            preview()
        return True, ""
    if preview:
        preview()
    try:
        ans = input(c("33", f"  allow {summary}? [y]es / [n]o / [a]lways for {tool} > ")).strip().lower()
    except EOFError:
        ans = "n"
    if ans == "a":
        S.allow.add(tool)
        return True, ""
    if ans == "y":
        return True, ""
    return False, f"user denied: {summary}" + (f" (user said: {ans})" if ans not in ("", "n", "no") else "")


# ---------------------------------------------------------------- tools
def list_dir(path="."):
    p = safe(path)
    rows = []
    for ch in sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
        if ch.name in SKIP:
            continue
        rows.append(f"{ch.name}/" if ch.is_dir() else f"{ch.name}  ({ch.stat().st_size:,} B)")
    return "\n".join(rows) or "(empty)"


def find_files(pattern, path="."):
    pat = pattern.lower()
    base = safe(path)
    hits = [rel(f) for f in walk(base)
            if fnmatch.fnmatch(f.name.lower(), pat) or fnmatch.fnmatch(rel(f).lower(), pat)]
    return "\n".join(hits[:200]) + ("\n(first 200 shown)" if len(hits) > 200 else "") if hits else "no matches"


def grep(pattern, glob="*", path="."):
    rx = re.compile(pattern, re.IGNORECASE)
    out = []
    for f in walk(safe(path)):
        if not fnmatch.fnmatch(f.name.lower(), glob.lower()) or f.stat().st_size > 2_000_000:
            continue
        try:
            with f.open(encoding="utf-8", errors="ignore") as fh:
                for n, line in enumerate(fh, 1):
                    if rx.search(line):
                        out.append(f"{rel(f)}:{n}: {line.strip()[:200]}")
                        if len(out) >= 100:
                            return "\n".join(out) + "\n(stopped at 100 matches; narrow the pattern or path)"
        except OSError:
            pass
    return "\n".join(out) or "no matches"


def read_file(path, start=1, lines=250):
    text, _ = read_text(safe(path))
    all_lines = text.split("\n")
    start = max(1, int(start))
    chunk = all_lines[start - 1:start - 1 + int(lines)]
    body = "\n".join(f"{i:>5}| {ln}" for i, ln in enumerate(chunk, start))
    end = start - 1 + len(chunk)
    return body + (f"\n... (lines {start}-{end} of {len(all_lines)}; call again with start={end + 1})"
                   if end < len(all_lines) else "")


LINE_PREFIX = re.compile(r"^\s*\d+\| ?", re.M)


def edit_file(path, old, new, replace_all=False):
    p = safe(path)
    text, eol = read_text(p)
    old, new = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
    if not old:
        return "error: old is empty. To create a file use write_file."
    n = text.count(old)
    if n == 0 and LINE_PREFIX.search(old):  # model copied read_file's line numbers: strip them and retry
        old, new = LINE_PREFIX.sub("", old), LINE_PREFIX.sub("", new)
        n = text.count(old)
    if n == 0:
        return ("error: old text not found in the file. Call read_file again and copy the exact lines "
                "(without the 'NN| ' prefix), including indentation.")
    if n > 1 and not replace_all:
        return f"error: old text appears {n} times. Include more surrounding lines to make it unique, or set replace_all."
    updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    ok, why = approve("edit_file", f"edit {rel(p)}", lambda: show_diff(text, updated, rel(p)))
    if not ok:
        return why
    S.undo.append((p, text, eol))
    write_text(p, updated, eol)
    return f"edited {rel(p)} ({n if replace_all else 1} replacement{'s' if replace_all and n > 1 else ''})"


def write_file(path, content):
    p = safe(path)
    old, eol = read_text(p) if p.exists() else (None, "\r\n" if WIN else "\n")
    preview = (lambda: show_diff(old, content, rel(p))) if old is not None else \
              (lambda: [print("    " + c("32", "+" + ln), file=sys.stderr) for ln in content.split("\n")[:20]])
    ok, why = approve("write_file", f"{'overwrite' if old is not None else 'create'} {rel(p)} ({len(content)} chars)", preview)
    if not ok:
        return why
    S.undo.append((p, old, eol))
    write_text(p, content, eol)
    return f"wrote {rel(p)}"


def run(command, timeout=120):
    ok, why = approve("run", f"run `{command}`")
    if not ok:
        return why
    argv = ["powershell", "-NoProfile", "-Command", command] if WIN else ["bash", "-lc", command]
    try:
        r = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=int(timeout),
                           encoding="utf-8", errors="replace")
        return f"exit {r.returncode}\n{r.stdout}{r.stderr}".strip()
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {timeout}s"


class _Text(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        self.skip += tag in ("script", "style", "noscript", "svg")

    def handle_endtag(self, tag):
        self.skip -= tag in ("script", "style", "noscript", "svg") and self.skip > 0

    def handle_data(self, d):
        if not self.skip and d.strip():
            self.out.append(d.strip())


def fetch_url(url):
    if not re.match(r"https?://", url):
        return "error: only http(s) URLs"
    ok, why = approve("fetch_url", f"fetch {url}")
    if not ok:
        return why
    req = urllib.request.Request(url, headers={"User-Agent": f"tinyagent/{VERSION}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        body = r.read(2_000_000).decode(r.headers.get_content_charset() or "utf-8", errors="replace")
        if "html" in (r.headers.get_content_type() or ""):
            p = _Text()
            p.feed(body)
            body = "\n".join(p.out)
    return body


TOOLS = {f.__name__: f for f in (list_dir, find_files, grep, read_file, edit_file, write_file, run, fetch_url)}
SPECS = [  # (name, description, {param: (type, description)}, required)
    ("list_dir", "List a folder.", {"path": ("string", "folder, default .")}, []),
    ("find_files", "Find files by name glob (case-insensitive), e.g. *.py or *readme*.",
     {"pattern": ("string", "glob"), "path": ("string", "folder to search, default .")}, ["pattern"]),
    ("grep", "Search file contents with a regex (case-insensitive). Returns path:line: text.",
     {"pattern": ("string", "regex"), "glob": ("string", "file name filter, e.g. *.md"),
      "path": ("string", "folder, default .")}, ["pattern"]),
    ("read_file", "Read a text file with line numbers, 250 lines at a time.",
     {"path": ("string", "file"), "start": ("integer", "first line, default 1")}, ["path"]),
    ("edit_file", "Replace an exact snippet in a file. Copy `old` exactly from read_file output, without the 'NN| ' prefix.",
     {"path": ("string", "file"), "old": ("string", "exact existing text"), "new": ("string", "replacement text"),
      "replace_all": ("boolean", "replace every occurrence")}, ["path", "old", "new"]),
    ("write_file", "Create a new file or fully overwrite one.",
     {"path": ("string", "file"), "content": ("string", "full content")}, ["path", "content"]),
    ("run", f"Run a {SHELL} command in the workspace (tests, git, builds). Returns exit code and output.",
     {"command": ("string", "command line"), "timeout": ("integer", "seconds, default 120")}, ["command"]),
    ("fetch_url", "Fetch a web page or API as text (docs, issues).", {"url": ("string", "http(s) URL")}, ["url"]),
]
TOOL_SCHEMA = [{"type": "function", "function": {
    "name": n, "description": d,
    "parameters": {"type": "object", "required": req,
                   "properties": {k: {"type": t, "description": v} for k, (t, v) in p.items()}}}}
    for n, d, p, req in SPECS]


# ---------------------------------------------------------------- prompt
def project_instructions(limit=int(os.environ.get("TINYAGENT_INSTR_CHARS", "2000"))):
    for name in ("TINYAGENT.md", "AGENTS.md", "CLAUDE.md"):
        f = ROOT / name
        if f.is_file():
            t = f.read_text(encoding="utf-8", errors="ignore").strip()
            cut = f"\n(...truncated; read_file {name} for the rest)" if len(t) > limit else ""
            return f"\n\nProject instructions from {name}:\n{t[:limit]}{cut}"
    return ""


def system_prompt():
    return f"""You are tinyagent, a coding agent in the folder {ROOT} ({'Windows' if WIN else sys.platform}; `run` uses {SHELL}).
- Explore before answering: find_files/grep to locate, then read_file. Never guess file contents.
- Change code with edit_file, copying `old` exactly from read_file output. Use write_file only for new files or rewrites.
- After changing code, verify it (run the tests or re-read the file) when you can.
- Prefer few, targeted tool calls. Paths are relative to the folder.
- Finish with a short answer: what you found or changed, and the files involved.{project_instructions()}"""


# ---------------------------------------------------------------- model backends
def to_ollama(msgs):
    """Internal messages use the OpenAI shape; Ollama's native API wants dict arguments and tool_name."""
    out = []
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": [
                {"function": {"name": tc["function"]["name"], "arguments": json.loads(tc["function"]["arguments"] or "{}")}}
                for tc in m["tool_calls"]]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "content": m["content"], "tool_name": m.get("name", "")})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


def http_lines(url, body, headers):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers})
    try:
        return urllib.request.urlopen(req, timeout=1800)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from {url}: {e.read().decode(errors='replace')[:500]}") from None
    except urllib.error.URLError as e:
        hint = " Is Ollama running? (ollama serve)" if S.provider == "ollama" else ""
        raise RuntimeError(f"cannot reach {url}: {e.reason}.{hint}") from None


def ollama_events(lines):
    for line in lines:
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("error"):
            raise RuntimeError(d["error"])
        m = d.get("message", {})
        if m.get("content"):
            yield "text", m["content"]
        for tc in m.get("tool_calls") or []:
            yield "call", (tc["function"]["name"], tc["function"].get("arguments") or {})
        if d.get("done"):
            yield "stats", {"in": d.get("prompt_eval_count", 0), "out": d.get("eval_count", 0),
                            "gen_s": d.get("eval_duration", 0) / 1e9}


def openai_events(lines):
    """Parse an OpenAI-compatible SSE stream; tool-call arguments arrive in fragments keyed by index."""
    slots = {}
    for raw in lines:
        line = (raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw).strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        d = json.loads(data)
        if d.get("error"):
            raise RuntimeError(str(d["error"]))
        if d.get("usage"):
            u = d["usage"]
            yield "stats", {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0), "gen_s": 0}
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            if delta.get("content"):
                yield "text", delta["content"]
            for tc in delta.get("tool_calls") or []:
                s = slots.setdefault(tc.get("index", 0), {"name": "", "args": ""})
                fn = tc.get("function") or {}
                s["name"] += fn.get("name") or ""
                s["args"] += fn.get("arguments") or ""
    for s in slots.values():
        try:
            args = json.loads(s["args"] or "{}")
        except json.JSONDecodeError:
            args = {"_raw": s["args"]}
        yield "call", (s["name"], args)


def stream(msgs):
    base, _, key_env = PROVIDERS[S.provider]
    if S.provider == "ollama":
        base = base if base.startswith("http") else "http://" + base
        body = {"model": S.model, "messages": to_ollama(msgs), "tools": TOOL_SCHEMA, "stream": True,
                "options": {"num_ctx": S.ctx, "temperature": 0.2}}
        return ollama_events(http_lines(f"{base}/api/chat", body, {}))
    key = os.environ.get(key_env or "", "")
    if not key:
        raise RuntimeError(f"set {key_env} to use --provider {S.provider}")
    body = {"model": S.model, "messages": msgs, "tools": TOOL_SCHEMA, "stream": True, "temperature": 0.2,
            "stream_options": {"include_usage": True}}
    return openai_events(http_lines(f"{base}/chat/completions", body,
                                    {"Authorization": f"Bearer {key}", "X-Title": "tinyagent"}))


TEXT_CALL = re.compile(r"<function=(\w+)>(.*?)</function>", re.S)
PARAM = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.S)


def text_tool_calls(text):
    """qwen3-coder sometimes writes a tool call as raw XML text that the server doesn't parse. Recover it."""
    return [{"function": {"name": n, "arguments": {k: (int(v) if k in ("start", "timeout") and v.isdigit() else v)
                                                   for k, v in PARAM.findall(body)}}}
            for n, body in TEXT_CALL.findall(text) if n in TOOLS]


# ---------------------------------------------------------------- context management
def est_tokens(msgs):
    return sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or "")) for m in msgs) // 3


def elide_old_outputs(keep_last=6):
    """Cheapest compaction: replace old tool outputs with a stub. The model can re-run a tool if it needs it."""
    freed = 0
    for m in S.messages[1:-keep_last]:
        if m["role"] == "tool" and not m["content"].startswith("[old output"):
            freed += len(m["content"])
            m["content"] = f"[old output removed to save context, {len(m['content'])} chars; re-run the tool if needed]"
    return freed


def summarize():
    """Ask the model for a summary of the work so far, then restart the history from it."""
    info("  compacting: summarizing the conversation...")
    req = S.messages + [{"role": "user", "content":
                         "Summarize this session for yourself in under 200 words: the goal, what was done, files "
                         "changed, key facts found, and what is left. No tool calls."}]
    text = "".join(v for k, v in stream(req) if k == "text")
    S.messages = [S.messages[0], {"role": "user", "content": f"[Summary of the earlier conversation]\n{text}"},
                  {"role": "assistant", "content": "Understood. Continuing from the summary."}]
    S.last_in = 0


def maybe_compact():
    used = max(S.last_in, est_tokens(S.messages))
    if used < 0.7 * S.ctx:
        return
    freed = elide_old_outputs()
    if freed:
        info(f"  context {used}/{S.ctx} tokens: removed {freed:,} chars of old tool output")
    if est_tokens(S.messages) > 0.85 * S.ctx:
        summarize()


# ---------------------------------------------------------------- agent loop
def ticker(stop, label, t0):
    while not stop.wait(0.5):
        if TTY:
            print(f"\r  {label} thinking... {time.time() - t0:4.0f}s", end="", file=sys.stderr, flush=True)


def step(n):
    """One model call. Streams text, collects tool calls. Returns the assistant message and its calls."""
    t0, stop = time.time(), threading.Event()
    label = f"[{S.model} step {n}]"
    threading.Thread(target=ticker, args=(stop, label, t0), daemon=True).start()
    text, calls, stats, first = [], [], {}, None

    def clear_ticker():
        stop.set()
        if TTY:
            print("\r" + " " * 60 + "\r", end="", file=sys.stderr, flush=True)

    try:
        for kind, val in stream(S.messages):
            if first is None and kind in ("text", "call"):
                first = time.time()
                clear_ticker()
            if kind == "text":
                text.append(val)
                so_far = "".join(text)
                if "<function" not in so_far and "<tool_call" not in so_far:
                    print(val, end="", flush=True)
            elif kind == "call":
                calls.append(val)
            elif kind == "stats":
                stats = val
    finally:
        clear_ticker()
    full = "".join(text)
    if text:
        print()
    if not calls:
        recovered = text_tool_calls(full)
        if recovered:
            calls = [(r["function"]["name"], r["function"]["arguments"]) for r in recovered]
            full = full[:full.find("<")].strip()
    S.last_in = stats.get("in", 0) or S.last_in
    gen = f", {stats['out'] / stats['gen_s']:.1f} tok/s" if stats.get("gen_s") else ""
    wait = f"first token {first - t0:.0f}s, " if first else ""
    info(f"  {label} {wait}total {time.time() - t0:.0f}s | {stats.get('in', '?')} tok in, {stats.get('out', '?')} out{gen}")
    msg = {"role": "assistant", "content": full}
    if calls:
        msg["tool_calls"] = [{"id": f"call_{S.calls_made + i}", "type": "function",
                              "function": {"name": nm, "arguments": json.dumps(a)}} for i, (nm, a) in enumerate(calls)]
        S.calls_made += len(calls)
    return msg, calls


def execute(name, args):
    if name not in TOOLS:
        return f"error: unknown tool {name}. Available: {', '.join(TOOLS)}"
    if "_raw" in args:
        return f"error: could not parse arguments as JSON: {args['_raw'][:200]}"
    try:
        out = str(TOOLS[name](**args))
    except TypeError as e:
        out = f"error: bad arguments for {name}: {e}"
    except Exception as e:  # missing file, bad regex, network... tell the model so it can recover
        out = f"error: {type(e).__name__}: {e}"
    if len(out) > MAX_OUT:
        out = out[:MAX_OUT] + f"\n... (truncated, {len(out):,} chars total; read a smaller range or narrow the search)"
    return out


def turn(max_steps=int(os.environ.get("TINYAGENT_MAX_STEPS", "25"))):
    recent = []
    for n in range(1, max_steps + 1):
        maybe_compact()
        msg, calls = step(n)
        S.messages.append(msg)
        if not calls:
            return
        for tc, (name, args) in zip(msg["tool_calls"], calls):
            sig = (name, json.dumps(args, sort_keys=True))
            info(f"  -> {name}({json.dumps(args)[:150]})")
            if recent.count(sig) >= 2:
                out = "error: you already ran this exact call twice. Use the earlier result or try something different."
            else:
                out = execute(name, args)
            recent.append(sig)
            info(f"     {out.splitlines()[0][:100] if out else '(empty)'}  [{len(out):,} chars]")
            S.messages.append({"role": "tool", "tool_call_id": tc["id"], "name": name, "content": out})
    info(f"  stopped after {max_steps} steps. Say 'continue' to keep going.")


# ---------------------------------------------------------------- sessions
def session_dir():
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(ROOT))[-60:].strip("-")
    d = HOME / "sessions" / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_session():
    if S.session is None:
        S.session = session_dir() / time.strftime("%Y%m%d-%H%M%S.json")
    S.session.write_text(json.dumps({"provider": S.provider, "model": S.model, "messages": S.messages[1:]},
                                    indent=1), encoding="utf-8")


def load_last_session():
    files = sorted(session_dir().glob("*.json"), key=lambda f: f.stat().st_mtime)
    if not files:
        return False
    data = json.loads(files[-1].read_text(encoding="utf-8"))
    S.messages = [{"role": "system", "content": system_prompt()}] + data["messages"]
    S.session = files[-1]
    S.calls_made = sum(len(m.get("tool_calls") or []) for m in data["messages"])
    return True


# ---------------------------------------------------------------- user input
HELP = """commands:
  /help                 this help
  /status               provider, model, mode, context use, undo stack
  /mode ask|auto|yolo|readonly
                        ask = approve every change; auto = edits auto-approved, commands still asked;
                        yolo = approve everything; readonly = no edits or commands
  /model NAME           switch model          /provider ollama|openrouter|openai [MODEL]
  /ctx N                context window (local models: bigger = more memory)
  /undo                 revert the last file change made by the agent
  /compact              summarize the conversation to free context
  /clear                start a fresh conversation
  /quit                 exit (Ctrl+C at the prompt also exits; Ctrl+C during a turn stops the turn)
shortcuts:
  @path/to/file         attach a file's contents to your message (saves a tool round-trip)
  !command              run a shell command yourself; output is added to the conversation
  line ending in \\      continue typing on the next line"""


def attach_files(q):
    extra = []
    for tok in re.findall(r"@([\w./\\-]+)", q):
        try:
            p = safe(tok)
            if p.is_file():
                text, _ = read_text(p)
                extra.append(f"\n\n[Attached {rel(p)}]\n{text[:MAX_OUT]}" + ("\n...(truncated)" if len(text) > MAX_OUT else ""))
        except (ValueError, OSError):
            pass
    return q + "".join(extra)


def do_undo():
    if not S.undo:
        return "nothing to undo"
    p, old, eol = S.undo.pop()
    if old is None:
        p.unlink(missing_ok=True)
        return f"removed {rel(p)} (it was created by the agent)"
    write_text(p, old, eol)
    return f"restored {rel(p)}"


def command(q):
    """Handle a /command. Returns True if the REPL should exit."""
    cmd, _, arg = q[1:].partition(" ")
    arg = arg.strip()
    if cmd in ("quit", "exit", "q"):
        return True
    if cmd == "help":
        print(HELP)
    elif cmd == "status":
        print(f"provider {S.provider} | model {S.model} | mode {S.mode} | ctx {S.ctx} | "
              f"~{max(S.last_in, est_tokens(S.messages))} tokens used | {len(S.undo)} undoable change(s) | "
              f"always-allowed: {', '.join(sorted(S.allow)) or 'none'}\nsession: {S.session or '(not saved yet)'}")
    elif cmd == "mode" and arg in MODES:
        S.mode = arg
        print(f"mode = {arg}")
    elif cmd == "model" and arg:
        S.model = arg
        print(f"model = {arg}")
    elif cmd == "provider" and arg.split()[0] in PROVIDERS:
        parts = arg.split()
        S.provider = parts[0]
        S.model = parts[1] if len(parts) > 1 else PROVIDERS[S.provider][1]
        print(f"provider = {S.provider}, model = {S.model}")
    elif cmd == "ctx" and arg.isdigit():
        S.ctx = int(arg)
        print(f"ctx = {S.ctx}")
    elif cmd == "undo":
        print(do_undo())
    elif cmd == "compact":
        elide_old_outputs(keep_last=2)
        summarize()
        print("compacted")
    elif cmd == "clear":
        S.messages, S.session, S.last_in = [{"role": "system", "content": system_prompt()}], None, 0
        print("new conversation")
    else:
        print(f"unknown or incomplete command: {q}  (try /help)")
    return False


def read_input():
    lines = []
    while True:
        line = input(c("1;36", "\n> ") if not lines else c("36", ". "))
        if line.endswith("\\"):
            lines.append(line[:-1])
            continue
        lines.append(line)
        return "\n".join(lines).strip()


def ask(q):
    """Run one user message through the agent, rolling back cleanly on Ctrl+C or errors."""
    snapshot = len(S.messages)
    S.messages.append({"role": "user", "content": attach_files(q)})
    try:
        turn()
    except KeyboardInterrupt:
        del S.messages[snapshot:]
        print(c("33", "\n  [interrupted: turn discarded; file changes stay on disk, /undo reverts them]"))
    except RuntimeError as e:
        del S.messages[snapshot:]
        print(c("31", f"\n  error: {e}"))
    save_session()


def main():
    ap = argparse.ArgumentParser(description="tinyagent: small-prompt coding agent for local models")
    ap.add_argument("-p", "--prompt", help="run one prompt and exit")
    ap.add_argument("-c", "--resume", action="store_true", help="continue the last session in this folder")
    ap.add_argument("--provider", choices=list(PROVIDERS), default=os.environ.get("TINYAGENT_PROVIDER", "ollama"))
    ap.add_argument("--model")
    ap.add_argument("--ctx", type=int)
    ap.add_argument("--mode", choices=MODES)
    a = ap.parse_args()
    for s in (sys.stdout, sys.stderr):  # Windows pipes default to cp1252 and crash on ✓, emoji, etc.
        s.reconfigure(encoding="utf-8", errors="replace")
    S.provider = a.provider
    S.model = a.model or PROVIDERS[S.provider][1]
    S.ctx = a.ctx or S.ctx
    S.mode = a.mode or (S.mode if S.mode in MODES else "ask")
    S.messages = [{"role": "system", "content": system_prompt()}]
    if a.resume and load_last_session():
        info(f"resumed {S.session.name} ({len(S.messages) - 1} messages)")
    if a.prompt:
        ask(a.prompt)
        return
    print(f"tinyagent {VERSION} | {S.provider}:{S.model} | mode {S.mode} | {ROOT}\n"
          f"type /help for commands, @file to attach, !cmd to run a command")
    while True:
        try:
            q = read_input()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            continue
        if q.startswith("/"):
            if command(q):
                return
        elif q.startswith("!"):
            out = execute_user_shell(q[1:].strip())
            print(out)
            S.messages.append({"role": "user", "content": f"I ran `{q[1:].strip()}`:\n{out[:MAX_OUT]}"})
            S.messages.append({"role": "assistant", "content": "Noted."})
        else:
            ask(q)


def execute_user_shell(cmd):
    argv = ["powershell", "-NoProfile", "-Command", cmd] if WIN else ["bash", "-lc", cmd]
    r = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return f"exit {r.returncode}\n{r.stdout}{r.stderr}".strip()


if __name__ == "__main__":
    main()
