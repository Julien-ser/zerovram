"""delegate.py: hand bulk reading to a local model and get back only a short result.

Built for another agent (or you) to save tokens: the local model reads the long files, and only the summary
or answer comes back. Results are cached by file content, so a re-run returns instantly.

    python delegate.py summarize FILE [FILE ...] [--focus "what matters"] [--bullets 8]
    python delegate.py find "QUESTION" FOLDER [--glob "*.md"] [--max-files 40]
    python delegate.py ask "QUESTION" FILE [FILE ...]

Exit code 2 means the local model is unreachable (the caller should fall back to reading the files itself).
Env: DELEGATE_MODEL (default qwen3-coder:30b), DELEGATE_CTX (default 16384), OLLAMA_HOST. Stdlib only.
"""
import argparse, fnmatch, hashlib, json, os, sys, time, urllib.error, urllib.request
from pathlib import Path

MODEL = os.environ.get("DELEGATE_MODEL", "qwen3-coder:30b")
CTX = int(os.environ.get("DELEGATE_CTX", "16384"))
HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
HOST = HOST if HOST.startswith("http") else "http://" + HOST
CHUNK_CHARS = int(CTX * 0.55 * 3.2)  # leave room for the instructions and the answer
CACHE = Path(os.environ.get("DELEGATE_CACHE", str(Path.home() / ".zerovram" / "cache")))
SKIP = {".git", "node_modules", ".venv", "venv", "site-packages", "__pycache__", ".obsidian", ".trash"}


class Unreachable(Exception):
    pass


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def generate(prompt, max_tokens=600):
    """One non-streaming call to the local model. Returns the text."""
    body = {"model": MODEL, "prompt": prompt, "stream": False, "think": False,
            "options": {"num_ctx": CTX, "temperature": 0.1, "num_predict": max_tokens}}
    req = urllib.request.Request(f"{HOST}/api/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=1800))
    except urllib.error.URLError as e:
        raise Unreachable(f"local model unreachable at {HOST}: {getattr(e, 'reason', e)}") from None
    return r.get("response", "").strip()


def cached(kind, key_parts, fn):
    """Cache results by (kind, model, content). Same input = same answer, no model call."""
    h = hashlib.sha256("\x00".join([kind, MODEL, *key_parts]).encode("utf-8", "replace")).hexdigest()[:32]
    f = CACHE / f"{h}.txt"
    if f.exists():
        return f.read_text(encoding="utf-8"), True
    out = fn()
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(out, encoding="utf-8")
    return out, False


def read(path):
    return Path(path).read_text(encoding="utf-8", errors="replace")


def chunks(text, size=None):
    """Split on line boundaries into pieces that fit the local context window."""
    size = size or CHUNK_CHARS
    out, cur = [], []
    n = 0
    for line in text.splitlines(keepends=True):
        while len(line) > size:  # a single giant line (minified file, CSV row...)
            out.append("".join(cur) + line[:size]) if cur else out.append(line[:size])
            cur, n, line = [], 0, line[size:]
        if n + len(line) > size and cur:
            out.append("".join(cur))
            cur, n = [], 0
        cur.append(line)
        n += len(line)
    if cur:
        out.append("".join(cur))
    return out or [""]


def summarize_text(text, name, bullets=8, focus=""):
    focus_line = f" Focus on: {focus}." if focus else ""
    rule = (f"Summarize the document below in at most {bullets} bullet points for an engineer.{focus_line} "
            "Keep exact names, numbers, commands, paths and versions. No preamble.")
    parts = chunks(text)
    if len(parts) == 1:
        return generate(f"{rule}\n\n--- {name} ---\n{text}")
    partials = []
    for i, part in enumerate(parts, 1):  # map: summarize each piece, then reduce
        log(f"  {name}: part {i}/{len(parts)}")
        partials.append(generate(f"{rule}\n\n--- {name}, part {i} of {len(parts)} ---\n{part}"))
    return generate(f"Merge these partial summaries of {name} into at most {bullets} bullet points. "
                    f"Keep exact names, numbers and commands. No preamble.{focus_line}\n\n" + "\n\n".join(partials))


def cmd_summarize(files, bullets, focus):
    for f in files:
        t0 = time.time()
        text = read(f)
        out, hit = cached("summarize", [str(bullets), focus, text], lambda: summarize_text(text, Path(f).name, bullets, focus))
        log(f"  {f}: {len(text):,} chars -> {len(out):,} ({'cache' if hit else f'{time.time() - t0:.0f}s'})")
        print(f"## {f}\n{out}\n")


def iter_files(folder, glob):
    for dirpath, dirs, names in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d not in SKIP)
        for n in sorted(names):
            if fnmatch.fnmatch(n.lower(), glob.lower()):
                yield Path(dirpath, n)


def parse_verdict(reply):
    """Expect 'YES: reason' or 'NO'. Be lenient with formatting from small models."""
    head = reply.strip().lstrip("*#-> ").upper()
    if head.startswith("YES"):
        reason = reply.strip().split(":", 1)[1].strip() if ":" in reply else ""
        return True, reason.splitlines()[0][:200] if reason else ""
    return False, ""


OUTLINE = __import__("re").compile(r"^\s*(#{1,6} .+|(?:async )?def \w+.*|class \w+.*|function \w+.*)$", __import__("re").M)


def preview(text, head_chars=1500, outline_chars=2500):
    """Opening text plus the whole file's outline (headings, defs, classes). Topics buried deep in a long file
    still show up in the outline, which the opening alone misses."""
    outline = "\n".join(m.strip() for m in OUTLINE.findall(text[head_chars:]))
    return text[:head_chars] + (f"\n\n[Outline of the rest of the file]\n{outline[:outline_chars]}" if outline else "")


def keywords(question):
    """One short model call: turn the question into search terms. Stems catch word variants (cach -> caching)."""
    reply, _ = cached("kw", [question], lambda: generate(
        "Give 3 to 6 short search terms for finding text that answers the question below. Use lowercase word stems "
        "where helpful (e.g. 'cach' for cache/caching). Reply with a comma-separated list only.\n"
        f"Question: {question}", max_tokens=40))
    kws = [k.strip().strip("\"'`.").lower() for k in reply.replace("\n", ",").split(",")]
    return [k for k in kws if 2 < len(k) < 40][:6]


def snippets(text, kws, limit=2500, after=4, after_heading=15):
    """Matching lines plus context. A matching heading brings its section body: the heading alone says too
    little for the model to judge relevance."""
    lines = text.splitlines()
    keep = set()
    for i, ln in enumerate(lines):
        if any(k in ln.lower() for k in kws):
            n = after_heading if ln.lstrip().startswith("#") else after
            keep.update(range(max(0, i - 1), min(len(lines), i + n + 1)))
    keep = sorted(j for j in keep if lines[j].strip() or j - 1 not in keep)  # drop runs of blank lines
    out, last = [], -2
    for j in keep:
        if j != last + 1:
            out.append("...")
        out.append(f"{j + 1}: {lines[j]}")
        last = j
    return "\n".join(out)[:limit]


def cmd_find(question, folder, glob, max_files):
    files = list(iter_files(folder, glob))
    kws = keywords(question)
    log(f"  keywords: {', '.join(kws) or '(none)'}")
    scored = []
    for f in files:  # free pass: plain text search over every whole file, no model involved
        low = read(f).lower()
        n = sum(low.count(k) for k in kws)
        if n:
            scored.append((n, f))
    if scored:
        scored.sort(key=lambda x: -x[0])
        log(f"  {len(scored)} of {len(files)} files contain a keyword; the model judges the top {min(max_files, len(scored))}")
        targets = [(f, "matching passages", snippets(read(f), kws)) for _, f in scored[:max_files]]
    else:  # nothing literal matched: let the model look at openings + outlines instead
        log(f"  no keyword hits; checking openings + outlines of {min(max_files, len(files))} files")
        files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:max_files]
        targets = [(f, "opening + outline", preview(read(f))) for f in files]
    hits = 0
    for i, (f, kind, head) in enumerate(targets, 1):
        prompt = (f"Question: {question}\n\nDoes this file contain information that helps answer the question? "
                  "Say YES only if it addresses the question itself, not merely a related topic. "
                  "Reply with exactly 'YES: <one line on what it contains>' or 'NO'.\n\n"
                  f"--- {f.name} ({kind}) ---\n{head}")
        reply, hit = cached("find", [question, head], lambda: generate(prompt, max_tokens=60))
        ok, reason = parse_verdict(reply)
        log(f"  [{i}/{len(targets)}] {'YES' if ok else 'no '} {f}{' (cache)' if hit else ''}")
        if ok:
            hits += 1
            print(f"- {f}: {reason}")
    if not hits:
        print("(no relevant files found)")


def cmd_ask(question, files):
    blob = "".join(f"\n--- {f} ---\n{read(f)}" for f in files)
    parts = chunks(blob)
    rule = (f"Answer the question using only the text below. Quote exact names and numbers. "
            f"If the text does not contain the answer, say 'NOT FOUND'.\nQuestion: {question}\n")
    if len(parts) == 1:
        out, _ = cached("ask", [question, blob], lambda: generate(rule + blob))
    else:
        notes = []
        for i, p in enumerate(parts, 1):
            log(f"  part {i}/{len(parts)}")
            r, _ = cached("ask", [question, p], lambda: generate(rule + p, max_tokens=300))
            if "NOT FOUND" not in r.upper():
                notes.append(r)
        out = generate(f"Combine these partial answers into one answer to: {question}\n\n" + "\n\n".join(notes)) \
            if notes else "NOT FOUND"
    print(out)


def main():
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize"); s.add_argument("files", nargs="+")
    s.add_argument("--bullets", type=int, default=8); s.add_argument("--focus", default="")
    f = sub.add_parser("find"); f.add_argument("question"); f.add_argument("folder")
    f.add_argument("--glob", default="*.md"); f.add_argument("--max-files", type=int, default=40)
    a = sub.add_parser("ask"); a.add_argument("question"); a.add_argument("files", nargs="+")
    args = ap.parse_args()
    t0 = time.time()
    try:
        if args.cmd == "summarize":
            cmd_summarize(args.files, args.bullets, args.focus)
        elif args.cmd == "find":
            cmd_find(args.question, args.folder, args.glob, args.max_files)
        else:
            cmd_ask(args.question, args.files)
    except Unreachable as e:
        log(f"error: {e}")
        sys.exit(2)
    log(f"done in {time.time() - t0:.0f}s ({MODEL})")


if __name__ == "__main__":
    main()
