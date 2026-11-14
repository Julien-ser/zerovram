"""Run: python test_delegate.py  (no model needed; the model call is faked)"""
import io, contextlib, os, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
tmp = Path(tempfile.mkdtemp())
os.environ["DELEGATE_CACHE"] = str(tmp / "cache")
import delegate as d

calls = []
def fake(prompt, max_tokens=600):
    calls.append(prompt)
    if "Does this file contain" in prompt:
        return "YES: mentions retries" if "retry" in prompt.split("---", 1)[1] else "NO"
    return f"summary#{len(calls)}"
d.generate = fake

# chunking: respects size, keeps every character, handles a giant single line
text = "".join(f"line {i}\n" for i in range(2000))
parts = d.chunks(text, size=1000)
assert all(len(p) <= 1000 for p in parts) and "".join(parts) == text
giant = d.chunks("x" * 2500, size=1000)
assert [len(p) for p in giant] == [1000, 1000, 500]

# summarize: one call for a small file, map-reduce for a big one, cache on re-run
(tmp / "small.md").write_text("short doc", encoding="utf-8")
(tmp / "big.md").write_text("word " * 30000, encoding="utf-8")
out = io.StringIO()
with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
    d.cmd_summarize([str(tmp / "small.md")], 5, "")
assert len(calls) == 1 and "summary#1" in out.getvalue()
n_parts = len(d.chunks("word " * 30000))
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    d.cmd_summarize([str(tmp / "big.md")], 5, "")
assert len(calls) == 1 + n_parts + 1, (len(calls), n_parts)  # each part + one merge
before = len(calls)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    d.cmd_summarize([str(tmp / "big.md")], 5, "")
assert len(calls) == before, "cache miss on identical input"

# find: lenient YES/NO parsing and only relevant files reported
assert d.parse_verdict("**YES**: has it") == (True, "has it")
assert d.parse_verdict("No.") == (False, "")
notes = tmp / "notes"; notes.mkdir()
(notes / "a.md").write_text("we added retry with backoff", encoding="utf-8")
(notes / "b.md").write_text("lunch menu", encoding="utf-8")
out = io.StringIO()
with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
    d.cmd_find("how do retries work?", str(notes), "*.md", 40)
assert "a.md: mentions retries" in out.getvalue() and "b.md" not in out.getvalue()

# unreachable model -> Unreachable (main() turns it into exit code 2)
import importlib
importlib.reload(d)  # restore the real generate()
d.HOST = "http://127.0.0.1:9"  # nothing listens here
try:
    d.generate("x")
    raise SystemExit("FAIL: expected Unreachable")
except d.Unreachable:
    pass
print("ALL TESTS PASS")
