"""Run: python test_tinyagent.py  (no Ollama needed)"""
import os, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

d = Path(tempfile.mkdtemp())
(d / "notes").mkdir()
(d / "notes" / "release-plan.md").write_text("phase 2 shipped\n", encoding="utf-8")
os.chdir(d)
import tinyagent as t  # ROOT is taken from cwd at import

for bad in ("../x", "..\\x", "C:\\Windows\\win.ini", "/etc/passwd", "notes/../../x"):
    try:
        t.safe(bad)
        raise SystemExit(f"FAIL: escaped workspace with {bad!r}")
    except ValueError:
        pass
assert t.safe("notes/release-plan.md").exists()
assert "notes" in t.find_files("*release*")
assert "phase 2" in t.grep("shipped", "*.md")
assert t.read_file("notes/release-plan.md").startswith("phase 2")
raw = "Let me search.\n\n<function=find_files>\n<parameter=pattern>\n*release*\n</parameter>\n</function>\n</tool_call>"
calls = t.text_tool_calls(raw)
assert calls == [{"function": {"name": "find_files", "arguments": {"pattern": "*release*"}}}], calls
assert t.text_tool_calls("plain answer, no tools") == []
print("ALL TESTS PASS")
