"""Run: python test_tinyagent.py  (no model or network needed)"""
import json, os, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
d = Path(tempfile.mkdtemp())
os.environ["TINYAGENT_HOME"] = str(d / ".home")
(d / "notes").mkdir()
(d / "notes" / "release-plan.md").write_text("phase 2 shipped\n", encoding="utf-8")
os.chdir(d)
import tinyagent as t  # ROOT comes from cwd at import

t.S.mode = "yolo"  # no prompts during tests
t.TTY = False

# --- sandbox
for bad in ("../x", "..\\x", "C:\\Windows\\win.ini", "/etc/passwd", "notes/../../x"):
    try:
        t.safe(bad)
        raise SystemExit(f"FAIL: escaped workspace with {bad!r}")
    except ValueError:
        pass

# --- read tools
assert "notes" in t.find_files("*release*")
assert "phase 2" in t.grep("shipped", "*.md")
assert "    1| phase 2 shipped" in t.read_file("notes/release-plan.md")
assert "notes/" in t.list_dir()

# --- edit_file: exact, not found, ambiguous, line-number prefixes, CRLF preserved
src = d / "app.py"
src.write_bytes(b"def f():\r\n    return 1\r\n\r\ndef g():\r\n    return 1\r\n")
assert "appears 2 times" in t.edit_file("app.py", "return 1", "return 2")
assert "not found" in t.edit_file("app.py", "return 3", "x")
assert t.edit_file("app.py", "def f():\n    return 1", "def f():\n    return 42").startswith("edited")
assert t.edit_file("app.py", "    4| def g():\n    5|     return 1", "    4| def g():\n    5|     return 7").startswith("edited")
assert src.read_bytes() == b"def f():\r\n    return 42\r\n\r\ndef g():\r\n    return 7\r\n", src.read_bytes()

# --- undo (two edits, then the file the agent created)
t.write_file("new.txt", "hello")
assert (d / "new.txt").exists()
assert "removed" in t.do_undo() and not (d / "new.txt").exists()
t.do_undo(); t.do_undo()
assert src.read_bytes() == b"def f():\r\n    return 1\r\n\r\ndef g():\r\n    return 1\r\n"
assert t.do_undo() == "nothing to undo"

# --- permissions
t.S.mode = "readonly"
assert "read-only" in t.edit_file("app.py", "def f", "def h")
assert "read-only" in t.run("echo hi")
t.S.mode = "yolo"
assert "exit 0" in t.run("echo hi") and "hi" in t.run("echo hi")

# --- recovering tool calls written as text
raw = "Let me search.\n\n<function=find_files>\n<parameter=pattern>\n*release*\n</parameter>\n</function>\n</tool_call>"
assert t.text_tool_calls(raw) == [{"function": {"name": "find_files", "arguments": {"pattern": "*release*"}}}]
assert t.text_tool_calls("plain answer, no tools") == []

# --- OpenAI-compatible SSE parsing (fragmented tool-call arguments)
sse = ['data: {"choices":[{"delta":{"content":"Hi "}}]}',
       'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"read_file","arguments":"{\\"pa"}}]}}]}',
       'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\\": \\"a.py\\"}"}}]}}]}',
       'data: {"choices":[],"usage":{"prompt_tokens":50,"completion_tokens":7}}',
       "data: [DONE]"]
ev = list(t.openai_events(sse))
assert ("text", "Hi ") in ev and ("call", ("read_file", {"path": "a.py"})) in ev
assert ("stats", {"in": 50, "out": 7, "gen_s": 0}) in ev

# --- Ollama stream parsing + message conversion
nd = [json.dumps({"message": {"content": "ok"}}),
      json.dumps({"message": {"tool_calls": [{"function": {"name": "grep", "arguments": {"pattern": "x"}}}]}}),
      json.dumps({"done": True, "prompt_eval_count": 9, "eval_count": 3, "eval_duration": 1e9})]
ev = list(t.ollama_events(nd))
assert ev[0] == ("text", "ok") and ev[1] == ("call", ("grep", {"pattern": "x"})) and ev[2][1]["in"] == 9
conv = t.to_ollama([{"role": "assistant", "content": "", "tool_calls": [
    {"id": "call_0", "type": "function", "function": {"name": "grep", "arguments": '{"pattern": "x"}'}}]},
    {"role": "tool", "tool_call_id": "call_0", "name": "grep", "content": "hit"}])
assert conv[0]["tool_calls"][0]["function"]["arguments"] == {"pattern": "x"} and conv[1]["tool_name"] == "grep"

# --- compaction drops old tool output but keeps recent messages
t.S.messages = [{"role": "system", "content": "s"}] + \
               [{"role": "tool", "content": "x" * 5000, "name": "grep", "tool_call_id": str(i)} for i in range(10)]
freed = t.elide_old_outputs(keep_last=3)
assert freed == 7 * 5000 and t.S.messages[-1]["content"] == "x" * 5000
assert t.S.messages[1]["content"].startswith("[old output")

# --- @file attachments and the sandbox for them
q = t.attach_files("explain @notes/release-plan.md and @../secret.txt")
assert "[Attached notes/release-plan.md]" in q and "phase 2 shipped" in q and "secret" not in q.split("]", 1)[1]

# --- sessions save and resume
t.S.messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "remember 42"}]
t.S.session = None
t.save_session()
t.S.messages = []
assert t.load_last_session() and t.S.messages[1]["content"] == "remember 42"

# --- execute() turns exceptions into messages the model can act on
assert t.execute("nope", {}).startswith("error: unknown tool")
assert "bad arguments" in t.execute("read_file", {"wrong": 1})
assert "FileNotFoundError" in t.execute("read_file", {"path": "missing.txt"})
assert "could not parse" in t.execute("grep", {"_raw": "{bad"})

print("ALL TESTS PASS")
