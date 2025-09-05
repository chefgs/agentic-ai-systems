Awesome—here’s a **ready-to-run Autogen** starter that does exactly what you asked:

* 3 role agents: **Product Manager → Developer → QA**
* Conversational, **multi-agent** collaboration
* **Tools**: safe file writer + safe shell runner (optional), both gated by **human approval**
* **Human-in-the-loop** checkpoints (you approve before actions)
* **Memory/transcripts** saved to `transcripts/<timestamp>.md`
* Works with **OpenAI** or **Ollama** (OpenAI-compatible) — configurable via env vars

---

## 0) Install & env

```bash
pip install pyautogen openai
# Optional (if you’ll use ollama instead of OpenAI)
# brew install ollama  &&  ollama serve
# ollama pull llama3:8b
```

Set env (choose one provider):

```bash
# Option A: OpenAI
export MODEL_PROVIDER=openai
export OPENAI_API_KEY=sk-...

# Option B: Ollama (local)
export MODEL_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434/v1
export OLLAMA_MODEL=llama3:8b
```

---

## 1) The code (save as `agentic_mvp.py`)

````python
import os
import re
import time
import pathlib
from typing import List, Dict

from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat.groupchat import GroupChat, GroupChatManager


# ==============
# CONFIG HELPERS
# ==============
def build_llm_config():
    """
    Build a provider-agnostic llm_config for Autogen.
    Supports OpenAI (cloud) or Ollama (local, OpenAI-compatible).
    """
    provider = os.getenv("MODEL_PROVIDER", "openai").lower()

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        model = os.getenv("OLLAMA_MODEL", "llama3:8b")
        # OpenAI-compatible client descriptor
        return {
            "config_list": [
                {
                    "model": model,
                    "api_key": os.getenv("OLLAMA_API_KEY", "ollama"),  # dummy
                    "base_url": base_url,
                }
            ],
            "temperature": 0.2,
        }

    # default: OpenAI
    return {
        "config_list": [
            {
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "api_key": os.getenv("OPENAI_API_KEY"),
            }
        ],
        "temperature": 0.2,
    }


# ======================
# SAFE TOOL IMPLEMENTATION
# ======================
WORK_DIR = pathlib.Path("workspace")
WORK_DIR.mkdir(exist_ok=True)
TRANSCRIPTS_DIR = pathlib.Path("transcripts")
TRANSCRIPTS_DIR.mkdir(exist_ok=True)


def safe_path(rel_path: str) -> pathlib.Path:
    """Prevent directory traversal; constrain all writes to WORK_DIR."""
    p = (WORK_DIR / rel_path).resolve()
    if WORK_DIR.resolve() not in p.parents and p != WORK_DIR.resolve():
        raise ValueError("Unsafe path.")
    return p


def write_file_tool(filename: str, content: str) -> str:
    """
    Safely write a file inside workspace/.
    Returns the absolute file path written.
    """
    path = safe_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Wrote file: {path}"


def run_shell_tool(command: str) -> str:
    """
    Minimal safe shell runner: only allows a tiny whitelist of commands.
    (Extend with care; this is intentionally restrictive.)
    """
    import subprocess, shlex

    allowed_prefixes = ("echo ", "python ", "pytest", "ls", "cat ")
    if not any(command.startswith(p) for p in allowed_prefixes):
        return "Rejected: command not allowed by policy."

    try:
        res = subprocess.run(
            shlex.split(command),
            cwd=str(WORK_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = res.stdout.strip()
        err = res.stderr.strip()
        code = res.returncode
        return f"exit={code}\nstdout:\n{out}\n\nstderr:\n{err}"
    except Exception as e:
        return f"Error running command: {e}"


# ======================
# AGENT DEFINITIONS
# ======================
llm_config = build_llm_config()

pm = AssistantAgent(
    name="ProductManager",
    system_message=(
        "You are a pragmatic Product Manager. "
        "Clarify requirements, break features into small deliverables, and hand off a concise spec to engineering. "
        "Prefer crisp acceptance criteria and simple scope for an MVP."
    ),
    llm_config=llm_config,
)

dev = AssistantAgent(
    name="Developer",
    system_message=(
        "You are a senior software engineer. "
        "Write clean, minimal, production-friendly code. "
        "Return code in a single fenced block with the intended filename in the first line as a comment, e.g. '# file: app.py'. "
        "Add brief inline comments and avoid unnecessary dependencies."
    ),
    llm_config=llm_config,
)

qa = AssistantAgent(
    name="QAEngineer",
    system_message=(
        "You are a quality engineer. "
        "Review the spec and code, propose tests, and point out edge cases. "
        "If tests are missing, suggest a minimal pytest test file. Keep feedback actionable."
    ),
    llm_config=llm_config,
)

# The Human proxy: prompts you at checkpoints.
human = UserProxyAgent(
    name="Human",
    human_input_mode="ALWAYS",   # asks you for approval/inputs during the run
    code_execution_config=False, # we’ll gate tools ourselves
)


# ==================================
# OPTIONAL: REGISTER TOOLS FOR AGENTS
# (Tool-calling works best with OpenAI models)
# ==================================
@human.register_for_execution()  # exposes to Python runtime via the Human agent
def tool_write_file(filename: str, content: str) -> str:
    """Write content to a file under workspace/ (approved by human)."""
    return write_file_tool(filename, content)


@human.register_for_execution()
def tool_run_shell(command: str) -> str:
    """Run a limited shell command inside workspace/ (approved by human)."""
    return run_shell_tool(command)


# Advertise tools to LLMs (so they can choose to call them, if model supports function calling)
@pm.register_for_llm(description="Write content to a file under workspace/")
@dev.register_for_llm(description="Write content to a file under workspace/")
@qa.register_for_llm(description="Write content to a file under workspace/")
def write_file(filename: str, content: str) -> str:
    return tool_write_file(filename, content)


@pm.register_for_llm(description="Run a restricted shell command inside workspace/")
@dev.register_for_llm(description="Run a restricted shell command inside workspace/")
@qa.register_for_llm(description="Run a restricted shell command inside workspace/")
def run_shell(command: str) -> str:
    return tool_run_shell(command)


# ======================
# DRIVER / ORCHESTRATION
# ======================
def extract_code_block(text: str) -> Dict[str, str]:
    """
    Extract first fenced code block and attempt to read filename from first line '# file: <name>'.
    Returns dict with keys: filename (optional), code (required).
    """
    match = re.search(r"```[a-zA-Z0-9_-]*\n(.*?)```", text, flags=re.S)
    if not match:
        return {"code": ""}
    code = match.group(1)
    # Try to parse filename from first line comment
    first_line = code.splitlines()[0].strip() if code else ""
    filename = None
    if first_line.lower().startswith("# file:"):
        filename = first_line.split(":", 1)[1].strip()
        code = "\n".join(code.splitlines()[1:])  # drop the filename line
    return {"filename": filename, "code": code}


def save_transcript(messages: List[Dict]):
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = TRANSCRIPTS_DIR / f"session-{ts}.md"
    lines = []
    for m in messages:
        role = m.get("name") or m.get("role") or "unknown"
        content = m.get("content", "")
        lines.append(f"### {role}\n\n{content}\n\n---\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📒 Transcript saved to: {path}\n")


def main():
    print("\n=== Agentic MVP: PM → Dev → QA with Human approvals ===\n")
    feature = input("Enter a small feature request (e.g., 'Build a CLI that reverses a string'): ")

    # Create a group chat so agents can coordinate
    group = GroupChat(agents=[human, pm, dev, qa], messages=[], max_round=10)
    manager = GroupChatManager(groupchat=group, llm_config=llm_config)

    # Kick off: Human describes the task to PM
    human.initiate_chat(
        manager,
        message=(
            "You are a cross-functional team. "
            "PM: refine requirements and pass a concise spec to Dev. "
            "Dev: propose minimal code with a single file and tests. "
            "QA: review and suggest test improvements.\n\n"
            f"Feature request: {feature}\n"
            "Follow the order PM → Dev → QA. Keep outputs compact."
        ),
    )

    # After the group run, locate Dev’s code (best-effort) and propose tool actions
    # (We still gate them with human approval)
    print("\n--- Post-run actions (Human approval required) ---\n")
    # Find the latest Dev message
    dev_msgs = [m for m in group.messages if (m.get("name") or "").lower() == "developer"]
    code_text = dev_msgs[-1]["content"] if dev_msgs else ""
    parsed = extract_code_block(code_text)

    if parsed.get("code"):
        filename = parsed.get("filename") or "app.py"
        print(f"Developer proposed file: {filename}")
        print("Preview (first 30 lines):")
        print("\n".join(parsed["code"].splitlines()[:30]))
        if input("\nApprove writing this file to workspace/? (y/n): ").strip().lower() == "y":
            result = write_file_tool(filename, parsed["code"])
            print(result)
        else:
            print("Skipped writing file.")

    # Optionally run a QA command (e.g., 'pytest' or 'python app.py')
    if input("\nRun a test command (e.g., 'pytest' or 'python app.py')? (y/n): ").strip().lower() == "y":
        cmd = input("Enter command (whitelisted: echo/python/pytest/ls/cat): ").strip()
        print(run_shell_tool(cmd))

    # Save transcript
    save_transcript(group.messages)
    print("Done.\n")


if __name__ == "__main__":
    main()
````

---

## 2) How it works (line-by-line, compact)

### Config

* `build_llm_config()` — reads `MODEL_PROVIDER`:

  * **OpenAI**: uses `OPENAI_API_KEY` and model (default `gpt-4o-mini`)
  * **Ollama**: points to `http://localhost:11434/v1` with `OLLAMA_MODEL` (e.g., `llama3:8b`)
* This returns `llm_config` Autogen expects (`config_list`, `temperature`).

### Tools (safe & human-gated)

* `write_file_tool()` — writes files **only inside `workspace/`** (prevents `../` escapes).
* `run_shell_tool()` — tiny **whitelist** (`echo`, `python`, `pytest`, `ls`, `cat`) to avoid dangerous commands.

### Agents

* `ProductManager` — clarifies and hands off a concise spec.
* `Developer` — returns **one code block**; first line encodes a file name (`# file: app.py`).
* `QAEngineer` — reviews and proposes tests.
* `Human` — **UserProxyAgent** that asks you for input (approval prompts).

### Tool registration

* `@human.register_for_execution()` exposes Python functions the orchestrator can run.
* `@*.register_for_llm(...)` **advertises** tools to LLMs (function calling when supported).

  * OpenAI models will often call these tools automatically.
  * Many Ollama models ignore function-calling; you still have the **manual, human-gated** path below.

### Orchestration

* `GroupChat(...)` + `GroupChatManager(...)` — lets the 3 agents + you **talk in turn**.
* `human.initiate_chat(...)` — seeds the conversation with **flow instructions** (PM → Dev → QA) and your feature.

### Post-run human approvals

* We **extract** the developer’s latest code block (`extract_code_block()`).
* Show a preview and ask: **“Approve writing?”**

  * If yes → `write_file_tool()` creates it in `workspace/`.
* Optionally **run a command** (e.g., `pytest`) — again **approved by you**.

### Memory / transcripts

* Every message in the group chat is saved to `transcripts/session-<timestamp>.md` for later review and learning.

---

## 3) Try it quickly

```bash
python agentic_mvp.py
# Enter a small feature, e.g.
#   Build a CLI that reverses a string and includes a minimal test.
# Approve file write → run: python app.py  → or pytest if it created tests.
```

---

## 4) What to tweak next

* Swap `gpt-4o-mini` for your preferred model, or use `OLLAMA_MODEL=llama3:8b`.
* Add more tools:

  * GitHub gist/pull-request creator
  * Terraform planner (but gate with stricter whitelists)
* Replace `GroupChat` with **LangGraph** later for **deterministic** flows.

If you want, I can add a **GitHub file publisher tool** and a **tiny pytest scaffold** generator next, still human-approved.
