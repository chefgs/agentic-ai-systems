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
GENERATED_DIR = pathlib.Path("generated_code")
GENERATED_DIR.mkdir(exist_ok=True)


def safe_path(rel_path: str) -> pathlib.Path:
    """Prevent directory traversal; constrain all writes to WORK_DIR."""
    p = (WORK_DIR / rel_path).resolve()
    if WORK_DIR.resolve() not in p.parents and p != WORK_DIR.resolve():
        raise ValueError("Unsafe path.")
    return p


def write_file_tool(filename: str = None, content: str = None) -> str:
    """
    Safely write a file inside workspace/.
    Returns the absolute file path written.
    """
    if not filename or not content:
        return "Error: Missing filename or content."
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
        "Do NOT call write_file yourself — the human will handle saving. "
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
# REGISTER TOOLS
# ==================================
@human.register_for_execution()
def tool_write_file(filename: str = None, content: str = None) -> str:
    """Write content to a file under workspace/ (approved by human)."""
    return write_file_tool(filename, content)


@human.register_for_execution()
def tool_run_shell(command: str) -> str:
    """Run a limited shell command inside workspace/ (approved by human)."""
    return run_shell_tool(command)


# ======================
# DRIVER / ORCHESTRATION
# ======================
def extract_code_block(text: str) -> Dict[str, str]:
    """Extract code block and filename if present."""
    match = re.search(r"```[a-zA-Z0-9_-]*\n(.*?)```", text, flags=re.S)
    if not match:
        return {"code": ""}
    code = match.group(1)
    first_line = code.splitlines()[0].strip() if code else ""
    filename = None
    if first_line.lower().startswith("# file:"):
        filename = first_line.split(":", 1)[1].strip()
        code = "\n".join(code.splitlines()[1:])
    return {"filename": filename, "code": code}


def save_generated_code(parsed: Dict[str, str], agent_name="Developer") -> str:
    """Always save generated code into generated_code/ with timestamp."""
    if not parsed.get("code"):
        return "No code found to save."
    ts = time.strftime("%Y%m%d-%H%M%S")
    filename = parsed.get("filename") or "snippet.py"
    dest = GENERATED_DIR / f"{ts}-{filename}"
    dest.write_text(parsed["code"], encoding="utf-8")
    return f"✅ Code saved for end users at {dest}"


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
    print("\n=== Agentic MVP (fixed): PM → Dev → QA with Human approvals ===\n")
    feature = input("Enter a small feature request (e.g., 'Build a CLI that reverses a string'): ")

    # Create a group chat so agents can coordinate
    group = GroupChat(agents=[human, pm, dev, qa], messages=[], max_round=10)
    manager = GroupChatManager(groupchat=group, llm_config=llm_config)

    # Kick off
    human.initiate_chat(
        manager,
        message=(
            "You are a cross-functional team. "
            "PM: refine requirements and pass a concise spec to Dev. "
            "Dev: propose minimal code with a single file and tests. "
            "QA: review and suggest improvements.\n\n"
            f"Feature request: {feature}\n"
            "Follow the order PM → Dev → QA."
        ),
    )

    # After the group run, get Developer output
    dev_msgs = [m for m in group.messages if (m.get("name") or "").lower() == "developer"]
    code_text = dev_msgs[-1]["content"] if dev_msgs else ""
    parsed = extract_code_block(code_text)

    # Always save for end-users
    msg = save_generated_code(parsed)
    print(msg)

    # Human approval for workspace write
    if parsed.get("code"):
        filename = parsed.get("filename") or "app.py"
        print(f"\nDeveloper proposed file: {filename}")
        print("Preview (first 20 lines):")
        print("\n".join(parsed["code"].splitlines()[:20]))
        if input("\nApprove writing this file to workspace/? (y/n): ").strip().lower() == "y":
            result = write_file_tool(filename, parsed["code"])
            print(result)
        else:
            print("Skipped writing file to workspace.")

    # Save transcript
    save_transcript(group.messages)
    print("Done.\n")


if __name__ == "__main__":
    main()
