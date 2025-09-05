# Python App Development Agent System

## Overview

This directory contains an agentic AI system that simulates a cross-functional product development team using multiple AI agents powered by large language models (LLMs). The system is built with [AutoGen](https://github.com/microsoft/autogen), a framework for building agent-based applications.

## System Architecture

The system consists of three specialized agents working together:

1. **Product Manager (PM)**: Clarifies requirements, breaks features into deliverables, and creates concise specifications for the developer.
2. **Developer**: Writes clean, minimal, production-friendly code based on the PM's specifications.
3. **QA Engineer**: Reviews the code, proposes tests, and identifies potential edge cases.

A **Human** agent (you) is also in the loop, approving actions before they are executed and providing initial feature requests.

## How It Works

1. You provide a feature request (e.g., "Build a CLI that reverses a string").
2. The PM refines this request into a clear specification.
3. The Developer creates code based on the specification.
4. The QA Engineer reviews the code and suggests improvements.
5. You are asked to approve the generated code before it's written to disk.
6. All conversations are saved as transcripts for future reference.

## Key Components

- **product-dev-agentic-ai.py**: The main script that orchestrates the agents and workflow.
- **what-is-this-system.md**: Detailed technical explanation of the system implementation.
- **generated_code/**: Directory where all generated code is archived with timestamps.
- **transcripts/**: Directory where conversation logs are stored with timestamps.
- **workspace/**: Working directory where the approved code is written and can be executed.

## Safety Features

The system implements several safety measures:

- **Path Safety**: The `safe_path()` function prevents directory traversal attacks by constraining all file operations to the `workspace/` directory.
- **Limited Shell Commands**: The `run_shell_tool()` function only allows a small whitelist of commands (echo, python, pytest, ls, cat) to be executed.
- **Human Approval**: All file write and shell execution operations require explicit human approval.

## Configuration

The system can be configured to use either OpenAI's cloud-based models or local models through Ollama:

### For OpenAI

```bash
export MODEL_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini  # Optional, defaults to gpt-4o-mini
```

### For Ollama

```bash
export MODEL_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434/v1
export OLLAMA_MODEL=llama3:8b
```

## Sample Use Cases

The system has been used to develop:

1. **Streamlit Applications**: Interactive web applications for tasks like comparing LLM model outputs.
2. **Command Line Tools**: Simple utilities with specific functionality.
3. **Python Libraries**: Reusable code components with tests.

## Understanding the Output

After running the system, you'll have access to:

1. **Transcripts**: Complete conversation logs between agents, saved with timestamps in the `transcripts/` directory.
2. **Generated Code**: Code created by the Developer agent, saved with timestamps in the `generated_code/` directory.
3. **Working Application**: If you approve the code, it will be written to the `workspace/` directory where you can run and test it.

## Running the System

Execute the main script from the command line:

```bash
python product-dev-agentic-ai.py
```

When prompted, enter your feature request and follow the interactive process to approve or reject actions suggested by the agents.

## Example Projects

The system has been used to create various applications, including:

- A Streamlit application for comparing LLM models
- Simple CLI tools for text manipulation
- Data analysis scripts
- Web scraping utilities

## Extending the System

You can extend this system by:

1. Adding more specialized agent roles (e.g., Designer, Data Scientist)
2. Implementing additional tools for agents (e.g., database access, API calls)
3. Customizing the agent system messages to specialize in different domains
4. Adding automated testing and deployment capabilities

## Requirements

- Python 3.8+
- pyautogen
- openai (or Ollama for local models)
- Additional dependencies based on the specific applications you're building (e.g., streamlit)
