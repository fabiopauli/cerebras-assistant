#!/usr/bin/env python3

"""
Configuration and constants for Cerebras Assistant
"""

import os
import json
import platform
from pathlib import Path
from typing import Dict, Any, Set

# =============================================================================
# GLOBAL STATE AND OS INFORMATION
# =============================================================================

# OS information
os_info: Dict[str, Any] = {
    'system': platform.system(),
    'release': platform.release(),
    'version': platform.version(),
    'machine': platform.machine(),
    'processor': platform.processor(),
    'python_version': platform.python_version(),
    'is_windows': platform.system() == "Windows",
    'is_mac': platform.system() == "Darwin",
    'is_linux': platform.system() == "Linux",
    'shell_available': {
        'bash': False,
        'powershell': False,
        'zsh': False,
        'cmd': False
    }
}

# Global base directory for operations (default: current working directory)
base_dir: Path = Path.cwd()

# Git context state
git_context: Dict[str, Any] = {
    'enabled': False,
    'skip_staging': False,
    'branch': None
}

# Model context state (will be initialized after config load)
model_context: Dict[str, Any] = {}

# Security settings (will be loaded from config)
security_context: Dict[str, Any] = {}

# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json file."""
    try:
        config_path = Path(__file__).parent / "config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# Load configuration
config = load_config()

# =============================================================================
# CONSTANTS (NON-CONFIGURABLE)
# =============================================================================

# Command prefixes
ADD_COMMAND_PREFIX: str = "/add "
COMMIT_COMMAND_PREFIX: str = "/git commit "
GIT_BRANCH_COMMAND_PREFIX: str = "/git branch "

# =============================================================================
# CONFIGURABLE CONSTANTS
# =============================================================================

# Configuration with fallbacks
file_limits = config.get("file_limits", {})
MAX_FILES_IN_ADD_DIR = file_limits.get("max_files_in_add_dir", 1000)
MAX_FILE_SIZE_IN_ADD_DIR = file_limits.get("max_file_size_in_add_dir", 5_000_000)
MAX_FILE_CONTENT_SIZE_CREATE = file_limits.get("max_file_content_size_create", 5_000_000)
MAX_MULTIPLE_READ_SIZE = file_limits.get("max_multiple_read_size", 100_000)

fuzzy_config = config.get("fuzzy_matching", {})
MIN_FUZZY_SCORE = fuzzy_config.get("min_fuzzy_score", 80)
MIN_EDIT_SCORE = fuzzy_config.get("min_edit_score", 85)

conversation_config = config.get("conversation", {})
MAX_HISTORY_MESSAGES = conversation_config.get("max_history_messages", 50)
MAX_CONTEXT_FILES = conversation_config.get("max_context_files", 5)
ESTIMATED_MAX_TOKENS = conversation_config.get("estimated_max_tokens", 66000)
TOKENS_PER_MESSAGE_ESTIMATE = conversation_config.get("tokens_per_message_estimate", 200)
TOKENS_PER_FILE_KB = conversation_config.get("tokens_per_file_kb", 300)
CONTEXT_WARNING_THRESHOLD = conversation_config.get("context_warning_threshold", 0.8)
AGGRESSIVE_TRUNCATION_THRESHOLD = conversation_config.get("aggressive_truncation_threshold", 0.9)

# Model-specific context limits
MODEL_CONTEXT_LIMITS = {
    "gpt-oss-120b": 60000,  
    "qwen-3-32b": 60000,  # 60k context limit
    "qwen-3-235b-a22b-instruct-2507": 60000,  # 60k context limit for reasoner model
}

def get_max_tokens_for_model(model_name: str) -> int:
    """Get the maximum context tokens for a specific model."""
    return MODEL_CONTEXT_LIMITS.get(model_name, 41000)  # Default to conservative limit

model_config = config.get("models", {})
DEFAULT_MODEL = model_config.get("default_model", "gpt-oss-120b")
REASONER_MODEL = model_config.get("reasoner_model", "qwen-3-32b")

security_config = config.get("security", {})
DEFAULT_SECURITY_CONTEXT = {
    'require_powershell_confirmation': security_config.get("require_powershell_confirmation", True),
    'require_bash_confirmation': security_config.get("require_bash_confirmation", True)
}

# File exclusion patterns from config
EXCLUDED_FILES: Set[str] = set(config.get("excluded_files", [
    ".DS_Store", "Thumbs.db", ".gitignore", ".python-version", "uv.lock", 
    ".uv", "uvenv", ".uvenv", ".venv", "venv", "__pycache__", ".pytest_cache", 
    ".coverage", ".mypy_cache", "node_modules", "package-lock.json", "yarn.lock", 
    "pnpm-lock.yaml", ".next", ".nuxt", "dist", "build", ".cache", ".parcel-cache", 
    ".turbo", ".vercel", ".output", ".contentlayer", "out", "coverage", 
    ".nyc_output", "storybook-static", ".env", ".env.local", ".env.development", 
    ".env.production", ".git", ".svn", ".hg", "CVS"
]))

EXCLUDED_EXTENSIONS: Set[str] = set(config.get("excluded_extensions", [
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".avif", 
    ".mp4", ".webm", ".mov", ".mp3", ".wav", ".ogg", ".zip", ".tar", 
    ".gz", ".7z", ".rar", ".exe", ".dll", ".so", ".dylib", ".bin", 
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".pyc", 
    ".pyo", ".pyd", ".egg", ".whl", ".uv", ".uvenv", ".db", ".sqlite", 
    ".sqlite3", ".log", ".idea", ".vscode", ".map", ".chunk.js", 
    ".chunk.css", ".min.js", ".min.css", ".bundle.js", ".bundle.css", 
    ".cache", ".tmp", ".temp", ".ttf", ".otf", ".woff", ".woff2", ".eot"
]))

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

def load_system_prompt() -> str:
    """Load system prompt from external file."""
    try:
        prompt_path = Path(__file__).parent / "system_prompt.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        return DEFAULT_SYSTEM_PROMPT

# Default system prompt (fallback)
DEFAULT_SYSTEM_PROMPT: str = f"""
You are an elite software engineer called Cerebras Engineer with decades of experience across all programming domains.
Your expertise spans system design, algorithms, testing, and best practices.
You provide thoughtful, well-structured solutions while explaining your reasoning.

**Current Environment:**
- Operating System: {os_info['system']} {os_info['release']}
- Machine: {os_info['machine']}
- Python: {os_info['python_version']}

Core capabilities:
1. Code Analysis & Discussion
   - Analyze code with expert-level insight
   - Explain complex concepts clearly
   - Suggest optimizations and best practices
   - Debug issues with precision

2. File Operations (via function calls):
   - read_file: Read a single file's content
   - read_multiple_files: Read multiple files at once (returns structured JSON)
   - create_file: Create or overwrite a single file
   - create_multiple_files: Create multiple files at once
   - edit_file: Make precise edits to existing files using fuzzy-matched snippet replacement

3. Git Operations (via function calls):
   - git_init: Initialize a new Git repository in the current directory.
   - git_add: Stage specified file(s) for the next commit. Use this before git_commit.
   - git_commit: Commit staged changes with a message. Ensure files are staged first using git_add.
   - git_create_branch: Create and switch to a new Git branch.
   - git_status: Show the current Git status, useful for seeing what is staged or unstaged.

4. System Operations (with security confirmation):
   - run_powershell: Execute PowerShell commands (Windows/Cross-platform PowerShell Core)
   - run_bash: Execute bash commands (macOS/Linux/WSL)
   
   Note: Choose the appropriate shell command based on the operating system:
   - On Windows: Prefer run_powershell
   - On macOS/Linux: Prefer run_bash
   - Both commands require user confirmation for security

Guidelines:
1. Provide natural, conversational responses explaining your reasoning
2. Use function calls when you need to read or modify files, or interact with Git.
3. For file operations:
   - The /add command and edit_file function now support fuzzy matching for more forgiving file operations
   - Always read files first before editing them to understand the context
   - The edit_file function will attempt fuzzy matching if exact snippets aren't found
   - Explain what changes you're making and why
   - Consider the impact of changes on the overall codebase
4. For Git operations:
   - Use `git_add` to stage files before using `git_commit`.
   - Provide clear commit messages.
   - Check `git_status` if unsure about the state of the repository.
5. For system commands:
   - Always consider the operating system when choosing between run_bash and run_powershell
   - Explain what the command does before executing
   - Use safe, non-destructive commands when possible
   - Be cautious with commands that modify system state
6. Follow language-specific best practices
7. Suggest tests or validation steps when appropriate
8. Be thorough in your analysis and recommendations

IMPORTANT: In your thinking process, if you realize that something requires a tool call, cut your thinking short and proceed directly to the tool call. Don't overthink - act efficiently when file operations are needed.

Remember: You're a senior engineer - be thoughtful, precise, and explain your reasoning clearly.
"""

# Load the actual system prompt
SYSTEM_PROMPT: str = load_system_prompt()

# Initialize global contexts with config values
model_context.update({
    'current_model': DEFAULT_MODEL,
    'is_reasoner': False
})
security_context.update(DEFAULT_SECURITY_CONTEXT)

# =============================================================================
# FUZZY MATCHING AVAILABILITY
# =============================================================================

# Fuzzy matching imports
try:
    from thefuzz import fuzz, process as fuzzy_process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

# =============================================================================
# FUNCTION CALLING TOOLS DEFINITION
# =============================================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the content of a single file from the filesystem",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string", "description": "The path to the file to read"}},
                "required": ["file_path"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_multiple_files",
            "description": "Read the content of multiple files",
            "parameters": {
                "type": "object",
                "properties": {"file_paths": {"type": "array", "items": {"type": "string"}, "description": "Array of file paths to read"}},
                "required": ["file_paths"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create or overwrite a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path for the file"},
                    "content": {"type": "string", "description": "Content for the file"}
                },
                "required": ["file_path", "content"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_multiple_files",
            "description": "Create multiple files",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                            "required": ["path", "content"]
                        },
                        "description": "Array of files to create (path, content)",
                    }
                },
                "required": ["files"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file by replacing a snippet (supports fuzzy matching)",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file"},
                    "original_snippet": {"type": "string", "description": "Snippet to replace (supports fuzzy matching)"},
                    "new_snippet": {"type": "string", "description": "Replacement snippet"}
                },
                "required": ["file_path", "original_snippet", "new_snippet"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_init",
            "description": "Initialize a new Git repository.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Commit staged changes with a message.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string", "description": "Commit message"}},
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_create_branch",
            "description": "Create and switch to a new Git branch.",
            "parameters": {
                "type": "object",
                "properties": {"branch_name": {"type": "string", "description": "Name of the new branch"}},
                "required": ["branch_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Show current Git status.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_add",
            "description": "Stage files for commit.",
            "parameters": {
                "type": "object",
                "properties": {"file_paths": {"type": "array", "items": {"type": "string"}, "description": "Paths of files to stage"}},
                "required": ["file_paths"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_powershell",
            "description": "Run a PowerShell command with security confirmation (Windows/Cross-platform PowerShell Core).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The PowerShell command to execute"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Run a bash command with security confirmation (macOS/Linux/WSL).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    }
                },
                "required": ["command"]
            }
        }
    }
]