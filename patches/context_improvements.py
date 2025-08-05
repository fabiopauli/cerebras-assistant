import tiktoken
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Third-party imports
from pydantic import BaseModel

# Rich console imports
from rich.console import Console
from rich.panel import Panel

# Prompt toolkit imports
from prompt_toolkit import PromptSession

# Import our modules
import config
from config import (
    os_info, git_context, model_context, security_context,
    ADD_COMMAND_PREFIX, COMMIT_COMMAND_PREFIX, GIT_BRANCH_COMMAND_PREFIX,
    FUZZY_AVAILABLE, DEFAULT_MODEL, REASONER_MODEL, tools, SYSTEM_PROMPT,
    MAX_FILES_IN_ADD_DIR, MAX_FILE_CONTENT_SIZE_CREATE, EXCLUDED_FILES, EXCLUDED_EXTENSIONS,
    MAX_MULTIPLE_READ_SIZE
)
from utils import (
    console, detect_available_shells, get_context_usage_info, smart_truncate_history,
    validate_tool_calls, get_prompt_indicator, normalize_path, is_binary_file,
    read_local_file, add_file_context_smartly, find_best_matching_file,
    apply_fuzzy_diff_edit, run_bash_command, run_powershell_command,
    get_directory_tree_summary
)


def get_token_encoder(model_name: str):
    """
    Returns a token counting function for the given model.
    Uses tiktoken if available, falls back to character-based estimation.
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")  # Works well for Qwen/Cerebras
        
        def count_tokens(text: str) -> int:
            return len(encoding.encode(text))
            
        return count_tokens
    except (ImportError, RuntimeError):
        console.print("[yellow]⚠ tiktoken not available, using fallback estimation[/yellow]")
        
        def estimate_tokens(text: str) -> int:
            # Conservative estimate: 1 token ~ 3 chars for code
            return max(1, len(text) // 3)
            
        return estimate_tokens

def safe_read_file_with_lines(file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    """
    Read a file, optionally only a range of lines.
    """
    try:
        norm_path = normalize_path(file_path)
        path_obj = Path(norm_path)
        
        if not path_obj.exists():
            raise ValueError(f"File '{norm_path}' does not exist.")
        
        if is_binary_file(norm_path):
            return f"Error: '{norm_path}' is a binary file and cannot be read."
        
        with open(norm_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if start_line is not None and end_line is not None:
            # Convert to zero-based indexing
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            selected_lines = lines[start_idx:end_idx]
            content = ''.join(selected_lines)
            return f"Content of '{norm_path}' (lines {start_line}-{end_line}):

{content}"
        else:
            content = ''.join(lines)
            return f"Content of '{norm_path}':

{content}"
    
    except Exception as e:
        return f"Error reading file '{file_path}': {e}"

def improved_execute_function_call_dict(tool_call_dict: Dict[str, Any], conversation_history: List[Dict[str, Any]]) -> str:
    """
    Improved version with line-range support and better token-aware reading.
    """
    func_name = "unknown_function"
    try:
        func_name = tool_call_dict["function"]["name"]
        args = BaseModel.model_validate_json(tool_call_dict["function"]["arguments"])
        
        # Get token counter
        count_tokens = get_token_encoder(model_context.get('current_model'))

        if func_name == "read_file":
            file_path = args.file_path
            start_line = getattr(args, "start_line", None)
            end_line = getattr(args, "end_line", None)
            
            return safe_read_file_with_lines(file_path, start_line, end_line)
            
        elif func_name == "read_multiple_files":
            response_data = {
                "files_read": {},
                "errors": {}
            }
            
            # Use model context for limits
            from config import get_max_tokens_for_model
            current_model = model_context.get('current_model', DEFAULT_MODEL)
            max_tokens = get_max_tokens_for_model(current_model)
            max_total_tokens = int(max_tokens * 0.4)  # Reserve space for conversation
            
            total_tokens = 0
            
            for fp in args.file_paths:
                try:
                    norm_path = normalize_path(fp)
                    content = read_local_file(norm_path)
                    content_tokens = count_tokens(content)
                    
                    if total_tokens + content_tokens > max_total_tokens:
                        response_data["errors"][norm_path] = f"File would exceed context budget ({total_tokens + content_tokens} > {max_total_tokens} tokens)."
                        continue
                        
                    response_data["files_read"][norm_path] = content
                    total_tokens += content_tokens
                    
                except (OSError, ValueError) as e:
                    response_data["errors"][fp] = str(e)
            
            return f"Read {len(response_data['files_read'])} files.\n\n" + json.dumps(response_data, indent=2)
            
        # ... rest of tool handlers unchanged ...
        
    except Exception as e:
        console.print(f"[red]Error in {func_name}: {e}[/red]")
        return f"Error in {func_name}: {e}"
"""