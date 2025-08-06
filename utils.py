#!/usr/bin/env python3

"""
Utility functions for Cerebras Assistant
"""

import os
import sys
import json
import re
import time
import subprocess
import platform
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

# Rich console imports
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax

# Import configuration
import config
from config import (
    os_info, git_context, model_context, security_context, display_context,
    FUZZY_AVAILABLE, MIN_FUZZY_SCORE, MIN_EDIT_SCORE, MAX_FILE_CONTENT_SIZE_CREATE,
    ESTIMATED_MAX_TOKENS, CONTEXT_WARNING_THRESHOLD, AGGRESSIVE_TRUNCATION_THRESHOLD,
    MAX_HISTORY_MESSAGES, MAX_CONTEXT_FILES, MAX_MULTIPLE_READ_SIZE,
    EXCLUDED_FILES, EXCLUDED_EXTENSIONS, SYSTEM_PROMPT, DEFAULT_MODEL, REASONER_MODEL
)

if FUZZY_AVAILABLE:
    from thefuzz import fuzz, process as fuzzy_process

# Tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Initialize Rich console
console = Console()

# Token count cache for performance optimization
_token_cache = {}

# =============================================================================
# MARKDOWN RENDERING UTILITIES
# =============================================================================

def render_markdown_response(content: str, force_plain: bool = False) -> None:
    """
    Render AI response content as markdown with proper formatting for Windows and bash.
    
    Args:
        content: The content to render
        force_plain: If True, render as plain text without markdown formatting
    """
    if not content.strip():
        return
    
    # Check configuration for markdown rendering preferences
    should_force_plain = (force_plain or 
                         display_context.get("force_plain_text", False) or 
                         not display_context.get("enable_markdown_rendering", True))
    
    if should_force_plain or not _should_use_markdown_rendering(content):
        # Use plain text rendering for non-markdown content
        console.print(content, style="bright_magenta")
        return
    
    try:
        # Create markdown object with improved theme for cross-platform compatibility
        markdown_obj = Markdown(
            content,
            code_theme="monokai",  # Good contrast on both dark and light terminals
            hyperlinks=False,     # Disable hyperlinks for better compatibility
            inline_code_theme="bold magenta"  # Make inline code stand out
        )
        
        # Render with magenta style to match the AI response color scheme
        console.print(markdown_obj, style="bright_magenta")
        
    except Exception as e:
        # Fallback to plain text if markdown rendering fails
        console.print(f"[dim]Markdown rendering failed: {e}[/dim]")
        console.print(content, style="bright_magenta")

def _should_use_markdown_rendering(content: str) -> bool:
    """
    Determine if content should be rendered as markdown based on markdown indicators.
    
    Args:
        content: The content to check
        
    Returns:
        True if content appears to contain markdown
    """
    if not content:
        return False
    
    # Look for common markdown patterns
    markdown_indicators = [
        '```',           # Code blocks
        '**',            # Bold text
        '__',            # Bold/italic text
        '* ',            # Bullet lists
        '- ',            # Bullet lists  
        '+ ',            # Bullet lists
        '[',             # Links (might contain brackets)
        '`',             # Inline code
        '> ',            # Blockquotes
        '1. ',           # Numbered lists
        '2. ',           # Numbered lists
        '\n---',         # Horizontal rules
        '\n***',         # Horizontal rules
    ]
    
    # Check for headers (any level)
    has_headers = any(content.startswith('#') or '\n#' in content for line in content.split('\n'))
    
    # Count markdown indicators
    indicator_count = sum(1 for indicator in markdown_indicators if indicator in content)
    
    # Use markdown rendering if we find multiple indicators, strong ones, or headers
    return (indicator_count >= 2 or 
            '```' in content or 
            has_headers or
            content.count('- ') >= 2 or  # Multiple list items
            content.count('* ') >= 2)    # Multiple list items

def format_code_block(code: str, language: str = "text", title: str = None) -> None:
    """
    Display a code block with syntax highlighting and proper formatting.
    
    Args:
        code: The code content
        language: Programming language for syntax highlighting
        title: Optional title for the code block
    """
    try:
        syntax = Syntax(
            code,
            language,
            theme="monokai",
            line_numbers=True,
            word_wrap=False,
            background_color="default"
        )
        
        if title:
            console.print(Panel(syntax, title=title, border_style="blue"))
        else:
            console.print(syntax)
            
    except Exception as e:
        # Fallback to plain text with panel
        console.print(f"[dim]Syntax highlighting failed: {e}[/dim]")
        if title:
            console.print(Panel(code, title=title, border_style="yellow"))
        else:
            console.print(Panel(code, border_style="yellow"))

def enhance_terminal_output(content: str) -> str:
    """
    Enhance content for better terminal display across platforms.
    
    Args:
        content: Content to enhance
        
    Returns:
        Enhanced content with better formatting
    """
    if not content:
        return content
    
    # Fix Windows-specific line ending issues
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    
    # Ensure proper spacing around code blocks
    import re
    content = re.sub(r'\n```(\w+)?\n', r'\n\n```\1\n', content)
    content = re.sub(r'\n```\n', r'\n```\n\n', content)
    
    # Add proper spacing around headers
    content = re.sub(r'\n(#{1,6}\s+)', r'\n\n\1', content)
    
    # Ensure lists have proper spacing
    content = re.sub(r'\n([*+-]\s+)', r'\n\1', content)
    content = re.sub(r'\n(\d+\.\s+)', r'\n\1', content)
    
    return content

def _get_cache_key(content: str, content_type: str = "content") -> str:
    """Generate a cache key for token counting."""
    return f"{content_type}:{hash(content)}"

def clear_token_cache() -> None:
    """Clear the token counting cache to free memory."""
    global _token_cache
    _token_cache.clear()

def get_token_cache_size() -> int:
    """Get the current size of the token cache."""
    return len(_token_cache)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def detect_available_shells() -> None:
    """Detect which shells are available on the system."""
    global os_info
    
    # Check for bash
    try:
        result = subprocess.run(["bash", "--version"], capture_output=True, text=True, timeout=2)
        os_info['shell_available']['bash'] = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        os_info['shell_available']['bash'] = False
    
    # Check for PowerShell (Windows PowerShell or PowerShell Core)
    for ps_cmd in ["powershell", "pwsh"]:
        try:
            result = subprocess.run([ps_cmd, "-Command", "$PSVersionTable.PSVersion"], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                os_info['shell_available']['powershell'] = True
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    # Check for zsh (macOS default)
    try:
        result = subprocess.run(["zsh", "--version"], capture_output=True, text=True, timeout=2)
        os_info['shell_available']['zsh'] = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        os_info['shell_available']['zsh'] = False
    
    # Check for cmd (Windows)
    if os_info['is_windows']:
        os_info['shell_available']['cmd'] = True

def estimate_token_usage(conversation_history: List[Dict[str, Any]], model_name: str = None) -> Tuple[int, Dict[str, int]]:
    """
    Estimate token usage for the conversation history using tiktoken for accurate counting.
    
    Args:
        conversation_history: List of conversation messages
        model_name: Model name for encoding selection (optional)
        
    Returns:
        Tuple of (total_estimated_tokens, breakdown_by_role)
    """
    token_breakdown = {"system": 0, "user": 0, "assistant": 0, "tool": 0}
    total_tokens = 0
    
    # Clean cache if it gets too large (prevent memory bloat)
    if len(_token_cache) > 1000:
        _token_cache.clear()
    
    # Get appropriate encoding
    if TIKTOKEN_AVAILABLE:
        try:
            # Use cl100k_base encoding which is used by GPT-4, GPT-3.5, and most modern models
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback to basic encoding if cl100k_base fails
            encoding = tiktoken.get_encoding("gpt2")
    else:
        encoding = None
    
    for msg in conversation_history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        if encoding and TIKTOKEN_AVAILABLE:
            # Accurate token counting with tiktoken using cache
            try:
                # Check cache for content tokens
                content_key = _get_cache_key(content, "content")
                if content_key in _token_cache:
                    content_tokens = _token_cache[content_key]
                else:
                    content_tokens = len(encoding.encode(content)) if content else 0
                    _token_cache[content_key] = content_tokens
                
                # Add extra tokens for tool calls and structured data
                if msg.get("tool_calls"):
                    tool_calls_str = str(msg["tool_calls"])
                    tool_calls_key = _get_cache_key(tool_calls_str, "tool_calls")
                    if tool_calls_key in _token_cache:
                        content_tokens += _token_cache[tool_calls_key]
                    else:
                        tool_tokens = len(encoding.encode(tool_calls_str))
                        _token_cache[tool_calls_key] = tool_tokens
                        content_tokens += tool_tokens
                
                if msg.get("tool_call_id"):
                    # More accurate estimation for tool metadata
                    tool_id_str = str(msg.get("tool_call_id", ""))
                    name_str = str(msg.get("name", ""))
                    meta_str = tool_id_str + name_str
                    meta_key = _get_cache_key(meta_str, "tool_meta")
                    if meta_key in _token_cache:
                        content_tokens += _token_cache[meta_key]
                    else:
                        meta_tokens = len(encoding.encode(meta_str)) + 5  # Small overhead
                        _token_cache[meta_key] = meta_tokens
                        content_tokens += meta_tokens
                    
            except Exception:
                # Fallback to character-based estimation if tiktoken fails
                content_tokens = len(content) // 4
                if msg.get("tool_calls"):
                    content_tokens += len(str(msg["tool_calls"])) // 4
                if msg.get("tool_call_id"):
                    content_tokens += 10
        else:
            # Fallback to original character-based estimation when tiktoken unavailable
            content_tokens = len(content) // 4
            if msg.get("tool_calls"):
                content_tokens += len(str(msg["tool_calls"])) // 4
            if msg.get("tool_call_id"):
                content_tokens += 10
            
        token_breakdown[role] = token_breakdown.get(role, 0) + content_tokens
        total_tokens += content_tokens
    
    return total_tokens, token_breakdown

def get_context_usage_info(conversation_history: List[Dict[str, Any]], model_name: str = None) -> Dict[str, Any]:
    """
    Get comprehensive context usage information.
    
    Returns:
        Dictionary with context usage statistics
    """
    from config import get_max_tokens_for_model
    
    total_tokens, breakdown = estimate_token_usage(conversation_history, model_name)
    file_contexts = sum(1 for msg in conversation_history if msg["role"] == "system" and "User added file" in msg["content"])
    
    # Use model-specific context limit if available
    max_tokens = get_max_tokens_for_model(model_name) if model_name else ESTIMATED_MAX_TOKENS
    
    return {
        "total_messages": len(conversation_history),
        "estimated_tokens": total_tokens,
        "max_tokens": max_tokens,
        "token_usage_percent": (total_tokens / max_tokens) * 100,
        "file_contexts": file_contexts,
        "token_breakdown": breakdown,
        "approaching_limit": total_tokens > (max_tokens * CONTEXT_WARNING_THRESHOLD),
        "critical_limit": total_tokens > (max_tokens * AGGRESSIVE_TRUNCATION_THRESHOLD)
    }

def smart_truncate_history(conversation_history: List[Dict[str, Any]], max_messages: int = MAX_HISTORY_MESSAGES, model_name: str = None) -> List[Dict[str, Any]]:
    """
    Truncate conversation history while preserving tool call sequences and important context.
    Now uses token-based estimation for more intelligent truncation.
    
    Args:
        conversation_history: List of conversation messages
        max_messages: Maximum number of messages to keep (fallback limit)
        
    Returns:
        Truncated conversation history
    """
    # Get current context usage with model-specific limits
    context_info = get_context_usage_info(conversation_history, model_name)
    current_tokens = context_info["estimated_tokens"]
    max_tokens = context_info["max_tokens"]
    
    # If we're not approaching limits, use message-based truncation (but still check max_messages)
    if current_tokens < (max_tokens * CONTEXT_WARNING_THRESHOLD) and len(conversation_history) <= max_messages:
        return conversation_history
    
    # Determine target token count based on current usage
    if current_tokens > max_tokens:
        target_tokens = int(max_tokens * 0.5)  # Very aggressive reduction for severe overflow
        console.print(f"[red]🚨 Context severely exceeded ({current_tokens} > {max_tokens}). Force truncating to ~{target_tokens} tokens.[/red]")
    elif context_info["critical_limit"]:
        target_tokens = int(max_tokens * 0.6)  # Aggressive reduction
        console.print(f"[yellow]⚠ Critical context limit reached. Aggressively truncating to ~{target_tokens} tokens.[/yellow]")
    elif context_info["approaching_limit"]:
        target_tokens = int(max_tokens * 0.7)  # Moderate reduction
        console.print(f"[yellow]⚠ Context limit approaching. Truncating to ~{target_tokens} tokens.[/yellow]")
    else:
        target_tokens = int(max_tokens * 0.8)  # Gentle reduction
    
    # Separate system messages from conversation messages
    system_messages: List[Dict[str, Any]] = []
    other_messages: List[Dict[str, Any]] = []
    
    for msg in conversation_history:
        if msg["role"] == "system":
            system_messages.append(msg)
        else:
            other_messages.append(msg)
    
    # Always keep the main system prompt
    essential_system = [system_messages[0]] if system_messages else []
    
    # Handle file context messages more intelligently
    file_contexts = [msg for msg in system_messages[1:] if "User added file" in msg["content"]]
    if file_contexts:
        # Keep most recent and smallest file contexts
        file_contexts_with_size = []
        for msg in file_contexts:
            content_size = len(msg["content"])
            file_contexts_with_size.append((msg, content_size))
        
        # Sort by size (smaller first) and recency (newer first)
        file_contexts_with_size.sort(key=lambda x: (x[1], -file_contexts.index(x[0])))
        
        # Keep up to 3 file contexts that fit within token budget
        kept_file_contexts = []
        file_context_tokens = 0
        max_file_context_tokens = target_tokens // 4  # Reserve 25% for file contexts
        
        for msg, size in file_contexts_with_size[:3]:
            msg_tokens = size // 4
            if file_context_tokens + msg_tokens <= max_file_context_tokens:
                kept_file_contexts.append(msg)
                file_context_tokens += msg_tokens
            else:
                break
        
        essential_system.extend(kept_file_contexts)
    
    # Calculate remaining token budget for conversation messages
    system_tokens, _ = estimate_token_usage(essential_system, model_name)
    remaining_tokens = target_tokens - system_tokens
    
    # Work backwards through conversation messages, preserving tool call sequences
    keep_messages: List[Dict[str, Any]] = []
    current_token_count = 0
    i = len(other_messages) - 1
    
    while i >= 0 and current_token_count < remaining_tokens:
        current_msg = other_messages[i]
        msg_tokens = len(str(current_msg)) // 4
        
        # If this is a tool result, we need to keep the corresponding assistant message
        if current_msg["role"] == "tool":
            # Collect all tool results for this sequence
            tool_sequence: List[Dict[str, Any]] = []
            tool_sequence_tokens = 0
            
            while i >= 0 and other_messages[i]["role"] == "tool":
                tool_msg = other_messages[i]
                tool_msg_tokens = len(str(tool_msg)) // 4
                tool_sequence.insert(0, tool_msg)
                tool_sequence_tokens += tool_msg_tokens
                i -= 1
            
            # Find the corresponding assistant message with tool_calls
            assistant_msg = None
            assistant_tokens = 0
            if i >= 0 and other_messages[i]["role"] == "assistant" and other_messages[i].get("tool_calls"):
                assistant_msg = other_messages[i]
                assistant_tokens = len(str(assistant_msg)) // 4
                i -= 1
            
            # Check if the complete tool sequence fits in our budget
            total_sequence_tokens = tool_sequence_tokens + assistant_tokens
            if current_token_count + total_sequence_tokens <= remaining_tokens:
                # Add the complete sequence
                if assistant_msg:
                    keep_messages.insert(0, assistant_msg)
                    current_token_count += assistant_tokens
                keep_messages = tool_sequence + keep_messages
                current_token_count += tool_sequence_tokens
            else:
                # Sequence too large, stop here
                break
        else:
            # Regular message (user or assistant)
            if current_token_count + msg_tokens <= remaining_tokens:
                keep_messages.insert(0, current_msg)
                current_token_count += msg_tokens
                i -= 1
            else:
                # Message too large, stop here
                break
    
    # Combine system messages with kept conversation messages
    result = essential_system + keep_messages
    
    # Log truncation results
    final_tokens, _ = estimate_token_usage(result, model_name)
    console.print(f"[dim]Context truncated: {len(conversation_history)} → {len(result)} messages, ~{current_tokens} → ~{final_tokens} tokens[/dim]")
    
    return result

def validate_tool_calls(accumulated_tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate accumulated tool calls and provide debugging info.
    
    Args:
        accumulated_tool_calls: List of tool calls to validate
        
    Returns:
        List of valid tool calls
    """
    if not accumulated_tool_calls:
        return []
    
    valid_calls: List[Dict[str, Any]] = []
    for i, tool_call in enumerate(accumulated_tool_calls):
        # Check for required fields
        if not tool_call.get("id"):
            console.print(f"[yellow]⚠ Tool call {i} missing ID, skipping[/yellow]")
            continue
        
        func_name = tool_call.get("function", {}).get("name")
        if not func_name:
            console.print(f"[yellow]⚠ Tool call {i} missing function name, skipping[/yellow]")
            continue
        
        func_args = tool_call.get("function", {}).get("arguments", "")
        
        # Validate JSON arguments
        try:
            if func_args:
                json.loads(func_args)
        except json.JSONDecodeError as e:
            console.print(f"[red]✗ Tool call {i} has invalid JSON arguments: {e}[/red]")
            console.print(f"[red]  Arguments: {func_args}[/red]")
            continue
        
        valid_calls.append(tool_call)
    
    if len(valid_calls) != len(accumulated_tool_calls):
        console.print(f"[yellow]⚠ Kept {len(valid_calls)}/{len(accumulated_tool_calls)} tool calls[/yellow]")
    
    return valid_calls

def get_model_indicator() -> str:
    """
    Get the model indicator for the prompt.
    
    Returns:
        Emoji indicator for current model
    """
    return "🧠" if model_context['is_reasoner'] else "💬"

def get_prompt_indicator(conversation_history: List[Dict[str, Any]], model_name: str = None) -> str:
    """
    Get the full prompt indicator including git, model, and context status.
    
    Returns:
        Formatted prompt indicator string
    """
    indicators = []
    
    # Add model indicator
    indicators.append(get_model_indicator())
    
    # Add git branch if enabled
    if git_context['enabled'] and git_context['branch']:
        indicators.append(f"🌳 {git_context['branch']}")
    
    # Add context status indicator with model-specific limits
    context_info = get_context_usage_info(conversation_history, model_name)
    if context_info["critical_limit"]:
        indicators.append("🔴")  # Critical context usage
    elif context_info["approaching_limit"]:
        indicators.append("🟡")  # Warning context usage
    else:
        indicators.append("🔵")  # Normal context usage
    
    return " ".join(indicators)

def normalize_path(path_str: str, allow_outside_project: bool = False) -> str:
    """
    Normalize a file path relative to the base directory with security validation.
    
    Args:
        path_str: Path string to normalize
        allow_outside_project: If False, restricts paths to within the project directory
        
    Returns:
        Normalized absolute path string
        
    Raises:
        ValueError: If path is outside project directory when allow_outside_project=False
    """
    try:
        p = Path(path_str)
        
        # If path is absolute, use it as-is
        if p.is_absolute():
            if p.exists() or p.is_symlink(): 
                resolved_p = p.resolve(strict=True) 
            else:
                resolved_p = p.resolve()
        else:
            # For relative paths, resolve against config.base_dir instead of cwd
            base_path = config.base_dir / p
            if base_path.exists() or base_path.is_symlink():
                resolved_p = base_path.resolve(strict=True)
            else:
                resolved_p = base_path.resolve()
                
    except (FileNotFoundError, RuntimeError): 
        # Fallback: resolve relative to config.base_dir
        p = Path(path_str)
        if p.is_absolute():
            resolved_p = p.resolve()
        else:
            resolved_p = (config.base_dir / p).resolve()
    
    # Security validation: ensure path is within project directory
    if not allow_outside_project:
        base_resolved = config.base_dir.resolve()
        try:
            # Check if the resolved path is within the base directory
            resolved_p.relative_to(base_resolved)
        except ValueError:
            raise ValueError(f"Path '{path_str}' is outside the project directory '{base_resolved}'. "
                           "This is not allowed for security reasons.")
    
    return str(resolved_p)

def is_binary_file(file_path: str, peek_size: int = 1024) -> bool:
    """
    Check if a file is binary by looking for null bytes.
    
    Args:
        file_path: Path to the file to check
        peek_size: Number of bytes to check
        
    Returns:
        True if file appears to be binary
    """
    try:
        with open(file_path, 'rb') as f: 
            chunk = f.read(peek_size)
        return b'\0' in chunk
    except Exception: 
        return True

def read_local_file(file_path: str) -> str:
    """
    Read content from a local file.
    
    Args:
        file_path: Path to the file to read (should already be normalized/resolved)
        
    Returns:
        File content as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        UnicodeDecodeError: If file can't be decoded as UTF-8
    """
    # Use the path directly since it should already be normalized by normalize_path()
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def add_file_context_smartly(conversation_history: List[Dict[str, Any]], file_path: str, content: str, max_context_files: int = MAX_CONTEXT_FILES) -> bool:
    """
    Add file context while managing system message bloat and avoiding duplicates.
    Now includes token-aware management and file size limits.
    Also ensures file context doesn't break tool call conversation flow.

    Args:
        conversation_history: List of conversation messages
        file_path: Path to the file being added
        content: Content of the file
        max_context_files: Maximum number of file contexts to keep

    Returns:
        True if file was added successfully, False if rejected due to size limits
    """
    marker = f"User added file '{file_path}'"

    # Check file size and context limits
    content_size_kb = len(content) / 1024
    estimated_tokens = len(content) // 4
    context_info = get_context_usage_info(conversation_history)

    # Only reject files that would use more than 80% of context
    MAX_SINGLE_FILE_TOKENS = int(ESTIMATED_MAX_TOKENS * 0.8)
    if estimated_tokens > MAX_SINGLE_FILE_TOKENS:
        console.print(f"[yellow]⚠ File '{file_path}' too large ({content_size_kb:.1f}KB, ~{estimated_tokens} tokens). Limit is 80% of context window.[/yellow]")
        return False

    # Check if there are any pending tool calls that haven't been responded to
    # If so, defer adding file context to avoid interrupting tool call sequences
    if conversation_history:
        # Look for the most recent assistant message with tool calls
        pending_tool_calls = set()
        
        # Go through messages in reverse to find pending tool calls
        for msg in reversed(conversation_history):
            if msg.get("role") == "tool" and msg.get("tool_call_id"):
                # Remove this tool call ID from pending set
                pending_tool_calls.discard(msg.get("tool_call_id"))
            elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                # Add all tool call IDs from this assistant message to pending set
                for tc in msg["tool_calls"]:
                    pending_tool_calls.add(tc["id"])
                # Stop at the first assistant message with tool calls (most recent sequence)
                break
        
        # If there are still pending tool calls, defer file context addition
        if pending_tool_calls:
            console.print(f"[dim]Deferring file context addition for '{Path(file_path).name}' until {len(pending_tool_calls)} tool responses complete[/dim]")
            return True  # Return True but don't add yet

    # Remove any existing context for this exact file to avoid duplicates
    conversation_history[:] = [
        msg for msg in conversation_history 
        if not (msg["role"] == "system" and marker in msg["content"])
    ]

    # Get current file contexts and their sizes
    file_contexts = []
    for msg in conversation_history:
        if msg["role"] == "system" and "User added file" in msg["content"]:
            # Extract file path from marker
            lines = msg["content"].split("\n", 1)
            if lines:
                context_file_path = lines[0].replace("User added file '", "").replace("'. Content:", "")
                context_size = len(msg["content"])
                file_contexts.append((msg, context_file_path, context_size))

    # If we're at the file limit, remove the largest or oldest file contexts
    while len(file_contexts) >= max_context_files:
        if context_info["approaching_limit"]:
            # Remove largest file context when approaching limits
            file_contexts.sort(key=lambda x: x[2], reverse=True)  # Sort by size, largest first
            to_remove = file_contexts.pop(0)
            console.print(f"[dim]Removed large file context: {Path(to_remove[1]).name} ({to_remove[2]//1024:.1f}KB)[/dim]")
        else:
            # Remove oldest file context normally
            to_remove = file_contexts.pop(0)
            console.print(f"[dim]Removed old file context: {Path(to_remove[1]).name}[/dim]")

        # Remove from conversation history
        conversation_history[:] = [msg for msg in conversation_history if msg != to_remove[0]]

    # Find the right position to insert the file context
    # Insert after system messages but before the most recent conversation turn
    insertion_point = len(conversation_history)
    
    # Find the start of the most recent conversation turn (user message + any responses)
    for i in range(len(conversation_history) - 1, -1, -1):
        msg = conversation_history[i]
        if msg.get("role") == "user":
            # Found a user message, insert before it (but after any preceding system messages)
            insertion_point = i
            break
        elif msg.get("role") == "system":
            # If we hit a system message without finding a user message, insert after system messages
            insertion_point = i + 1
            break
    
    # Ensure we don't insert in the middle of tool call sequences
    # If the insertion point would break a tool sequence, move it to a safer position
    if insertion_point < len(conversation_history):
        # Check if we're inserting between an assistant message with tool calls and tool responses
        if insertion_point > 0:
            prev_msg = conversation_history[insertion_point - 1]
            if (prev_msg.get("role") == "assistant" and prev_msg.get("tool_calls")):
                # Find the end of this tool sequence
                tool_call_ids = {tc["id"] for tc in prev_msg["tool_calls"]}
                for j in range(insertion_point, len(conversation_history)):
                    msg = conversation_history[j]
                    if msg.get("role") == "tool" and msg.get("tool_call_id") in tool_call_ids:
                        tool_call_ids.discard(msg.get("tool_call_id"))
                        if not tool_call_ids:  # All tool calls have responses
                            insertion_point = j + 1
                            break

    # Add new file context at the appropriate position
    new_context_msg = {
        "role": "system", 
        "content": f"{marker}. Content:\n\n{content}"
    }
    conversation_history.insert(insertion_point, new_context_msg)

    # Log the addition
    console.print(f"[dim]Added file context: {Path(file_path).name} ({content_size_kb:.1f}KB, ~{estimated_tokens} tokens)[/dim]")

    return True

# =============================================================================
# FUZZY MATCHING UTILITIES
# =============================================================================

def find_best_matching_file(root_dir: Path, user_path: str, min_score: int = MIN_FUZZY_SCORE) -> Optional[str]:
    """
    Find the best file match for a given user path within a directory.

    Args:
        root_dir: The directory to search within.
        user_path: The (potentially messy) path provided by the user.
        min_score: The minimum fuzzy match score to consider a match.

    Returns:
        The full, corrected path of the best match, or None if no good match is found.
    """
    if not FUZZY_AVAILABLE:
        return None
        
    best_match = None
    highest_score = 0
    
    # Use the filename from the user's path for matching
    user_filename = Path(user_path).name

    for dirpath, _, filenames in os.walk(root_dir):
        # Skip hidden directories and excluded patterns for efficiency
        if any(part in EXCLUDED_FILES or part.startswith('.') for part in Path(dirpath).parts):
            continue

        for filename in filenames:
            if filename in EXCLUDED_FILES or os.path.splitext(filename)[1] in EXCLUDED_EXTENSIONS:
                continue

            # Compare user's filename with actual filenames
            score = fuzz.ratio(user_filename.lower(), filename.lower())
            
            # Boost score for files in the immediate directory
            if Path(dirpath) == root_dir:
                score += 10

            if score > highest_score:
                highest_score = score
                best_match = os.path.join(dirpath, filename)

    if highest_score >= min_score:
        return str(Path(best_match).resolve())
    
    return None

def apply_fuzzy_diff_edit(path: str, original_snippet: str, new_snippet: str) -> None:
    """
    Apply a diff edit to a file by replacing original snippet with new snippet.
    Uses fuzzy matching to find the best location for the snippet.
    """
    
    normalized_path_str = normalize_path(path)
    content = ""
    try:
        content = read_local_file(normalized_path_str)
        
        # 1. First, try for an exact match for performance and accuracy
        if content.count(original_snippet) == 1:
            updated_content = content.replace(original_snippet, new_snippet, 1)
            with open(normalized_path_str, "w", encoding="utf-8") as f:
                f.write(updated_content)
            console.print(f"[bold blue]✓[/bold blue] Applied exact diff edit to '[bright_cyan]{normalized_path_str}[/bright_cyan]'")
            return

        # 2. If exact match fails, use fuzzy matching (if available)
        if not FUZZY_AVAILABLE:
            raise ValueError("Original snippet not found and fuzzy matching not available")
            
        console.print("[dim]Exact snippet not found. Trying fuzzy matching...[/dim]")

        # Create a list of "choices" to match against. These are overlapping chunks of the file.
        lines = content.split('\n')
        original_lines_count = len(original_snippet.split('\n'))
        
        # Create sliding window of text chunks
        choices = []
        for i in range(len(lines) - original_lines_count + 1):
            chunk = '\n'.join(lines[i:i+original_lines_count])
            choices.append(chunk)
        
        if not choices:
            raise ValueError("File content is too short to perform a fuzzy match.")

        # Find the best match
        best_match, score = fuzzy_process.extractOne(original_snippet, choices)

        if score < MIN_EDIT_SCORE:
            raise ValueError(f"Fuzzy match score ({score}) is below threshold ({MIN_EDIT_SCORE}). Snippet not found or too different.")

        # Ensure the best match is unique to avoid ambiguity
        if choices.count(best_match) > 1:
            raise ValueError(f"Ambiguous fuzzy edit: The best matching snippet appears multiple times in the file.")
        
        # Replace the best fuzzy match
        updated_content = content.replace(best_match, new_snippet, 1)
        with open(normalized_path_str, "w", encoding="utf-8") as f:
            f.write(updated_content)
        console.print(f"[bold blue]✓[/bold blue] Applied [bold]fuzzy[/bold] diff edit to '[bright_cyan]{normalized_path_str}[/bright_cyan]' (score: {score})")

    except FileNotFoundError:
        console.print(f"[bold red]✗[/bold red] File not found for diff: '[bright_cyan]{path}[/bright_cyan]'")
        raise
    except ValueError as e:
        console.print(f"[bold yellow]⚠[/bold yellow] {str(e)} in '[bright_cyan]{path}[/bright_cyan]'. No changes.")
        if "Original snippet not found" in str(e) or "Fuzzy match score" in str(e) or "Ambiguous edit" in str(e):
            console.print("\n[bold blue]Expected snippet:[/bold blue]")
            console.print(Panel(original_snippet, title="Expected", border_style="blue"))
            if content:
                console.print("\n[bold blue]Actual content (or relevant part):[/bold blue]")
                start_idx = max(0, content.find(original_snippet[:20]) - 100)
                end_idx = min(len(content), start_idx + len(original_snippet) + 200)
                display_snip = ("..." if start_idx > 0 else "") + content[start_idx:end_idx] + ("..." if end_idx < len(content) else "")
                console.print(Panel(display_snip or content, title="Actual", border_style="yellow"))
        raise

# =============================================================================
# SHELL COMMAND UTILITIES
# =============================================================================

def run_bash_command(command: str, cwd: Optional[Union[str, Path]] = None) -> Tuple[str, str]:
    """
    Run a bash command and return (stdout, stderr).
    Includes platform checks to ensure bash is available.
    
    Args:
        command: The bash command to execute
        cwd: Working directory for the command (defaults to config.base_dir if None)
    """
    # Check if we're on a supported platform
    current_platform = platform.system()
    if current_platform == "Windows":
        # Try WSL bash first, then Git Bash
        bash_paths = ["wsl", "bash"]
        bash_executable = None
        
        for bash_cmd in bash_paths:
            try:
                test_run = subprocess.run(
                    [bash_cmd, "-c", "echo test"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if test_run.returncode == 0:
                    bash_executable = bash_cmd
                    break
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        if not bash_executable:
            return "", "Bash not available on Windows. Install WSL or Git Bash."
    else:
        bash_executable = "bash"
    
    # Test if bash is available
    try:
        bash_check = subprocess.run(
            [bash_executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if bash_check.returncode != 0:
            return "", f"Bash not available or not working properly on {current_platform}"
        
        bash_info = bash_check.stdout.split('\n')[0] if bash_check.stdout else "Bash"
        console.print(f"[dim]Running {bash_info} on {current_platform}[/dim]")
        
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return "", f"Bash not found on {current_platform}: {e}"
    
    # Set working directory - use config.base_dir if cwd not specified
    if cwd is None:
        cwd = config.base_dir
    working_dir = str(Path(cwd).resolve())
    
    # Execute the actual command
    try:
        if bash_executable == "wsl":
            # For WSL, we need to format the command differently
            completed = subprocess.run(
                ["wsl", "bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=working_dir
            )
        else:
            completed = subprocess.run(
                [bash_executable, "-c", command],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=working_dir
            )
        return completed.stdout, completed.stderr
    except subprocess.TimeoutExpired:
        return "", "Bash command timed out after 30 seconds"
    except Exception as e:
        return "", f"Error executing bash command: {e}"

def run_powershell_command(command: str, cwd: Optional[Union[str, Path]] = None) -> Tuple[str, str]:
    """
    Run a PowerShell command and return (stdout, stderr).
    Includes platform checks to ensure PowerShell is available.
    
    Args:
        command: The PowerShell command to execute
        cwd: Working directory for the command (defaults to config.base_dir if None)
    """
    # Check if we're on a supported platform
    current_platform = platform.system()
    if current_platform not in ["Windows", "Linux", "Darwin"]:
        return "", f"PowerShell is not supported on {current_platform}"
    
    # Determine the PowerShell executable
    pwsh_executable = "pwsh"  # PowerShell Core (cross-platform)
    if current_platform == "Windows":
        # Try Windows PowerShell first, fall back to PowerShell Core
        try:
            test_run = subprocess.run(
                ["powershell", "-Command", "$PSVersionTable.PSEdition"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if test_run.returncode == 0:
                pwsh_executable = "powershell"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # Fall back to pwsh
    
    # Test if PowerShell is available
    try:
        os_check = subprocess.run(
            [pwsh_executable, "-Command", "$PSVersionTable.PSEdition"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if os_check.returncode != 0:
            return "", f"PowerShell not available or not working properly on {current_platform}"
        
        os_info = os_check.stdout.strip() if os_check.stdout else "PowerShell"
        console.print(f"[dim]Running PowerShell ({os_info}) on {current_platform}[/dim]")
        
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return "", f"PowerShell not found on {current_platform}: {e}"
    
    # Set working directory - use config.base_dir if cwd not specified
    if cwd is None:
        cwd = config.base_dir
    working_dir = str(Path(cwd).resolve())
    
    # Execute the actual command
    try:
        completed = subprocess.run(
            [pwsh_executable, "-Command", command],
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout for safety
            cwd=working_dir
        )
        return completed.stdout, completed.stderr
    except subprocess.TimeoutExpired:
        return "", "PowerShell command timed out after 30 seconds"
    except Exception as e:
        return "", f"Error executing PowerShell command: {e}"

def get_directory_tree_summary(root_dir: Path, max_depth: int = 3, max_entries: int = 100) -> str:
    """
    Generate a summary of the directory tree up to a certain depth and entry count.
    """
    lines = []
    entry_count = 0

    def walk(dir_path: Path, prefix: str = "", depth: int = 0):
        nonlocal entry_count
        if depth > max_depth or entry_count >= max_entries:
            return
        try:
            entries = sorted([e for e in dir_path.iterdir() if not e.name.startswith('.')])
        except Exception:
            return
        for entry in entries:
            if entry_count >= max_entries:
                lines.append(f"{prefix}... (truncated)")
                return
            if entry.is_dir():
                lines.append(f"{prefix}{entry.name}/")
                entry_count += 1
                walk(entry, prefix + "  ", depth + 1)
            else:
                lines.append(f"{prefix}{entry.name}")
                entry_count += 1

    walk(root_dir)
    return "\n".join(lines)