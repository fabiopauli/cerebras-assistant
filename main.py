#!/usr/bin/env python3

"""
Main application for Cerebras Assistant
        # Sanitize tools for current model (remove unsupported fields like 'strict')
        sanitized_tools = []
        for tool in tools:
            clean_tool = tool.copy()
            func = clean_tool["function"].copy()
            func.pop("strict", None)  # Remove 'strict' if present
            clean_tool["function"] = func
            sanitized_tools.append(clean_tool)
            
        kwargs = {
            "model": current_model,
            "messages": conversation_history,
            "tools": sanitized_tools,
            "tool_choice": "auto",
            "stream": False,
            "max_completion_tokens": 5000,
            "temperature": 0.7,
            "top_p": 1
        }
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Third-party imports
from cerebras.cloud.sdk import Cerebras
from pydantic import BaseModel
from dotenv import load_dotenv

# Rich console imports
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Prompt toolkit imports
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PromptStyle

# Import our modules
import config
from config import (
    os_info, model_context, security_context,
    ADD_COMMAND_PREFIX,
    FUZZY_AVAILABLE, DEFAULT_MODEL, REASONER_MODEL, tools, SYSTEM_PROMPT,
    MAX_FILES_IN_ADD_DIR, MAX_FILE_CONTENT_SIZE_CREATE, EXCLUDED_FILES, EXCLUDED_EXTENSIONS,
)
from utils import (
    console, detect_available_shells, get_context_usage_info, smart_truncate_history,
    validate_tool_calls, get_prompt_indicator, normalize_path, is_binary_file,
    read_local_file, add_file_context_smartly, find_best_matching_file,
    apply_fuzzy_diff_edit, run_bash_command, run_powershell_command,
    get_directory_tree_summary, render_markdown_response, enhance_terminal_output
)

# Initialize Cerebras client
load_dotenv()
client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))

# Initialize prompt session
prompt_session = PromptSession(
    style=PromptStyle.from_dict({
        'prompt': '#0066ff bold',
        'completion-menu.completion': 'bg:#1e3a8a fg:#ffffff',
        'completion-menu.completion.current': 'bg:#3b82f6 fg:#ffffff bold',
    })
)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class FileToCreate(BaseModel):
    path: str
    content: str

class FileToEdit(BaseModel):
    path: str
    original_snippet: str
    new_snippet: str

# =============================================================================
# FILE OPERATIONS
# =============================================================================

def create_file(path: str, content: str, require_confirmation: bool = True) -> None:
    """
    Create or overwrite a file with given content.
    
    Args:
        path: File path
        content: File content
        require_confirmation: If True, prompt for confirmation when overwriting existing files
        
    Raises:
        ValueError: If file content exceeds size limit, path contains invalid characters, 
                   or user cancels overwrite
    """
    file_path = Path(path)
    if any(part.startswith('~') for part in file_path.parts):
        raise ValueError("Home directory references not allowed")
    
    # Check content size limit
    if len(content.encode('utf-8')) > MAX_FILE_CONTENT_SIZE_CREATE:
        raise ValueError(f"File content exceeds maximum size limit of {MAX_FILE_CONTENT_SIZE_CREATE} bytes")
    
    normalized_path_str = normalize_path(str(file_path))
    normalized_path = Path(normalized_path_str)
    
    # Check if file exists and prompt for confirmation if required
    if require_confirmation and normalized_path.exists():
        try:
            # Get file info for the confirmation prompt
            file_size = normalized_path.stat().st_size
            file_size_str = f"{file_size:,} bytes" if file_size < 1024 else f"{file_size/1024:.1f} KB"
            
            confirm = prompt_session.prompt(
                f"🔵 File '{normalized_path_str}' exists ({file_size_str}). Overwrite? (y/N): ",
                default="n"
            ).strip().lower()
            
            if confirm not in ["y", "yes"]:
                raise ValueError("File overwrite cancelled by user")
                
        except (KeyboardInterrupt, EOFError):
            raise ValueError("File overwrite cancelled by user")
    
    # Create the file
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    with open(normalized_path_str, "w", encoding="utf-8") as f:
        f.write(content)
    
    action = "Updated" if normalized_path.exists() else "Created"
    console.print(f"[bold blue]✓[/bold blue] {action} file at '[bright_cyan]{normalized_path_str}[/bright_cyan]'")
    
    # Git staging removed - agent can use bash commands for git operations

def add_directory_to_conversation(directory_path: str, conversation_history: List[Dict[str, Any]]) -> None:
    """
    Add all files from a directory to the conversation context.
    
    Args:
        directory_path: Path to directory to scan
        conversation_history: Conversation history to add files to
    """
    with console.status("[bold bright_blue]🔍 Scanning directory...[/bold bright_blue]") as status:
        skipped: List[str] = []
        added: List[str] = []
        total_processed = 0
        
        for root, dirs, files in os.walk(directory_path):
            if total_processed >= MAX_FILES_IN_ADD_DIR: 
                console.print(f"[yellow]⚠ Max files ({MAX_FILES_IN_ADD_DIR}) reached for dir scan.")
                break
            status.update(f"[bold bright_blue]🔍 Scanning {root}...[/bold bright_blue]")
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in EXCLUDED_FILES]
            
            for file in files:
                if total_processed >= MAX_FILES_IN_ADD_DIR: 
                    break
                if (file.startswith('.') or 
                    file in EXCLUDED_FILES or 
                    os.path.splitext(file)[1] in EXCLUDED_EXTENSIONS):
                    continue
                    
                full_path = os.path.join(root, file)
                try:
                    if is_binary_file(full_path): 
                        skipped.append(f"{full_path} (binary)")
                        continue
                        
                    norm_path = normalize_path(full_path)
                    content = read_local_file(norm_path)
                    if add_file_context_smartly(conversation_history, norm_path, content):
                        added.append(norm_path)
                    else:
                        skipped.append(f"{full_path} (too large for context)")
                    total_processed += 1
                except (OSError, ValueError) as e: 
                    skipped.append(f"{full_path} (error: {e})")
                    
        console.print(f"[bold blue]✓[/bold blue] Added folder '[bright_cyan]{directory_path}[/bright_cyan]'.")
        if added: 
            console.print(f"\n[bold bright_blue]📁 Added:[/bold bright_blue] ({len(added)} of {total_processed} valid) {[Path(f).name for f in added[:5]]}{'...' if len(added) > 5 else ''}")
        if skipped: 
            console.print(f"\n[yellow]⏭ Skipped:[/yellow] ({len(skipped)}) {[Path(f).name for f in skipped[:3]]}{'...' if len(skipped) > 3 else ''}")
        console.print()

# =============================================================================
# GIT OPERATIONS - REMOVED
# All git operations have been removed - agent can use bash commands instead
# =============================================================================

# =============================================================================
# COMMAND HANDLERS
# =============================================================================

# Git command handlers removed - agent can use bash commands for git operations



# Git info command removed - agent can use bash commands for git operations

def try_handle_r1_command(user_input: str, conversation_history: List[Dict[str, Any]]) -> bool:
    """Handle /r command for one-off reasoner calls."""
    if user_input.strip().lower() == "/r":
        try:
            user_prompt = prompt_session.prompt("🔵 Enter your reasoning prompt: ").strip()
            if not user_prompt:
                console.print("[yellow]No input provided. Aborting.[/yellow]")
                return True
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Cancelled.[/yellow]")
            return True
        
        temp_conversation = conversation_history + [{"role": "user", "content": user_prompt}]
        
        try:
            with console.status("[bold yellow]Qwen (R1) is thinking...[/bold yellow]", spinner="dots"):
                response = client.chat.completions.create(
                    model=REASONER_MODEL,
                    messages=temp_conversation,
                    tools=tools,
                    tool_choice="auto",
                    stream=False,
                    max_completion_tokens=5000,
                    temperature=0.7,
                    top_p=1
                )
            
            message = response.choices[0].message
            full_response_content = message.content or ""
            accumulated_tool_calls = []
            
            # Extract tool calls if present
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    accumulated_tool_calls.append({
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
            
            console.print("[bold bright_blue]🧠 Qwen:[/bold bright_blue]")
            if full_response_content:
                clean_content = full_response_content.replace("<think>", "").replace("</think>", "")
                enhanced_content = enhance_terminal_output(clean_content)
                render_markdown_response(enhanced_content)
            else:
                console.print("[dim]Processing tool calls...[/dim]", style="bright_blue")
            
            conversation_history.append({"role": "user", "content": user_prompt})
            assistant_message = {"role": "assistant", "content": full_response_content}
            
            valid_tool_calls = validate_tool_calls(accumulated_tool_calls)
            if valid_tool_calls:
                assistant_message["tool_calls"] = valid_tool_calls
                console.print("[dim]Note: R1 reasoner made tool calls. Executing...[/dim]")
                for tool_call in valid_tool_calls:
                    try:
                        result = execute_function_call_dict(tool_call)
                        tool_response = {
                            "role": "tool",
                            "content": str(result),
                            "tool_call_id": tool_call["id"]
                        }
                        conversation_history.append(tool_response)
                    except Exception as e:
                        console.print(f"[red]✗ R1 tool call error: {e}[/red]")
            
            conversation_history.append(assistant_message)
            return True
            
        except Exception as e:
            console.print(f"\n[red]✗ R1 reasoner error: {e}[/red]")
            return True
    
    return False

def try_handle_reasoner_command(user_input: str) -> bool:
    """Handle /reasoner command to toggle between models."""
    if user_input.strip().lower() == "/reasoner":
        if model_context['current_model'] == DEFAULT_MODEL:
            model_context['current_model'] = REASONER_MODEL
            model_context['is_reasoner'] = True
            console.print(f"[green]✓ Switched to {REASONER_MODEL} model 🧠[/green]")
            console.print("[dim]All subsequent conversations will use the reasoner model.[/dim]")
        else:
            model_context['current_model'] = DEFAULT_MODEL
            model_context['is_reasoner'] = False
            console.print(f"[green]✓ Switched to {DEFAULT_MODEL} model 💬[/green]")
            console.print("[dim]All subsequent conversations will use the chat model.[/dim]")
        return True
    return False

def try_handle_markdown_command(user_input: str) -> bool:
    """Handle /markdown command to toggle markdown rendering."""
    if user_input.strip().lower() == "/markdown":
        from config import display_context
        current_state = display_context.get("enable_markdown_rendering", True)
        display_context["enable_markdown_rendering"] = not current_state
        
        if display_context["enable_markdown_rendering"]:
            console.print("[green]✓ Markdown rendering enabled 🎨[/green]")
            console.print("[dim]AI responses will be formatted as markdown when detected.[/dim]")
        else:
            console.print("[yellow]✓ Markdown rendering disabled 📝[/yellow]")
            console.print("[dim]AI responses will be displayed as plain text.[/dim]")
        return True
    return False

def try_handle_clear_command(user_input: str) -> bool:
    """Handle /clear command to clear screen."""
    if user_input.strip().lower() == "/clear":
        console.clear()
        return True
    return False

def try_handle_clear_context_command(user_input: str, conversation_history: List[Dict[str, Any]]) -> bool:
    """Handle /clear-context command to clear conversation history."""
    if user_input.strip().lower() == "/clear-context":
        if len(conversation_history) <= 1:
            console.print("[yellow]Context already empty (only system prompt).[/yellow]")
            return True
            
        file_contexts = sum(1 for msg in conversation_history if msg["role"] == "system" and "User added file" in msg["content"])
        total_messages = len(conversation_history) - 1
        
        console.print(f"[yellow]Current context: {total_messages} messages, {file_contexts} file contexts[/yellow]")
        
        confirm = prompt_session.prompt("🔵 Clear conversation context? This cannot be undone (y/n): ").strip().lower()
        if confirm in ["y", "yes"]:
            original_system_prompt = conversation_history[0]
            conversation_history[:] = [original_system_prompt]
            console.print("[green]✓ Conversation context cleared. Starting fresh![/green]")
            console.print("[green]  All file contexts and conversation history removed.[/green]")
        else:
            console.print("[yellow]Context clear cancelled.[/yellow]")
        return True
    return False

def try_handle_folder_command(user_input: str, conversation_history: List[Dict[str, Any]]) -> bool:
    """Handle /folder command to manage base directory."""
    if user_input.strip().lower().startswith("/folder"):
        folder_path = user_input[len("/folder"):].strip()
        if not folder_path:
            console.print(f"[yellow]Current base directory: '{config.base_dir}'[/yellow]")
            console.print("[yellow]Usage: /folder <path> or /folder reset[/yellow]")
            return True
        if folder_path.lower() == "reset":
            old_base = config.base_dir
            current_cwd = Path.cwd()
            config.base_dir = current_cwd
            console.print(f"[green]✓ Base directory reset from '{old_base}' to: '{config.base_dir}'[/green]")
            console.print(f"[green]  Synchronized with current working directory: '{current_cwd}'[/green]")
            
            # Add directory change to conversation context so the assistant knows
            dir_summary = get_directory_tree_summary(config.base_dir)
            conversation_history.append({
                "role": "system",
                "content": f"Working directory reset to: {config.base_dir}\n\nCurrent directory structure:\n\n{dir_summary}"
            })
            
            return True
        try:
            new_base = Path(folder_path).resolve()
            if not new_base.exists() or not new_base.is_dir():
                console.print(f"[red]✗ Path does not exist or is not a directory: '{folder_path}'[/red]")
                return True
            test_file = new_base / ".eng-git-test"
            try:
                test_file.touch()
                test_file.unlink()
            except PermissionError:
                console.print(f"[red]✗ No write permissions in directory: '{new_base}'[/red]")
                return True
            old_base = config.base_dir
            config.base_dir = new_base
            console.print(f"[green]✓ Base directory changed from '{old_base}' to: '{config.base_dir}'[/green]")
            console.print(f"[green]  All relative paths will now be resolved against this directory.[/green]")
            
            # Add directory change to conversation context so the assistant knows
            dir_summary = get_directory_tree_summary(config.base_dir)
            conversation_history.append({
                "role": "system",
                "content": f"Working directory changed to: {config.base_dir}\n\nNew directory structure:\n\n{dir_summary}"
            })
            
            return True
        except Exception as e:
            console.print(f"[red]✗ Error setting base directory: {e}[/red]")
            return True
    return False

def try_handle_exit_command(user_input: str) -> bool:
    """Handle /exit and /quit commands."""
    if user_input.strip().lower() in ("/exit", "/quit"):
        console.print("[bold blue]👋 Goodbye![/bold blue]")
        sys.exit(0)
    return False

def try_handle_context_command(user_input: str, conversation_history: List[Dict[str, Any]]) -> bool:
    """Handle /context command to show context usage statistics."""
    if user_input.strip().lower() == "/context":
        context_info = get_context_usage_info(conversation_history, model_context.get('current_model'))
        
        context_table = Table(title="📊 Context Usage Statistics", show_header=True, header_style="bold bright_blue")
        context_table.add_column("Metric", style="bright_cyan")
        context_table.add_column("Value", style="white")
        context_table.add_column("Status", style="white")
        
        context_table.add_row("Total Messages", str(context_info["total_messages"]), "📝")
        context_table.add_row("Estimated Tokens", f"{context_info['estimated_tokens']:,}", f"{context_info['token_usage_percent']:.1f}% of {context_info['max_tokens']:,}")
        context_table.add_row("File Contexts", str(context_info["file_contexts"]), f"Max: 5")
        
        if context_info["critical_limit"]:
            status_color = "red"
            status_text = "🔴 Critical - aggressive truncation active"
        elif context_info["approaching_limit"]:
            status_color = "yellow"
            status_text = "🟡 Warning - approaching limits"
        else:
            status_color = "green"
            status_text = "🟢 Healthy - plenty of space"
        
        context_table.add_row("Context Health", status_text, "")
        console.print(context_table)
        
        if context_info["token_breakdown"]:
            breakdown_table = Table(title="📋 Token Breakdown by Role", show_header=True, header_style="bold bright_blue", border_style="blue")
            breakdown_table.add_column("Role", style="bright_cyan")
            breakdown_table.add_column("Tokens", style="white")
            breakdown_table.add_column("Percentage", style="white")
            
            total_tokens = context_info["estimated_tokens"]
            for role, tokens in context_info["token_breakdown"].items():
                if tokens > 0:
                    percentage = (tokens / total_tokens * 100) if total_tokens > 0 else 0
                    breakdown_table.add_row(
                        role.capitalize(),
                        f"{tokens:,}",
                        f"{percentage:.1f}%"
                    )
            
            console.print(breakdown_table)
        
        if context_info["approaching_limit"]:
            console.print("\n[yellow]💡 Recommendations to manage context:[/yellow]")
            console.print("[yellow]  • Use /clear-context to start fresh[/yellow]")
            console.print("[yellow]  • Remove large files from context[/yellow]")
            console.print("[yellow]  • Work with smaller file sections[/yellow]")
        
        return True
    return False

def try_handle_help_command(user_input: str) -> bool:
    """Handle /help command to show available commands."""
    if user_input.strip().lower() == "/help":
        help_table = Table(title="📝 Available Commands", show_header=True, header_style="bold bright_blue")
        help_table.add_column("Command", style="bright_cyan")
        help_table.add_column("Description", style="white")
        
        # General commands
        help_table.add_row("/help", "Show this help")
        help_table.add_row("/r", "Call Reasoner model for one-off reasoning tasks")
        help_table.add_row("/reasoner", "Toggle between chat and reasoner models")
        help_table.add_row("/markdown", "Toggle markdown rendering for AI responses")
        help_table.add_row("/clear", "Clear screen")
        help_table.add_row("/clear-context", "Clear conversation context")
        help_table.add_row("/context", "Show context usage statistics")
        help_table.add_row("/os", "Show operating system information")
        help_table.add_row("/exit, /quit", "Exit application")
        
        # Directory & file management
        help_table.add_row("/folder", "Show current base directory")
        help_table.add_row("/folder <path>", "Set base directory for file operations")
        help_table.add_row("/folder reset", "Reset base directory to current working directory")
        help_table.add_row(f"{ADD_COMMAND_PREFIX.strip()} <path>", "Add file/dir to conversation context (supports fuzzy matching)")
        
        # Git workflow commands removed - agent can use bash commands for git operations
        
        console.print(help_table)
        
        # Show current model status
        current_model_name = "Reasoner 🧠" if model_context['is_reasoner'] else "Chat 💬"
        console.print(f"\n[dim]Current model: {current_model_name}[/dim]")
        
        # Show markdown rendering status
        from config import display_context
        markdown_status = "✓ Enabled" if display_context.get("enable_markdown_rendering", True) else "✗ Disabled"
        console.print(f"[dim]Markdown rendering: {markdown_status}[/dim]")
        
        # Show fuzzy matching status
        fuzzy_status = "✓ Available" if FUZZY_AVAILABLE else "✗ Not installed (pip install thefuzz python-levenshtein)"
        console.print(f"[dim]Fuzzy matching: {fuzzy_status}[/dim]")
        
        # Show OS and shell status
        available_shells = [shell for shell, available in os_info['shell_available'].items() if available]
        shell_status = ", ".join(available_shells) if available_shells else "None detected"
        console.print(f"[dim]OS: {os_info['system']} | Available shells: {shell_status}[/dim]")
        
        return True
    return False

def try_handle_os_command(user_input: str) -> bool:
    """Handle /os command to show operating system information."""
    if user_input.strip().lower() == "/os":
        os_table = Table(title="🖥️ Operating System Information", show_header=True, header_style="bold bright_blue")
        os_table.add_column("Property", style="bright_cyan")
        os_table.add_column("Value", style="white")
        
        # Basic OS info
        os_table.add_row("System", os_info['system'])
        os_table.add_row("Release", os_info['release'])
        os_table.add_row("Version", os_info['version'])
        os_table.add_row("Machine", os_info['machine'])
        if os_info['processor']:
            os_table.add_row("Processor", os_info['processor'])
        os_table.add_row("Python Version", os_info['python_version'])
        
        console.print(os_table)
        
        # Shell availability
        shell_table = Table(title="🐚 Shell Availability", show_header=True, header_style="bold bright_blue")
        shell_table.add_column("Shell", style="bright_cyan")
        shell_table.add_column("Status", style="white")
        
        for shell, available in os_info['shell_available'].items():
            status = "✓ Available" if available else "✗ Not available"
            shell_table.add_row(shell.capitalize(), status)
        
        console.print(shell_table)
        
        # Platform-specific recommendations
        if os_info['is_windows']:
            console.print("\n[yellow]💡 Windows detected:[/yellow]")
            console.print("[yellow]  • PowerShell commands are preferred[/yellow]")
            if os_info['shell_available']['bash']:
                console.print("[yellow]  • Bash is available (WSL or Git Bash)[/yellow]")
        elif os_info['is_mac']:
            console.print("\n[yellow]💡 macOS detected:[/yellow]")
            console.print("[yellow]  • Bash and zsh commands are preferred[/yellow]")
            console.print("[yellow]  • PowerShell Core may be available[/yellow]")
        elif os_info['is_linux']:
            console.print("\n[yellow]💡 Linux detected:[/yellow]")
            console.print("[yellow]  • Bash commands are preferred[/yellow]")
            console.print("[yellow]  • PowerShell Core may be available[/yellow]")
        
        return True
    return False

# Git add command removed - agent can use bash commands for git operations

# Git commit command removed - agent can use bash commands for git operations

# Git command handlers removed - agent can use bash commands for git operations

def try_handle_add_command(user_input: str, conversation_history: List[Dict[str, Any]]) -> bool:
    """Handle /add command with fuzzy file finding support."""
    if user_input.strip().lower().startswith(ADD_COMMAND_PREFIX):
        path_to_add = user_input[len(ADD_COMMAND_PREFIX):].strip()
        
        # 1. Try direct path first
        try:
            p = (config.base_dir / path_to_add).resolve()
            if p.exists():
                normalized_path = str(p)
            else:
                # This will raise an error if it doesn't exist, triggering the fuzzy search
                _ = p.resolve(strict=True) 
        except (FileNotFoundError, OSError):
            # 2. If direct path fails, try fuzzy finding
            console.print(f"[dim]Path '{path_to_add}' not found directly, attempting fuzzy search...[/dim]")
            fuzzy_match = find_best_matching_file(config.base_dir, path_to_add)

            if fuzzy_match:
                # Optional: Confirm with user for better UX
                relative_fuzzy = Path(fuzzy_match).relative_to(config.base_dir)
                confirm = prompt_session.prompt(f"🔵 Did you mean '[bright_cyan]{relative_fuzzy}[/bright_cyan]'? (Y/n): ", default="y").strip().lower()
                if confirm in ["y", "yes"]:
                    normalized_path = fuzzy_match
                else:
                    console.print("[yellow]Add command cancelled.[/yellow]")
                    return True
            else:
                console.print(f"[bold red]✗[/bold red] Path does not exist: '[bright_cyan]{path_to_add}[/bright_cyan]'")
                if FUZZY_AVAILABLE:
                    console.print("[dim]Tip: Try a partial filename (e.g., 'main.py' instead of exact path)[/dim]")
                return True
        
        # --- Process the found file/directory ---
        try:
            if Path(normalized_path).is_dir():
                add_directory_to_conversation(normalized_path, conversation_history)
            else:
                content = read_local_file(normalized_path)
                if add_file_context_smartly(conversation_history, normalized_path, content):
                    console.print(f"[bold blue]✓[/bold blue] Added file '[bright_cyan]{normalized_path}[/bright_cyan]' to conversation.\n")
                else:
                    console.print(f"[bold yellow]⚠[/bold yellow] File '[bright_cyan]{normalized_path}[/bright_cyan]' too large for context.\n")
        except (OSError, ValueError) as e:
            console.print(f"[bold red]✗[/bold red] Could not add path '[bright_cyan]{path_to_add}[/bright_cyan]': {e}\n")
        return True
    return False

# =============================================================================
# LLM TOOL HANDLER FUNCTIONS
# =============================================================================

def ensure_file_in_context(file_path: str, conversation_history: List[Dict[str, Any]]) -> bool:
    """
    Ensure a file is loaded in the conversation context.
    
    Args:
        file_path: Path to the file
        conversation_history: Conversation history to add to
        
    Returns:
        True if file was successfully added to context
    """
    try:
        normalized_path = normalize_path(file_path)
        content = read_local_file(normalized_path)
        marker = f"User added file '{normalized_path}'"
        if not any(msg["role"] == "system" and marker in msg["content"] for msg in conversation_history):
            return add_file_context_smartly(conversation_history, normalized_path, content)
        return True
    except (OSError, ValueError) as e:
        console.print(f"[red]✗ Error reading file for context '{file_path}': {e}[/red]")
        return False

# LLM git tool handlers removed - agent can use bash commands for git operations





def execute_function_call_dict(tool_call_dict: Dict[str, Any]) -> str:
    """
    Execute a function call from the LLM with enhanced fuzzy matching and security.
    
    Args:
        tool_call_dict: Dictionary containing function call information
        
    Returns:
        String result of the function execution
    """
    func_name = "unknown_function"
    try:
        func_name = tool_call_dict["function"]["name"]
        args = json.loads(tool_call_dict["function"]["arguments"])
        
        if func_name == "read_file":
            norm_path = normalize_path(args["file_path"])
            
            # Check file size before reading to prevent context overflow
            try:
                file_size = Path(norm_path).stat().st_size
                # Estimate tokens (roughly 4 chars per token)
                estimated_tokens = file_size // 4
                
                # Get model-specific context limit
                from config import get_max_tokens_for_model
                current_model = model_context.get('current_model', DEFAULT_MODEL)
                max_tokens = get_max_tokens_for_model(current_model)
                
                # Don't read files that would use more than 60% of context window
                max_file_tokens = int(max_tokens * 0.6)
                
                if estimated_tokens > max_file_tokens:
                    file_size_kb = file_size / 1024
                    return f"Error: File '{norm_path}' is too large ({file_size_kb:.1f}KB, ~{estimated_tokens} tokens) to read safely. Current model ({current_model}) has a context limit of {max_tokens} tokens. Maximum safe file size is ~{max_file_tokens} tokens ({(max_file_tokens * 4) / 1024:.1f}KB). Consider reading the file in smaller sections or using a different approach."
                    
            except OSError as e:
                return f"Error: Could not check file size for '{norm_path}': {e}"
            
            content = read_local_file(norm_path)
            return f"Content of file '{norm_path}':\n\n{content}"
            
        elif func_name == "read_multiple_files":
            response_data = {
                "files_read": {},
                "errors": {}
            }
            total_content_size = 0
            
            # Get model-specific context limit for multiple files
            from config import get_max_tokens_for_model
            current_model = model_context.get('current_model', DEFAULT_MODEL)
            max_tokens = get_max_tokens_for_model(current_model)
            # Use smaller percentage for multiple files to be safer
            max_total_tokens = int(max_tokens * 0.4)
            max_total_size = max_total_tokens * 4  # Convert tokens back to character estimate

            for fp in args["file_paths"]:
                try:
                    norm_path = normalize_path(fp)
                    
                    # Check individual file size first
                    try:
                        file_size = Path(norm_path).stat().st_size
                        if file_size > max_total_size // 2:  # Individual file shouldn't be more than half the total budget
                            response_data["errors"][norm_path] = f"File too large ({file_size/1024:.1f}KB) for multiple file read operation."
                            continue
                    except OSError:
                        pass  # Continue with normal reading if size check fails
                    
                    content = read_local_file(norm_path)

                    if total_content_size + len(content) > max_total_size:
                        response_data["errors"][norm_path] = f"Could not read file, as total content size would exceed the safety limit ({max_total_size/1024:.1f}KB for model {current_model})."
                        continue

                    response_data["files_read"][norm_path] = content
                    total_content_size += len(content)

                except (OSError, ValueError) as e:
                    # Use the original path in the error if normalization fails
                    error_key = str(config.base_dir / fp)
                    response_data["errors"][error_key] = str(e)

            # Return a JSON string, which is much easier for the LLM to parse reliably
            return json.dumps(response_data, indent=2)
            
        elif func_name == "create_file": 
            create_file(args["file_path"], args["content"])
            return f"File '{args['file_path']}' created/updated."
            
        elif func_name == "create_multiple_files":
            created: List[str] = []
            errors: List[str] = []
            for f_info in args["files"]:
                try: 
                    create_file(f_info["path"], f_info["content"])
                    created.append(f_info["path"])
                except Exception as e: 
                    errors.append(f"Error creating {f_info.get('path','?path')}: {e}")
            res_parts = []
            if created: 
                res_parts.append(f"Created/updated {len(created)} files: {', '.join(created)}")
            if errors: 
                res_parts.append(f"Errors: {'; '.join(errors)}")
            return ". ".join(res_parts) if res_parts else "No files processed."
            
        elif func_name == "edit_file":
            fp = args["file_path"]
            original = args["original_snippet"]
            new = args["new_snippet"]
            
            # Normalize the path relative to base_dir
            norm_fp = normalize_path(fp)
            
            # Check if file exists before editing
            if not Path(norm_fp).exists():
                return f"Error: File '{norm_fp}' does not exist."
            
            try:
                # Read the file before editing to show the change
                content_before = read_local_file(norm_fp)
                
                # Pre-process snippets for better matching
                original_clean = original.strip()
                new_clean = new.strip()
                
                # Check for common issues with model-generated snippets
                if not original_clean:
                    return f"Error: Original snippet is empty for '{norm_fp}'"
                
                original_lines = original_clean.split('\n')
                if len(original_lines) > 50:
                    return f"Error: Original snippet too large ({len(original_lines)} lines) for safe editing in '{norm_fp}'"
                
                # Log the edit attempt for debugging
                console.print(f"[dim]Attempting to edit {Path(norm_fp).name}:[/dim]")
                console.print(f"[dim]  Original snippet: {len(original_clean)} chars, {len(original_clean.split(chr(10)))} lines[/dim]")
                console.print(f"[dim]  New snippet: {len(new_clean)} chars, {len(new_clean.split(chr(10)))} lines[/dim]")
                
                # Use improved fuzzy edit function
                apply_fuzzy_diff_edit(norm_fp, original_clean, new_clean)
                
                # Verify the edit was successful
                content_after = read_local_file(norm_fp)
                
                if content_before == content_after:
                    return f"No changes made to '{norm_fp}'. The original snippet was not found or the content is already as specified."
                else:
                    # Show a summary of what changed
                    lines_before = len(content_before.split('\n'))
                    lines_after = len(content_after.split('\n'))
                    line_diff = lines_after - lines_before
                    
                    if line_diff != 0:
                        return f"Successfully edited '{norm_fp}'. File now has {lines_after} lines ({line_diff:+d} lines changed)."
                    else:
                        return f"Successfully edited '{norm_fp}'. Content updated with same line count ({lines_after} lines)."
                        
            except ValueError as ve:
                # This is expected for fuzzy matching failures
                error_msg = str(ve)
                console.print(f"[yellow]Edit failed for '{norm_fp}': {error_msg}[/yellow]")
                
                # Provide helpful suggestions
                suggestions = []
                if "score" in error_msg and "below threshold" in error_msg:
                    suggestions.append("Try using a smaller, more specific code snippet")
                    suggestions.append("Check for extra whitespace or formatting differences")
                elif "too ambiguous" in error_msg:
                    suggestions.append("Include more context to make the snippet unique")
                    suggestions.append("Add surrounding lines or unique variable names")
                elif "not found" in error_msg:
                    suggestions.append("Check if the file content has changed since last read")
                    suggestions.append("Verify the exact formatting matches the file")
                
                suggestion_text = "Suggestions: " + "; ".join(suggestions) if suggestions else ""
                return f"Edit failed for '{norm_fp}': {error_msg}. {suggestion_text}"
                
            except Exception as e:
                console.print_exception()
                return f"Unexpected error during edit_file for '{norm_fp}': {e}"
                
        # Git tool functions removed - agent can use bash commands for git operations
        elif func_name == "run_powershell":
            command = args["command"]
            
            # SECURITY GATE
            if security_context["require_powershell_confirmation"]:
                console.print(Panel(
                    f"The assistant wants to run this PowerShell command:\n\n[bold yellow]{command}[/bold yellow]", 
                    title="🚨 Security Confirmation Required", 
                    border_style="red"
                ))
                confirm = prompt_session.prompt("🔵 Do you want to allow this command to run? (y/N): ", default="n").strip().lower()
                
                if confirm not in ["y", "yes"]:
                    console.print("[red]Execution denied by user.[/red]")
                    return "PowerShell command execution was denied by the user."
            
            output, error = run_powershell_command(command, config.base_dir)
            if error:
                return f"PowerShell Error:\n{error}"
            
            # Handle empty output more clearly for the model
            if not output.strip():
                return f"PowerShell command executed successfully. No output produced (this is normal for commands like Remove-Item, New-Item, etc.)."
            else:
                return f"PowerShell Output:\n{output}"
        elif func_name == "run_bash":
            command = args["command"]
            
            # SECURITY GATE
            if security_context["require_bash_confirmation"]:
                console.print(Panel(
                    f"The assistant wants to run this bash command:\n\n[bold yellow]{command}[/bold yellow]", 
                    title="🚨 Security Confirmation Required", 
                    border_style="red"
                ))
                confirm = prompt_session.prompt("🔵 Do you want to allow this command to run? (y/N): ", default="n").strip().lower()
                
                if confirm not in ["y", "yes"]:
                    console.print("[red]Execution denied by user.[/red]")
                    return "Bash command execution was denied by the user."
            
            output, error = run_bash_command(command, config.base_dir)
            if error:
                return f"Bash Error:\n{error}"
            
            # Handle empty output more clearly for the model
            if not output.strip():
                return f"Bash command executed successfully. No output produced (this is normal for commands like rm, mkdir, etc.)."
            else:
                return f"Bash Output:\n{output}"
        else: 
            return f"Unknown LLM function: {func_name}"
            
    except json.JSONDecodeError as e: 
        console.print(f"[red]JSON Decode Error for {func_name}: {e}\nArgs: {tool_call_dict.get('function',{}).get('arguments','')}[/red]")
        return f"Error: Invalid JSON args for {func_name}."
    except KeyError as e: 
        console.print(f"[red]KeyError in {func_name}: Missing key {e}[/red]")
        return f"Error: Missing param for {func_name} (KeyError: {e})."
    except Exception as e: 
        console.print(f"[red]Unexpected Error in LLM func '{func_name}':[/red]")
        console.print_exception()
        return f"Unexpected error in {func_name}: {e}"

# =============================================================================
# MAIN LOOP & ENTRY POINT
# =============================================================================

def initialize_application() -> None:
    """Initialize the application."""
    # Detect available shells
    detect_available_shells()
    # Git repository detection removed - agent can use bash commands for git operations

def main_loop() -> None:
    """Main application loop."""
    # Initialize conversation history
    conversation_history: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    # Add initial context
    dir_summary = get_directory_tree_summary(config.base_dir)
    conversation_history.append({
        "role": "system",
        "content": f"Project directory structure at startup:\n\n{dir_summary}"
    })
    
    # Add OS and shell info
    shell_status = ", ".join([f"{shell}({'✓' if available else '✗'})" 
                             for shell, available in os_info['shell_available'].items()])
    conversation_history.append({
        "role": "system",
        "content": f"Runtime environment: {os_info['system']} {os_info['release']}, "
                  f"Python {os_info['python_version']}, Shells: {shell_status}"
    })

    while True:
        try:
            prompt_indicator = get_prompt_indicator(conversation_history, model_context['current_model'])
            user_input = prompt_session.prompt(f"{prompt_indicator} You: ")
            
            if not user_input.strip(): 
                continue

            # Handle commands
            if try_handle_add_command(user_input, conversation_history): continue
            # Git command handling removed - agent can use bash commands for git operations
            if try_handle_r1_command(user_input, conversation_history): continue
            if try_handle_reasoner_command(user_input): continue
            if try_handle_markdown_command(user_input): continue
            if try_handle_clear_command(user_input): continue
            if try_handle_clear_context_command(user_input, conversation_history): continue
            if try_handle_context_command(user_input, conversation_history): continue
            if try_handle_folder_command(user_input, conversation_history): continue
            if try_handle_os_command(user_input): continue
            if try_handle_exit_command(user_input): continue
            if try_handle_help_command(user_input): continue
            
            # Add user message to conversation
            conversation_history.append({"role": "user", "content": user_input})
            
            # Determine which model to use
            current_model = model_context['current_model']
            model_name = "GPT OSS" if current_model == DEFAULT_MODEL else "Qwen"
            
            # Check context usage and force truncation if needed
            context_info = get_context_usage_info(conversation_history, current_model)
            
            # Always truncate if we're over the limit (not just 95%)
            if context_info["estimated_tokens"] > context_info["max_tokens"] or context_info["token_usage_percent"] > 90:
                console.print(f"[red]🚨 Context exceeded ({context_info['estimated_tokens']} > {context_info['max_tokens']} tokens). Force truncating...[/red]")
                conversation_history = smart_truncate_history(conversation_history, model_name=current_model)
                context_info = get_context_usage_info(conversation_history, current_model)  # Recalculate after truncation
                console.print(f"[green]✓ Context truncated to {context_info['estimated_tokens']} tokens ({context_info['token_usage_percent']:.1f}% of limit)[/green]")
            elif context_info["critical_limit"] and len(conversation_history) % 10 == 0:
                console.print(f"[red]⚠ Context critical: {context_info['token_usage_percent']:.1f}% used. Consider /clear-context or /context for details.[/red]")
            elif context_info["approaching_limit"] and len(conversation_history) % 20 == 0:
                console.print(f"[yellow]⚠ Context high: {context_info['token_usage_percent']:.1f}% used. Use /context for details.[/yellow]")

            # Final safety check before API call
            final_context_info = get_context_usage_info(conversation_history, current_model)
            if final_context_info["estimated_tokens"] > final_context_info["max_tokens"]:
                console.print(f"[red]🚨 Final safety check failed: {final_context_info['estimated_tokens']} > {final_context_info['max_tokens']} tokens. Emergency truncation...[/red]")
                conversation_history = smart_truncate_history(conversation_history, model_name=current_model)
                final_context_info = get_context_usage_info(conversation_history, current_model)
                console.print(f"[green]✓ Emergency truncation complete: {final_context_info['estimated_tokens']} tokens[/green]")

            # Sanitize tools for Cerebras/other non-strictly-OpenAI models
            sanitized_tools = []
            for tool in tools:
                clean_tool = tool.copy()
                func = clean_tool["function"].copy()
                func.pop("strict", None)  # Remove unsupported field
                clean_tool["function"] = func
                sanitized_tools.append(clean_tool)

            # Make API call
            with console.status(f"[bold yellow]{model_name} is thinking...[/bold yellow]", spinner="dots"):
                try:
                    kwargs = {
                        "model": current_model,
                        "messages": conversation_history,
                        "tools": sanitized_tools, # Use sanitized tools
                        "tool_choice": "auto",
                        "stream": False,
                        "max_completion_tokens": 5000, # Reduced from 10000
                        "temperature": 0.7,
                        "top_p": 1
                    }
                    response = client.chat.completions.create(**kwargs)
                except Exception as e:
                    console.print(f"[red]✗ API error with {current_model}: {e}[/red]")
                    console.print(f"[red]Request had {len(sanitized_tools)} tools, {len(conversation_history)} messages[/red]")
                    continue

            # Process non-streaming response
            message = response.choices[0].message
            full_response_content = message.content or ""
            accumulated_tool_calls: List[Dict[str, Any]] = []
            
            
            # Extract tool calls if present
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    accumulated_tool_calls.append({
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })

            # Validate and check tool calls first
            valid_tool_calls = validate_tool_calls(accumulated_tool_calls)
            
            # Display the response content only if there's content or valid tool calls
            if full_response_content or valid_tool_calls:
                console.print(f"[bold bright_blue]🤖 {model_name}:[/bold bright_blue]")
                if full_response_content:
                    # Strip <think> and </think> tags from the content
                    clean_content = full_response_content.replace("<think>", "").replace("</think>", "")
                    enhanced_content = enhance_terminal_output(clean_content)
                    render_markdown_response(enhanced_content)
                elif valid_tool_calls:
                    console.print("[dim]Processing tool calls...[/dim]", style="bright_blue")

            # Always add assistant message to maintain conversation flow
            assistant_message: Dict[str, Any] = {"role": "assistant"}
            assistant_message["content"] = full_response_content

            # Add tool calls if any exist
            if valid_tool_calls:
                assistant_message["tool_calls"] = valid_tool_calls
            
            # Always add the assistant message
            conversation_history.append(assistant_message)

            # Execute tool calls and allow assistant to continue naturally
            if valid_tool_calls:
                # Execute all tool calls first
                for tool_call_to_exec in valid_tool_calls: 
                    console.print(Panel(
                        f"[bold blue]Calling:[/bold blue] {tool_call_to_exec['function']['name']}\n"
                        f"[bold blue]Args:[/bold blue] {tool_call_to_exec['function']['arguments']}",
                        title="🛠️ Function Call", border_style="yellow", expand=False
                    ))
                    tool_output = execute_function_call_dict(tool_call_to_exec) 
                    console.print(Panel(tool_output, title=f"↪️ Output of {tool_call_to_exec['function']['name']}", border_style="green", expand=False))
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call_to_exec["id"],
                        "name": tool_call_to_exec["function"]["name"],
                        "content": tool_output 
                    })
                
                # Now let the assistant continue with the tool results
                max_continuation_rounds = 15
                current_round = 0
                
                while current_round < max_continuation_rounds:
                    current_round += 1
                    
                    with console.status(f"[bold yellow]{model_name} is processing results...[/bold yellow]", spinner="dots"):
                        try:
                            # Use the same sanitized tools for continuation calls
                            continue_kwargs = {
                                "model": current_model, 
                                "messages": conversation_history,
                                "tools": sanitized_tools, # Use sanitized tools
                                "tool_choice": "auto",
                                "stream": False,
                                "max_completion_tokens": 5000,
                                "temperature": 0.7,
                                "top_p": 1
                            }
                            continue_response = client.chat.completions.create(**continue_kwargs)
                        except Exception as e:
                            console.print(f"[red]✗ API error during continuation with {current_model}: {e}[/red]")
                            console.print(f"[red]Request had {len(sanitized_tools)} tools, {len(conversation_history)} messages[/red]")
                            # Break out of the continuation loop on error
                            break

                    # Process the continuation response
                    continue_message = continue_response.choices[0].message
                    continuation_content = continue_message.content or ""
                    continuation_tool_calls: List[Dict[str, Any]] = []
                    
                    # Extract tool calls if present
                    if continue_message.tool_calls:
                        for tool_call in continue_message.tool_calls:
                            continuation_tool_calls.append({
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            })

                    # Check if there are more tool calls to execute before displaying anything
                    valid_continuation_tools = validate_tool_calls(continuation_tool_calls)
                    
                    # Only display response if there's content or valid tool calls
                    if continuation_content or valid_continuation_tools:
                        console.print(f"[bold bright_blue]🤖 {model_name}:[/bold bright_blue]")
                        if continuation_content:
                            clean_content = continuation_content.replace("<think>", "").replace("</think>", "")
                            enhanced_content = enhance_terminal_output(clean_content)
                            render_markdown_response(enhanced_content)
                        elif valid_continuation_tools:
                            console.print("[dim]Continuing with tool calls...[/dim]", style="bright_blue")
                    
                    # Add the continuation response to conversation history
                    continuation_message_obj: Dict[str, Any] = {"role": "assistant", "content": continuation_content}
                    
                    # Execute tool calls if any exist
                    if valid_continuation_tools:
                        continuation_message_obj["tool_calls"] = valid_continuation_tools
                        conversation_history.append(continuation_message_obj)
                        
                        # Execute the additional tool calls
                        for tool_call_to_exec in valid_continuation_tools:
                            console.print(Panel(
                                f"[bold blue]Calling:[/bold blue] {tool_call_to_exec['function']['name']}\n"
                                f"[bold blue]Args:[/bold blue] {tool_call_to_exec['function']['arguments']}",
                                title="🛠️ Function Call", border_style="yellow", expand=False
                            ))
                            tool_output = execute_function_call_dict(tool_call_to_exec)
                            console.print(Panel(tool_output, title=f"↪️ Output of {tool_call_to_exec['function']['name']}", border_style="green", expand=False))
                            conversation_history.append({
                                "role": "tool",
                                "tool_call_id": tool_call_to_exec["id"],
                                "name": tool_call_to_exec["function"]["name"],
                                "content": tool_output
                            })
                        
                        # Continue the loop to let assistant process these new results
                        continue
                    else:
                        # No more tool calls, add the final response and break
                        conversation_history.append(continuation_message_obj)
                        break
                
                # If we hit the max rounds, warn about it
                if current_round >= max_continuation_rounds:
                    console.print(f"[yellow]⚠ Reached maximum continuation rounds ({max_continuation_rounds}). Conversation continues.[/yellow]")
            
            # Smart truncation that preserves tool call sequences
            conversation_history = smart_truncate_history(conversation_history, model_name=current_model)

        except KeyboardInterrupt: 
            console.print("\n[yellow]⚠ Interrupted. Ctrl+D or /exit to quit.[/yellow]")
        except EOFError: 
            console.print("[blue]👋 Goodbye! (EOF)[/blue]")
            sys.exit(0)
        except Exception as e:
            console.print(f"\n[red]✗ Unexpected error in main loop:[/red]")
            console.print_exception(width=None, extra_lines=1, show_locals=True)

def main() -> None:
    """Application entry point."""
    console.print(Panel.fit(
        "[bold bright_blue] Cerebras Assistant - Enhanced Edition[/bold bright_blue]\n"
        "[dim]✨ Now with fuzzy matching for files and cross-platform shell support![/dim]\n"
        "[dim]Type /help for commands. Ctrl+C to interrupt, Ctrl+D or /exit to quit.[/dim]",
        border_style="bright_blue"
    ))

    # Show fuzzy matching status on startup
    if FUZZY_AVAILABLE:
        console.print("[green]✓ Fuzzy matching enabled for intelligent file finding and code editing[/green]")
    else:
        console.print("[yellow]⚠ Fuzzy matching disabled. Install with: pip install thefuzz python-levenshtein[/yellow]")

    # Initialize application first (detects git repo and shells)
    initialize_application()
    
    # Show detected shells
    available_shells = [shell for shell, available in os_info['shell_available'].items() if available]
    if available_shells:
        console.print(f"[green]✓ Detected shells: {', '.join(available_shells)}[/green]")
    else:
        console.print("[yellow]⚠ No supported shells detected[/yellow]")
    
    # Start the main loop
    main_loop()

if __name__ == "__main__":
    main()