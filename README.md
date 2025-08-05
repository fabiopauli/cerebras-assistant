# Cerebras AI Code Assistant

An intelligent AI-powered development environment that combines conversational AI with powerful development tools, Git integration, and file operations featuring advanced fuzzy matching capabilities.

## Overview

Cerebras AI Code Assistant is an enhanced command-line interface built on top of the Cerebras AI API that serves as your intelligent coding companion. It provides seamless integration between natural language conversations and practical development tasks, making coding more intuitive and productive.

## Key Features

### 🤖 **AI-Powered Development**
- **Conversational Interface**: Interactive chat with advanced Cerebras AI models
- **Dual Model Support**: Toggle between chat model (`gpt-oss-120b`) and reasoning model (`qwen-3-235b-a22b-instruct-2507`) (Adjust config.json to change models)
- **Function Calling**: AI can automatically execute tools and operations
- **Streaming Responses**: Real-time AI response with rich formatting

### 📁 **Intelligent File Operations**
- **Fuzzy File Matching**: Find files even with typos or partial paths
- **Smart Code Editing**: Precise code modifications with fuzzy matching for snippet replacement
- **Batch Operations**: Read and create multiple files efficiently
- **Context-Aware**: Maintains file context across conversations

### 🔧 **Seamless Git Integration**
- **Repository Management**: Initialize, status checking, branch operations
- **Smart Staging**: Automatic or manual file staging with intelligent commit handling
- **Branch Workflow**: Create and switch between Git branches effortlessly
- **Status Monitoring**: Rich display of Git status with detailed file change tracking

### 💬 **Advanced Context Management**
- **Token-Based Estimation**: Intelligent conversation history management
- **Smart Truncation**: Preserves important context while staying within limits
- **File Context Tracking**: Manages multiple files in conversation context
- **Usage Monitoring**: Real-time context usage statistics and warnings

### 🛡️ **Security & Safety**
- **Command Confirmation**: Security prompts for potentially dangerous operations
- **Path Validation**: Robust file path sanitization and validation
- **Size Limits**: Configurable limits for file operations and content
- **Exclusion Patterns**: Smart filtering of system and temporary files

## Installation

### Prerequisites
- Python 3.8+
- Cerebras AI API key

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/fabiopauli/cerebras-assistant.git
   cd cerebras
   ```

2. **Install required dependencies**:
   Using uv (recommended - faster)
   ```bash
   uv venv
   uv pip install -r requirements.txt
   uv run main.py
   ```
   
   ```bash
   pip install -r requirements.txt
   ```

3. **Install optional dependencies for enhanced fuzzy matching**:
   ```bash
   pip install thefuzz python-levenshtein
   ```

4. **Set up environment variables**:
   Create a `.env` file or set the environment variable:
   
   ```bash
   # Create .env file
   echo "CEREBRAS_API_KEY=your-cerebras-api-key-here" > .env
   ```
   Or
   ```bash
   export CEREBRAS_API_KEY="your-cerebras-api-key-here"
   ```

5. **Run the assistant**:
   ```bash
   python main.py
   ```
   

## Usage

### Command-Line Interface

The assistant supports both natural language conversation and special commands:

#### **Special Commands**
- `/help` - Show all available commands and their descriptions
- `/add <path>` - Add files or directories to conversation context (supports fuzzy matching)
- `/git add . or <files>` - Stage all changes
- `/git commit [message]` - Stage all changes and commit with optional message
- `/git branch <name>` - Create and switch to a new Git branch
- `/git init` - Initialize a new Git repository
- `/git status` - Display current Git repository status
- `/reasoner` - Toggle between chat and reasoning models
- `/r` - Make a one-off call to the reasoning model
- `/clear` - Clear the console screen
- `/clear-context` - Reset conversation context
- `/context` - Show current context usage statistics
- `/folder [path]` - Set or display the current base directory
- `/exit` or `/quit` - Exit the application

#### **Example Workflows**

**Adding files to context:**
```bash
/add src/main.py
/add src/  # Add entire directory
/add mai.py  # Fuzzy matching will find main.py
```

**Git workflow:**
```bash
/git init
/git add .
/git commit "Initial commit with source files"
/git branch feature/new-feature
```

**File operations through conversation:**
```
User: "Create a new Python function that calculates Fibonacci numbers"
Assistant: [Creates the file with the function]

User: "Now modify the function to use memoization"
Assistant: [Edits the existing function using fuzzy matching]
```

### AI Tool Integration

The assistant can automatically execute these operations:

1. **File Operations**: `read_file`, `create_file`, `edit_file`, `read_multiple_files`
2. **Git Operations**: `git_init`, `git_add`, `git_commit`, `git_create_branch`, `git_status`
3. **System Operations**: `run_powershell` (with security confirmation)

## Configuration

### Fuzzy Matching Thresholds
- **File Path Matching**: 80% similarity minimum
- **Code Edit Matching**: 85% similarity minimum

### Context Limits
- **Maximum History**: 50 messages
- **Context Files**: 5 files maximum
- **Token Estimation**: ~66,000 tokens context window

## Project Structure

```bash
cerebras/
├── main.py             # Main application script
├── config.py           # Configuration and constants
├── utils.py            # Utility functions
├── system_prompt.txt   # System prompt configuration
├── README.md           # This documentation
├── pyproject.toml      # Project configuration
├── uv.lock             # UV lock file
└── .env                # Environment variables (create this)
```

## Advanced Features

### Fuzzy Matching
The assistant uses advanced fuzzy matching for:
- **File Path Resolution**: Find files even with typos (`mai.py` → `main.py`)
- **Code Snippet Matching**: Edit code even if the exact snippet has minor differences
- **Directory Navigation**: Smart directory and file discovery

### Context Management
- **Intelligent Truncation**: Automatically manages conversation history
- **File Priority**: Keeps most recently accessed files in context
- **Token Estimation**: Provides real-time context usage feedback
- **Memory Optimization**: Efficient handling of large codebases

### Security Features
- **Command Validation**: Confirms potentially dangerous operations
- **Path Sanitization**: Prevents directory traversal attacks
- **File Type Filtering**: Excludes binary and system files automatically
- **Size Limitations**: Prevents memory exhaustion from large files

## Dependencies

### Required
- `cerebras-cloud-sdk` - Cerebras AI API client
- `rich` - Enhanced console output and formatting
- `prompt_toolkit` - Interactive command-line interface
- `pydantic` - Data validation and settings management
- `python-dotenv` - Environment variable management

### Optional
- `thefuzz` - Fuzzy string matching capabilities
- `python-levenshtein` - Performance optimization for fuzzy matching

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

If you encounter any issues or have questions, please:
1. Check the `/help` command within the application
2. Review this documentation
3. Open an issue on the project repository

---

**Powered by Cerebras AI Models**