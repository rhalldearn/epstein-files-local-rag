# Quick Start Guide

Get the Epstein Files chatbot running in 3 steps.

## Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA support (RTX 4070 or similar)
- ~3GB free disk space

## Step 1: Install Dependencies

### Using UV (Recommended)

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install llama-cpp-python with CUDA support
CMAKE_ARGS="-DLLAMA_CUDA=on" uv pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Using pip

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-chatbot.txt

# Install llama-cpp-python with CUDA support
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

## Step 2: Test Setup

Verify everything is installed correctly:

```bash
python -m scripts.test_setup
```

This will check:
- All required Python modules are installed
- PDF files are present
- Project structure is correct

## Step 3: Initialize the Chatbot

Process documents and download the model:

```bash
python -m scripts.initialize
```

This will:
- Process all 525 PDF files (~5 minutes)
- Download Llama 3.2 3B model (~2GB)
- Build the vector search index

## Step 4: Start Chatting

```bash
python -m src.chatbot
```

Or use the convenience script:

```bash
./run_chatbot.sh
```

## Example Usage

```
Your question: Who is mentioned in these files?

Answer:
[AI-generated answer based on the documents]

Sources:
Source 1: document.pdf (page 10, dataset: set_1)
"Relevant excerpt..."

Your question: /help
[Shows available commands]

Your question: /quit
Goodbye!
```

## Commands

- `/help` - Show available commands
- `/quit` - Exit chatbot
- `/reset` - Clear conversation history
- `/sources` - Show sources from last answer
- `/info` - Display system information

## Troubleshooting

### CUDA not detected

```bash
# Check CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall llama-cpp-python with CUDA
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Out of memory

Reduce GPU layers in `src/llm.py`:

```python
self.n_gpu_layers = 20  # Instead of -1 (all layers)
```

### Slow performance

Check that GPU is being used:

```
Your question: /info
```

Look for "GPU Layers: -1" in the output.

## Full Documentation

See [README_CHATBOT.md](README_CHATBOT.md) for complete documentation.
