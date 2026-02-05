# Epstein Files RAG Chatbot

A CLI-based RAG (Retrieval-Augmented Generation) chatbot that answers questions about the Epstein Files using a local Llama 3.2 3B model running on RTX 4070.

## Features

- **Local AI**: Runs entirely offline using Llama 3.2 3B (quantized)
- **Fast Search**: Semantic search across 525 PDFs using ChromaDB
- **Source Citations**: Every answer includes references to source documents
- **GPU Accelerated**: Full GPU offload for fast inference on RTX 4070
- **Interactive CLI**: Rich terminal interface with helpful commands

## System Requirements

- **GPU**: NVIDIA RTX 4070 (or similar with 12GB+ VRAM)
- **RAM**: 8GB minimum
- **Disk**: ~3GB for model and vector database
- **Python**: 3.11+
- **CUDA**: CUDA-compatible PyTorch installation

## Installation

### 1. Install UV (Python Package Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies

The project uses UV for dependency management. Install all dependencies:

```bash
uv sync
```

For llama-cpp-python with CUDA support, you may need to reinstall it with CUDA flags:

```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" uv pip install llama-cpp-python --force-reinstall --no-cache-dir
```

## Setup

### 1. Initialize the Chatbot

Run the initialization script to process documents, download the model, and build the vector index:

```bash
uv run python -m scripts.initialize
```

This will:
- Extract text from all 525 PDFs
- Create ~5,000-10,000 searchable chunks
- Download Llama 3.2 3B model (~2GB)
- Build the ChromaDB vector index

**Note**: First-time initialization takes 5-10 minutes.

### 2. Start the Chatbot

Once initialized, start the interactive chatbot:

```bash
uv run python -m src.chatbot
```

## Usage

### Basic Questions

Simply type your question:

```
Your question: Who is mentioned in these files?
```

The chatbot will:
1. Search for relevant document chunks
2. Generate an answer using Llama 3.2 3B
3. Display the answer with source citations

### Available Commands

- `/help` - Show available commands
- `/quit` - Exit the chatbot
- `/reset` - Clear conversation history
- `/sources` - Show detailed sources from last answer
- `/info` - Display system information and statistics

### Example Session

```
Your question: What are the key allegations?

Answer:
Based on the provided documents, the key allegations include...
[Sources cited automatically]

Sources:
Source 1: document_name.pdf (page 42, dataset: set_1)
"Relevant excerpt from the document..."

Your question: /sources
[Shows detailed information about all sources]

Your question: /quit
Goodbye!
```

## Architecture

### Document Processing
- **Input**: 525 PDFs across 12 datasets
- **Chunking**: 512 tokens per chunk, 50 token overlap
- **Output**: JSON file with chunks and metadata

### Vector Store
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Database**: ChromaDB (persistent, local)
- **Search**: Semantic similarity with top-k retrieval

### Language Model
- **Model**: Llama 3.2 3B Instruct (Q4_K_M quantization)
- **Backend**: llama-cpp-python with CUDA
- **Context**: 4096 tokens
- **GPU**: Full offload (-1 layers) for RTX 4070

### RAG Pipeline
1. User asks question
2. Generate question embedding
3. Search ChromaDB for top 5 relevant chunks
4. Construct prompt with context + question
5. Generate answer with Llama 3.2 3B
6. Display answer with source citations

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── document_processor.py   # PDF extraction and chunking
│   ├── vector_store.py         # ChromaDB and embeddings
│   ├── llm.py                  # Llama model wrapper
│   ├── rag.py                  # RAG orchestration
│   └── chatbot.py              # CLI interface
├── scripts/
│   └── initialize.py           # Setup script
├── models/                     # Downloaded models
├── chroma_db/                  # Vector database
├── processed_chunks.json       # Cached document chunks
├── pyproject.toml              # Project dependencies
└── README_CHATBOT.md           # This file
```

## Configuration

### Chunk Size
Default: 512 tokens with 50 token overlap

Modify in `scripts/initialize.py`:
```python
processor = DocumentProcessor(chunk_size=512, overlap=50)
```

### Retrieval Count
Default: Top 5 most relevant chunks

Modify in `src/chatbot.py`:
```python
answer, sources = chatbot.answer_question(question, top_k=5)
```

### Model Parameters
Modify in `src/llm.py`:
- `n_ctx`: Context window size (default: 4096)
- `n_gpu_layers`: GPU offload (-1 = all layers)
- `temperature`: Sampling temperature (default: 0.7)

## Troubleshooting

### CUDA/GPU Issues

If the model doesn't use GPU:

1. Check CUDA installation:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

2. Reinstall llama-cpp-python with CUDA:
```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" uv pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Memory Issues

If you run out of VRAM:

1. Reduce GPU layers in `src/llm.py`:
```python
self.n_gpu_layers = 20  # Instead of -1
```

2. Use a smaller quantization (e.g., Q3_K_M instead of Q4_K_M)

### Slow Performance

- Ensure GPU is being used (check `/info` command)
- Reduce chunk count with smaller top_k
- Use a faster embedding model

## Performance

On RTX 4070 with full GPU offload:
- **Initialization**: ~5-10 minutes (one-time)
- **Search**: <1 second
- **Generation**: ~2-5 seconds per response
- **Total Response Time**: ~3-6 seconds

## Rebuilding the Index

To rebuild the vector index from scratch:

```bash
uv run python -m scripts.initialize
```

Choose "yes" when prompted to reprocess documents or rebuild the index.

## Credits

- **Model**: Meta's Llama 3.2 3B
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector DB**: ChromaDB
- **PDF Processing**: PyMuPDF
- **CLI**: Rich library
