# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local RAG chatbot for querying the 525 DOJ Epstein Files PDF documents using Llama 3.2 3B and ChromaDB vector storage.

**Prerequisites:** PDF files must be downloaded separately using [Surebob's Epstein Files Downloader](https://github.com/Surebob/epstein-files-downloader)

## Critical Known Issue: Text Trapped in Images

**The PDFs contain significant text embedded in images that isn't being extracted.** Analysis shows:
- 20% of chunks have < 50 characters (poor extraction)
- Most PDFs have 1+ images per page containing text
- Standard PyMuPDF `get_text()` ignores image content

**Solution: Use OCR** - See `docs/OCR_GUIDE.md` for complete instructions.

## Common Commands

### Initial Setup

```bash
# Install all dependencies with CUDA support
./install.sh

# Standard initialization (fast, but misses text in images)
python -m scripts.initialize

# OR: Initialize with OCR (slower, but extracts text from images)
pip install pytesseract pillow
sudo apt-get install tesseract-ocr  # Linux
python -m scripts.initialize_with_ocr

# Verify installation
python -m scripts.test_setup

# Test OCR improvement on sample PDFs
./tools/test_ocr_extraction.py
```

### Running the Chatbot

```bash
# Start the chatbot interface
./run_chatbot.sh
# or
python -m src.chatbot
```

### Development Workflow

```bash
# Reprocess PDFs with different chunking parameters
python -m scripts.initialize  # Answer 'y' to reprocess

# Test specific components
python -c "from pathlib import Path; from src.vector_store import VectorStore; vs = VectorStore(Path('chroma_db')); vs.initialize(); print(vs.get_stats())"

# Check GPU usage while chatbot runs
watch -n 1 nvidia-smi
```

## Architecture

### RAG Pipeline Data Flow

```
User Question
    ↓
[chatbot.py] - CLI interface, conversation management
    ↓
[rag.py] - RAGChatbot orchestrates the pipeline
    ↓
[vector_store.py] - Semantic search via ChromaDB
    ↓ (returns top-k chunks)
[llm.py] - Llama 3.2 3B generates answer from context
    ↓
[chatbot.py] - Display answer with source citations
```

### Core Module Responsibilities

**document_processor.py** (Original - limited extraction)
- Extracts text from PDFs using PyMuPDF (fitz)
- Only extracts native text - **IGNORES text in images**
- Chunks text with configurable size (default: 512 tokens) and overlap (default: 50 tokens)
- Maintains page-level metadata (file_name, dataset, page_num)
- Saves/loads processed chunks to JSON for reuse

**document_processor_ocr.py** (Enhanced - recommended)
- Extends document_processor.py with OCR support
- Extracts native text PLUS text from images using Tesseract OCR
- OCR triggers when page has < 100 chars of native text
- Significantly improves extraction quality (2-3x more chunks expected)
- Requires: `pip install pytesseract pillow` + Tesseract engine

**vector_store.py**
- Manages ChromaDB persistent collection
- Uses SentenceTransformer model for embeddings)
- Batch processing for efficient indexing via `put_many()`
- Semantic search via `find(mode='sem')` returns top-k chunks with metadata and scores
- Single portable file replaces directory-based storage

**llm.py**
- Downloads and loads Llama 3.2 3B Instruct Q4_K_M quantized model (~2GB)
- Uses llama-cpp-python for efficient CPU/GPU inference
- Formats RAG prompts with Llama 3.2 chat template
- Default: offloads all layers to GPU (n_gpu_layers=-1) for maximum speed
- Context window: 4096 tokens

**rag.py**
- Orchestrates vector search + LLM generation
- Manages conversation history
- Provides formatted output with Rich library
- Retrieves top_k=5 chunks by default

**scripts/initialize.py**
- One-time setup: processes PDFs → downloads model → builds vector index
- Checks for existing artifacts to avoid redundant work
- Interactive prompts for rebuilding components

### Key Technical Details

**Chunking Strategy**
- Word-based chunking (approximates tokens)
- Overlap prevents information loss at boundaries
- Each chunk maintains full metadata trail to source PDF and page

**Vector Storage & Search**
- ChromaDB: persistent storage in chroma_db/ directory
- Built-in embeddings (no separate model management)
- Semantic search with 'sem' mode for vector similarity
- Also supports 'lex' (BM25) and 'auto' modes
- 1,372× faster than traditional vector stores

**LLM Configuration**
- Model: Llama-3.2-3B-Instruct-Q4_K_M.gguf from HuggingFace (bartowski)
- Quantization: Q4_K_M balances quality and speed
- GPU acceleration via llama-cpp-python with CUDA
- System prompt enforces grounding to provided context only

**Source Attribution**
- Every chunk stores: file_path, file_name, dataset, page_num, total_pages
- Chatbot displays sources with page numbers
- `/sources` command shows detailed excerpts from last query

## Important File Paths

```
epstein_files/          # 525 PDFs organized by DataSet_1/ through DataSet_12/
processed_chunks.json   # Extracted & chunked text (~50MB, generated by initialize)
chroma_db/              # ChromaDB vector database (generated by initialize)
models/                 # Llama 3.2 3B GGUF model (generated by initialize)
src/                    # Core application modules
scripts/                # Initialization and testing utilities
docs/                   # Documentation
tools/                  # Optional utilities (OCR testing)
```

## Modifying Behavior

### Change Chunk Size
Edit `scripts/initialize.py`:
```python
processor = DocumentProcessor(chunk_size=256, overlap=25)  # Smaller chunks
```
Then rerun `python -m scripts.initialize`.

### Adjust Retrieval Count
Edit `src/rag.py` in `answer_question()`:
```python
context_chunks = self.vector_store.search(question, top_k=10)  # More context
```

### Reduce GPU Memory Usage
Edit `src/llm.py` or `src/rag.py` initialization:
```python
self.llm = LLM(self.model_dir, n_ctx=4096, n_gpu_layers=20)  # Offload fewer layers
```

### Change Search Mode
Edit `src/vector_store.py` in `search()`:
```python
# Semantic search (default)
results = self.memory.find(query, k=top_k, mode='sem')

# Lexical search (BM25)
results = self.memory.find(query, k=top_k, mode='lex')

# Auto (intelligent mode selection)
results = self.memory.find(query, k=top_k, mode='auto')
```

### Modify System Prompt
Edit `src/llm.py` in `format_rag_prompt()` method to change instructions.

## Requirements

- **Python**: 3.11+
- **GPU**: NVIDIA with CUDA support (tested on RTX 4070, 12GB VRAM)
- **CUDA**: llama-cpp-python must be compiled with CUDA support
- **Disk**: ~3GB (2GB model + 500MB index + 50MB processed chunks)

## Critical Installation Note

llama-cpp-python MUST be installed with CUDA flags:
```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

Without this, the model runs on CPU only (~50x slower).

## Package Management

The project supports both UV and pip:
- `pyproject.toml` defines dependencies for UV
- `requirements-chatbot.txt` for pip users
- `install.sh` handles both installation methods

## Dependencies

### Core
- `llama-cpp-python` - LLM inference with CUDA
- `chromadb - Vector database
- `pymupdf` - PDF text extraction
- `rich` - Terminal UI
- `torch` - GPU acceleration
- `tqdm` - Progress bars

### Optional (OCR)
- `pytesseract` - OCR wrapper
- `pillow` - Image handling
- `tesseract-ocr` - System OCR engine

## Testing

`scripts/test_setup.py` verifies:
- All Python modules are importable
- PDF files exist
- Directory structure is correct
- Does NOT test CUDA/GPU (check with `nvidia-smi`)

## Documentation

- `README.md` - Main project overview
- `docs/QUICKSTART.md` - Step-by-step setup guide
- `docs/DEVELOPMENT.md` - Technical architecture details
- `docs/OCR_GUIDE.md` - OCR enhancement instructions
- `docs/README_CHATBOT.md` - Chatbot usage guide
- `CONTRIBUTING.md` - Contribution guidelines
