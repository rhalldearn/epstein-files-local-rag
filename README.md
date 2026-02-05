# Epstein Files RAG Chatbot

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Local AI-powered chatbot for querying the DOJ Epstein Files documents using Retrieval Augmented Generation (RAG)**

Chat with the Epstein Files using a completely local AI system powered by Llama 3.2 3B and ChromaDB vector search. No cloud services, no API keys, no data leaves your machine.

TODO: Still needs testing with new downloading approach.

## ⚠️ CONTENT WARNING

- If you are a survivor of sexual abuse, please consider whether engaging with this material is appropriate for your wellbeing.
- The Epstein documents processed by this tool contain detailed descriptions and testimony related to serious crimes, including the sexual exploitation and abuse of minors.
- The content is graphic, disturbing, and may be triggering for survivors of abuse.

---

## ✨ Features

- 🤖 **Local LLM**: Runs Llama 3.2 3B quantized model (~2GB) entirely on your GPU/CPU
- 🔍 **Fast Semantic Search**: ChromaDB vector storage with efficient retrieval
- 📄 **OCR Support**: Optional Tesseract OCR extracts text trapped in images (2-3x more content)
- 💾 **No Size Limits**: ChromaDB handles any dataset size
- 📚 **Source Attribution**: Every answer includes page numbers and source documents
- 🎨 **Rich CLI**: Beautiful terminal interface with syntax highlighting

## 🚀 Quick Start

### Prerequisites

1. **Python 3.11+** installed
2. **NVIDIA GPU** with CUDA support (recommended, but CPU works)
3. **files** from DOJ Epstein Files ([download instructions below](#downloading-the-files))

### Installation - I am using Ubuntu with Nvidia RTX4070 GPU

```bash
# Clone the repository
git clone https://github.com/rhalldearn/epstein-files-local-rag.git
cd epstein-files-local-rag

# Run the installation script
./install.sh

# This will:
# - Install dependencies (including CUDA support)
# - Download the Llama 3.2 3B model (~2GB)
# - Process PDFs and build the searchable index
```

### Running the Chatbot

```bash
# Start the interactive chatbot
./run_chatbot.sh
# or
python -m src.chatbot
```

## 📥 Downloading the Files

The chatbot requires the DOJ documents from the DOJ's Epstein Files Transparency Act. We recommend using the excellent downloader by [@Surebob](https://github.com/Surebob):

**[Surebob's Epstein Files Downloader](https://github.com/Surebob/epstein-files-downloader)**

This tool provides:
- ⚡ Fast torrent downloads via Archive.org mirrors
- 📦 Direct ZIP downloads for datasets 1-8 and 12
- 🌐 Web scraping for individual PDFs from Dataset 9
- ▶️ Resume capability for interrupted downloads
- ✅ Checksum verification

Take a long time to download!

After downloading, ensure the files are in: `./epstein_files/DataSet_1/` through `DataSet_12/`

## 🧠 How It Works

```
User Question
    ↓
[Vector Search] - Find relevant PDF chunks using semantic similarity
    ↓
[LLM Generation] - Llama 3.2 3B generates answer from context
    ↓
[Source Attribution] - Display answer with page numbers
```

**Architecture:**
- **Document Processing**: PyMuPDF extracts text, optional Tesseract OCR for images
- **Chunking**: 512-token chunks with 50-token overlap for context preservation
- **Vector Storage**: ChromaDB with SentenceTransformer embeddings (all-MiniLM-L6-v2)
- **LLM**: Llama 3.2 3B Instruct (Q4_K_M quantized) via llama-cpp-python
- **Interface**: Rich CLI with commands (`/help`, `/sources`, `/info`)

## 📖 Usage Examples - Needs data munging improvements to be useful!

```
> Tell me about Epstein's communication with Mandelson.

blah blah....

Sources: DataSet_3/EFTA00012345.pdf (p. 14), DataSet_5/EFTA00023456.pdf (p. 7)

> /sources
[Shows detailed excerpts from last answer]

> /info
Total chunks: 18,432
Storage: ChromaDB
Model: Llama 3.2 3B Instruct
```

## 🔧 Advanced Configuration

### Use OCR for Better Extraction

Many PDFs contain text in images that standard extraction misses. Enable OCR:

```bash
# Install OCR dependencies
pip install pytesseract pillow
sudo apt-get install tesseract-ocr  # Linux
# brew install tesseract  # macOS

# Run initialization with OCR (slower but extracts 2-3x more text)
python -m scripts.initialize_with_ocr
```

See [docs/OCR_GUIDE.md](docs/OCR_GUIDE.md) for details.

### Adjust GPU Memory Usage

Edit `src/llm.py` to control how many model layers are offloaded to GPU:

```python
self.llm = Llama(
    model_path=str(self.model_path),
    n_gpu_layers=-1,  # -1 = all layers to GPU, reduce for less VRAM
    n_ctx=4096
)
```

### Change Retrieval Settings

Edit `src/rag.py` to retrieve more context:

```python
context_chunks = self.vector_store.search(question, top_k=10)  # Default: 5
```

## 🤝 Contributing

We welcome contributions! This project is in active development and there are many opportunities to improve it:

**Areas for improvement:**
- Better chunking strategies (semantic chunking, section-aware)
- Conversation context memory
- Web interface (Gradio/Streamlit)
- Answer quality evaluation
- Support for other LLMs
- Performance optimizations
- Testing and documentation

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📚 Documentation

- [Quick Start Guide](docs/QUICKSTART.md) - Step-by-step setup
- [Architecture Deep Dive](docs/DEVELOPMENT.md) - Technical details
- [OCR Enhancement Guide](docs/OCR_GUIDE.md) - Extract text from images
- [Claude Code Instructions](CLAUDE.md) - For AI-assisted development

## 🛠️ System Requirements

- **Minimum:**
  - Python 3.11+
  - 8GB RAM
  - 5GB disk space
  - CPU-only inference (slow but functional)

- **Recommended:**
  - NVIDIA GPU with 6GB+ VRAM (RTX 3060 or better)
  - 16GB RAM
  - 10GB disk space
  - CUDA 12.0+

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Llama 3.2 3B** by Meta AI
- **ChromaDB** - Vector database for semantic search
- **PDF Downloader** by [@Surebob](https://github.com/Surebob/epstein-files-downloader)
- **llama-cpp-python** by [@abetlen](https://github.com/abetlen/llama-cpp-python)
- **SentenceTransformers** for embeddings

## ⚠️ Disclaimer

This tool is intended strictly for research, journalism, legal analysis, and educational purposes. The documents processed by this software are public records released by the U.S. Department of Justice under the Epstein Files Transparency Act.

**Important:**
- This project is not affiliated with the U.S. Department of Justice or any government agency
- The content includes testimony and evidence related to serious crimes against minors
- Users are responsible for complying with all applicable laws regarding the handling of sensitive legal documents
- This software does not modify, editorialize, or filter the source documents
- The AI-generated responses should be verified against the original source documents
- This tool should not be used for any purpose that could harm victims or impede justice

## 🔗 Related Projects

- [Surebob's Epstein Files Downloader](https://github.com/Surebob/epstein-files-downloader) - Download the PDFs
- [ChromaDB](https://www.trychroma.com/) - Vector database for AI applications
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - LLM inference in C/C++
