# Contributing to Epstein Files RAG Chatbot

Thank you for your interest in contributing! This project welcomes contributions from developers of all skill levels. Whether you're fixing bugs, adding features, improving documentation, or optimizing performance, your help is appreciated.

## 🎯 Areas for Contribution

### High Priority

1. **Better Chunking Strategies**
   - Current: Simple word-based chunking with fixed size
   - Needed: Semantic chunking, section-aware splitting, better overlap handling
   - Impact: Improves answer quality and relevance

2. **Conversation Context Memory**
   - Current: Each question is independent
   - Needed: Track conversation history, allow follow-up questions
   - Impact: More natural interactions

3. **Web Interface**
   - Current: CLI only
   - Needed: Gradio or Streamlit web UI
   - Impact: Easier access for non-technical users

4. **Answer Quality Evaluation**
   - Current: No automated quality metrics
   - Needed: Evaluation framework, benchmark questions, quality scoring
   - Impact: Objective measurement of improvements

### Medium Priority

5. **Testing & CI/CD**
   - Unit tests for document processing
   - Integration tests for RAG pipeline
   - GitHub Actions for automated testing

6. **Performance Optimizations**
   - Faster embedding generation
   - Batch query processing
   - Memory usage optimization

7. **Multi-Model Support**
   - Support for other LLMs (Llama 3.1, Mistral, etc.)
   - Model switching from CLI
   - Comparative benchmarks

8. **Enhanced OCR**
   - Better image detection
   - Multi-language OCR support
   - Table extraction

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/yourusername/epstein-files.git
cd epstein-files
```

### 2. Set Up Development Environment

```bash
# Install dependencies
./install.sh

# Or manually with pip:
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements-chatbot.txt

# Install llama-cpp-python with CUDA
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Install development dependencies
pip install pytest black ruff
```

### 3. Download PDFs (if needed)

Use [Surebob's downloader](https://github.com/Surebob/epstein-files-downloader) to get the PDF files.

### 4. Initialize the System

```bash
# Process PDFs and build index
python -m scripts.initialize
```

### 5. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## 📝 Code Style

We follow Python best practices:

- **Formatting**: Use `black` for code formatting
  ```bash
  black src/ scripts/
  ```

- **Linting**: Use `ruff` for linting
  ```bash
  ruff check src/ scripts/
  ```

- **Docstrings**: Use Google-style docstrings
  ```python
  def process_document(file_path: Path, chunk_size: int = 512) -> List[Dict]:
      """Process a PDF document and extract chunks.

      Args:
          file_path: Path to the PDF file
          chunk_size: Size of each text chunk in words

      Returns:
          List of chunk dictionaries with text and metadata

      Raises:
          FileNotFoundError: If the PDF file doesn't exist
      """
      pass
  ```

- **Type Hints**: Add type hints to function signatures
- **Comments**: Explain *why*, not *what* (code should be self-documenting)

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_document_processor.py

# Run with coverage
pytest --cov=src tests/
```

### Writing Tests

Create test files in `tests/` directory:

```python
# tests/test_document_processor.py
import pytest
from src.document_processor import DocumentProcessor

def test_chunk_text():
    processor = DocumentProcessor(chunk_size=10, overlap=2)
    text = "This is a test document with multiple words."
    chunks = processor.chunk_text(text)

    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)
```

## 📋 Pull Request Process

### 1. Make Your Changes

- Write clear, focused commits
- Follow the code style guidelines
- Add tests for new features
- Update documentation as needed

### 2. Test Thoroughly

```bash
# Format code
black src/ scripts/

# Run linter
ruff check src/ scripts/

# Run tests
pytest

# Test the chatbot manually
python -m src.chatbot
```

### 3. Update Documentation

- Update relevant `.md` files in `docs/`
- Add docstrings to new functions/classes
- Update `CLAUDE.md` if architecture changes

### 4. Commit and Push

```bash
git add .
git commit -m "feat: add conversation memory to RAG pipeline"
# or
git commit -m "fix: resolve OCR issue with rotated images"

git push origin feature/your-feature-name
```

### 5. Create Pull Request

- Go to GitHub and create a PR from your branch
- Fill out the PR template (title, description, testing notes)
- Link any related issues
- Request review from maintainers

## 🏷️ Commit Message Format

We use conventional commits:

```
type(scope): brief description

Longer explanation if needed.

Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(rag): add conversation context memory
fix(ocr): handle rotated images correctly
docs(readme): add GPU memory requirements
perf(vector): optimize batch embedding generation
```

## 🐛 Reporting Bugs

### Before Submitting

1. Check existing [Issues](https://github.com/rhalldearn/epstein-files/issues)
2. Verify you're using the latest version
3. Test with a minimal reproducible example

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. Enter query '...'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11.5]
- GPU: [e.g., RTX 4070, 12GB VRAM]
- CUDA version: [e.g., 12.1]

**Logs**
```
Paste relevant error messages or logs here
```
```

## 💡 Feature Requests

We welcome feature ideas! Please open an issue with:

1. **Use case**: What problem does this solve?
2. **Proposed solution**: How should it work?
3. **Alternatives**: Other approaches you've considered
4. **Impact**: Who would benefit from this feature?

## 🔍 Code Review Guidelines

When reviewing PRs, we look for:

- **Correctness**: Does it work as intended?
- **Style**: Follows code style guidelines?
- **Tests**: Are there tests? Do they pass?
- **Documentation**: Is it documented?
- **Performance**: Any performance implications?
- **Backwards compatibility**: Does it break existing functionality?

## 📞 Getting Help

- **Questions**: Open a [Discussion](https://github.com/rhalldearn/epstein-files/discussions)
- **Bugs**: Open an [Issue](https://github.com/rhalldearn/epstein-files/issues)
- **Chat**: Join our Discord (coming soon)

## 🎓 Learning Resources

New to RAG or LLMs? Check out:

- [Memvid Documentation](https://docs.memvid.com)
- [llama.cpp Guide](https://github.com/ggerganov/llama.cpp)
- [RAG Fundamentals](https://docs.llamaindex.ai/en/stable/understanding/rag/)
- [LangChain Documentation](https://docs.langchain.com)

## 📜 Code of Conduct

This project follows a simple code of conduct:

- Be respectful and constructive
- Welcome newcomers and help them learn
- Focus on the technical merits of contributions
- Give credit where credit is due

## 🏆 Recognition

Contributors will be:
- Listed in README acknowledgments
- Credited in release notes
- Given appropriate GitHub badges

Thank you for contributing to make this project better!

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.
