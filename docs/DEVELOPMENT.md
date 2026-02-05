# Development Guide

Guide for extending and customizing the Epstein Files RAG Chatbot.

## Project Architecture

### Module Overview

```
src/
├── document_processor.py  # PDF → Text chunks
├── vector_store.py        # Text chunks → Embeddings → Search
├── llm.py                 # Context + Question → Answer
├── rag.py                 # Orchestrates the above
└── chatbot.py             # User interface
```

### Data Flow

```
User Question
    ↓
[chatbot.py] Parse input
    ↓
[rag.py] Coordinate pipeline
    ↓
[vector_store.py] Semantic search → Top-K chunks
    ↓
[llm.py] Format prompt + Generate answer
    ↓
[chatbot.py] Display answer + sources
```

## Customization Examples

### 1. Adjust Chunk Size

Smaller chunks = more precise retrieval, less context per chunk
Larger chunks = more context per chunk, less precise retrieval

**Edit `scripts/initialize.py`:**
```python
processor = DocumentProcessor(
    chunk_size=256,    # Smaller chunks
    overlap=25         # Adjust overlap proportionally
)
```

**Re-run initialization:**
```bash
python -m scripts.initialize
```

### 2. Change Retrieval Count

Fewer chunks = faster, less context
More chunks = slower, more context

**Edit `src/chatbot.py` or `src/rag.py`:**
```python
# In answer_question method
context_chunks = self.vector_store.search(question, top_k=10)  # More context
```

### 3. Use Different Embedding Model

**Edit `src/vector_store.py`:**
```python
# In initialize method
self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# Or: 'all-mpnet-base-v2' (higher quality, slower)
# Or: 'all-MiniLM-L12-v2' (better quality than L6)
```

**Rebuild index:**
```bash
python -m scripts.initialize
# Choose "yes" to rebuild
```

### 4. Change LLM Model

**Edit `src/llm.py`:**
```python
class LLM:
    # Update model URL and filename
    MODEL_URL = "https://huggingface.co/user/model/resolve/main/model.gguf"
    MODEL_FILENAME = "model.gguf"
```

Compatible models:
- Llama 3.2 1B (faster, less capable)
- Llama 3.2 3B (balanced - current)
- Mistral 7B (more capable, slower)
- Phi-3 (efficient alternative)

### 5. Modify System Prompt

**Edit `src/llm.py` in `format_rag_prompt` method:**
```python
system_message = """You are a helpful assistant...

Instructions:
- Answer based ONLY on the provided context
- [Add your custom instructions here]
- Be factual and objective
"""
```

### 6. Add Filtering by Dataset

**Add to `src/vector_store.py`:**
```python
def search(self, query: str, top_k: int = 5,
           dataset_filter: str = None) -> List[Dict[str, Any]]:
    """Search with optional dataset filtering."""

    query_embedding = self.embedding_model.encode([query])[0].tolist()

    # Add where clause for filtering
    where_clause = None
    if dataset_filter:
        where_clause = {"dataset": dataset_filter}

    results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_clause  # Apply filter
    )

    # ... rest of method
```

**Use in chatbot:**
```python
# Search only in specific dataset
chunks = self.vector_store.search(
    question,
    top_k=5,
    dataset_filter="set_1"
)
```

### 7. Add Date Range Filtering

**Modify `src/document_processor.py` to extract dates:**
```python
import re
from datetime import datetime

def extract_dates(text: str) -> List[str]:
    """Extract dates from text."""
    # Simple date pattern (extend as needed)
    date_pattern = r'\b\d{1,2}/\d{1,2}/\d{4}\b'
    return re.findall(date_pattern, text)

# In chunk_text method, add:
dates = extract_dates(chunk_text)
chunks.append({
    "text": chunk_text,
    "metadata": {
        **metadata,
        "dates": dates  # Store extracted dates
    }
})
```

### 8. Add Conversation Context

**Modify `src/rag.py` to include conversation history:**
```python
def answer_question(self, question: str, top_k: int = 5,
                    use_history: bool = False) -> Tuple[str, List[Dict[str, Any]]]:
    """Answer with optional conversation history."""

    # Enhance question with recent context
    if use_history and self.conversation_history:
        last_qa = self.conversation_history[-1]
        enhanced_question = f"Previous: {last_qa['question']}\nCurrent: {question}"
        context_chunks = self.vector_store.search(enhanced_question, top_k=top_k)
    else:
        context_chunks = self.vector_store.search(question, top_k=top_k)

    # ... rest of method
```

### 9. Add Response Streaming

**Modify `src/llm.py` for streaming:**
```python
def generate_streaming(self, question: str, context_chunks: List[Dict[str, Any]]):
    """Generate answer with streaming output."""

    prompt = self.format_rag_prompt(question, context_chunks)

    for token in self.llm(
        prompt,
        max_tokens=512,
        temperature=0.7,
        stream=True  # Enable streaming
    ):
        yield token["choices"][0]["text"]
```

**Use in chatbot:**
```python
for token in self.llm.generate_streaming(question, context_chunks):
    console.print(token, end="")
```

### 10. Add Export Functionality

**Add to `src/chatbot.py`:**
```python
import json
from datetime import datetime

def export_conversation(chatbot: RAGChatbot):
    """Export conversation history to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(chatbot.conversation_history, f, indent=2)

    console.print(f"[green]Conversation exported to {filename}[/green]")

# Add to command handler:
elif command == "/export":
    export_conversation(chatbot)
```

## Performance Optimization

### 1. Reduce Model Size

Use smaller quantization:
- Q2_K: Smallest, lowest quality
- Q3_K_M: Small, decent quality
- Q4_K_M: Balanced (current)
- Q5_K_M: Larger, better quality
- Q8_0: Largest, best quality

### 2. Batch Processing

**For multiple questions:**
```python
questions = ["Question 1", "Question 2", "Question 3"]

# Generate embeddings in batch
embeddings = vector_store.embedding_model.encode(questions, batch_size=32)

# Search for each
for question, embedding in zip(questions, embeddings):
    results = collection.query(query_embeddings=[embedding.tolist()], n_results=5)
    # ... process results
```

### 3. Cache Frequent Queries

**Add caching to `src/rag.py`:**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_answer(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Cache answers to frequent questions."""
    return self.answer_question(question)
```

### 4. Optimize Context Window

Reduce context size to fit more in GPU:
```python
llm = LLM(model_dir, n_ctx=2048)  # Smaller context window
```

## Testing

### Unit Tests

```python
# tests/test_document_processor.py
import pytest
from src.document_processor import DocumentProcessor

def test_chunk_text():
    processor = DocumentProcessor(chunk_size=10, overlap=2)
    text = "word " * 50
    metadata = {"file": "test.pdf"}

    chunks = processor.chunk_text(text, metadata)

    assert len(chunks) > 0
    assert all("text" in chunk for chunk in chunks)
    assert all("metadata" in chunk for chunk in chunks)
```

### Integration Tests

```python
# tests/test_rag_pipeline.py
def test_full_pipeline():
    chatbot = RAGChatbot(chroma_db_path, model_dir)
    chatbot.initialize()

    answer, sources = chatbot.answer_question("Test question?")

    assert isinstance(answer, str)
    assert len(answer) > 0
    assert len(sources) > 0
```

## Debugging

### Enable Verbose Logging

**In `src/llm.py`:**
```python
self.llm = Llama(
    model_path=str(self.model_path),
    n_ctx=self.n_ctx,
    n_gpu_layers=self.n_gpu_layers,
    verbose=True  # Enable verbose output
)
```

### Inspect Retrieved Chunks

**Add to `src/chatbot.py`:**
```python
if command == "/debug":
    if not chatbot.conversation_history:
        console.print("[yellow]No history available[/yellow]")
    else:
        last = chatbot.conversation_history[-1]
        console.print("\n[bold cyan]Last Query Debug Info:[/bold cyan]")
        console.print(f"Question: {last['question']}")
        console.print(f"Number of sources: {len(last['sources'])}")
        for i, source in enumerate(last['sources'], 1):
            console.print(f"\n[cyan]Chunk {i}:[/cyan]")
            console.print(source['text'][:200])
```

### Monitor GPU Usage

```bash
# In another terminal
watch -n 1 nvidia-smi
```

## Contributing

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to all public methods
- Keep functions focused and small

### Adding New Features

1. Create feature branch
2. Implement in appropriate module
3. Add tests
4. Update documentation
5. Test with `python -m scripts.test_setup`
6. Submit pull request

## Common Modifications

### Add Web UI (Flask Example)

```python
# app.py
from flask import Flask, request, jsonify
from src.rag import RAGChatbot

app = Flask(__name__)
chatbot = RAGChatbot(chroma_db_path, model_dir)
chatbot.initialize()

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question")
    answer, sources = chatbot.answer_question(question)
    return jsonify({
        "answer": answer,
        "sources": [s["metadata"] for s in sources]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

### Add API Endpoint

```python
# api.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
chatbot = RAGChatbot(chroma_db_path, model_dir)

class Question(BaseModel):
    text: str
    top_k: int = 5

@app.post("/query")
async def query(q: Question):
    answer, sources = chatbot.answer_question(q.text, top_k=q.top_k)
    return {"answer": answer, "sources": sources}
```

## Resources

- [LLama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [RAG Best Practices](https://www.anthropic.com/index/retrieval-augmented-generation)

## Support

For issues or questions:
1. Check `QUICKSTART.md` for common setup issues
2. Run `python -m scripts.test_setup` to diagnose problems
3. Check logs and error messages
4. Review this development guide for customization options
