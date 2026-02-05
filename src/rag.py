"""RAG pipeline orchestration module."""

from pathlib import Path
from typing import List, Dict, Any, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from src.vector_store import VectorStore
from src.llm import LLM


class RAGChatbot:
    """Main RAG chatbot orchestration class."""

    def __init__(self, chroma_db_path: Path, model_dir: Path):
        """
        Initialize the RAG chatbot.

        Args:
            chroma_db_path: Path to ChromaDB storage directory
            model_dir: Path to model directory
        """
        self.chroma_db_path = chroma_db_path
        self.model_dir = model_dir
        self.vector_store = None
        self.llm = None
        self.console = Console()
        self.conversation_history = []

    def initialize(self):
        """Initialize vector store and LLM."""
        self.console.print("\n[bold cyan]Initializing RAG Chatbot...[/bold cyan]\n")

        # Initialize vector store
        self.console.print("[yellow]Loading vector store...[/yellow]")
        self.vector_store = VectorStore(self.chroma_db_path)
        self.vector_store.initialize()

        stats = self.vector_store.get_stats()
        self.console.print(f"[green]✓[/green] Vector store loaded: {stats['total_chunks']} chunks indexed")

        # Initialize LLM
        self.console.print("\n[yellow]Loading language model...[/yellow]")
        self.llm = LLM(self.model_dir, n_ctx=4096, n_gpu_layers=-1)

        if not self.llm.model_path.exists():
            self.console.print("[red]Model not found. Downloading...[/red]")
            self.llm.download_model()

        self.llm.initialize()

        model_info = self.llm.get_model_info()
        self.console.print(f"[green]✓[/green] Model loaded: {model_info['model_name']} ({model_info['model_size']})")

        self.console.print("\n[bold green]Chatbot ready![/bold green]\n")

    def answer_question(self, question: str, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Answer a question using the RAG pipeline.

        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve

        Returns:
            Tuple of (answer, source_chunks)
        """
        # Retrieve relevant chunks
        self.console.print("[dim]Searching documents...[/dim]")
        context_chunks = self.vector_store.search(question, top_k=top_k)

        if not context_chunks:
            return "I couldn't find any relevant information in the documents.", []

        # Generate answer
        self.console.print("[dim]Generating answer...[/dim]\n")
        answer = self.llm.generate(question, context_chunks)

        # Store in history
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "sources": context_chunks
        })

        return answer, context_chunks

    def format_answer_with_sources(self, answer: str, sources: List[Dict[str, Any]]):
        """
        Format and display the answer with source citations.

        Args:
            answer: Generated answer text
            sources: List of source chunks
        """
        # Display answer
        self.console.print(Panel(
            Markdown(answer),
            title="[bold cyan]Answer[/bold cyan]",
            border_style="cyan"
        ))

        # Display sources
        if sources:
            self.console.print("\n[bold yellow]Sources:[/bold yellow]")
            for i, source in enumerate(sources, 1):
                metadata = source.get("metadata", {})
                file_name = metadata.get("file_name", "Unknown")
                page_num = metadata.get("page_num", "?")
                dataset = metadata.get("dataset", "Unknown")

                source_text = source["text"][:200] + "..." if len(source["text"]) > 200 else source["text"]

                self.console.print(f"\n[bold cyan]Source {i}:[/bold cyan] {file_name} (page {page_num}, dataset: {dataset})")
                self.console.print(f"[dim]{source_text}[/dim]")

    def get_last_sources(self) -> List[Dict[str, Any]]:
        """
        Get sources from the last question.

        Returns:
            List of source chunks from last query
        """
        if not self.conversation_history:
            return []
        return self.conversation_history[-1]["sources"]

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        self.console.print("[green]Conversation history cleared[/green]")

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information.

        Returns:
            Dictionary with system stats
        """
        info = {
            "vector_store": self.vector_store.get_stats() if self.vector_store else {},
            "llm": self.llm.get_model_info() if self.llm else {},
            "conversation_length": len(self.conversation_history)
        }
        return info
