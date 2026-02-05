"""Initialization script for setting up the chatbot."""

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.llm import LLM


def main():
    """Main initialization function."""
    console = Console()

    console.print(Panel(
        "[bold cyan]Epstein Files Chatbot - Initialization[/bold cyan]\n\n"
        "This script will:\n"
        "1. Process PDF documents and extract text\n"
        "2. Download the Llama 3.2 3B model (~2GB)\n"
        "3. Build the vector search index\n\n"
        "[yellow]This may take 5-10 minutes on first run.[/yellow]",
        border_style="cyan"
    ))

    project_root = Path(__file__).parent.parent
    epstein_files_dir = project_root / "epstein_files"
    processed_chunks_file = project_root / "processed_chunks.json"
    chroma_db_path = project_root / "chroma_db"
    model_dir = project_root / "models"

    # Check if epstein_files directory exists
    if not epstein_files_dir.exists():
        console.print(Panel(
            f"[bold red]Error:[/bold red] Could not find epstein_files directory at:\n"
            f"{epstein_files_dir}\n\n"
            "Please ensure the PDF files are downloaded first.",
            border_style="red"
        ))
        sys.exit(1)

    # Step 1: Process documents
    console.print("\n[bold cyan]Step 1: Processing PDF Documents[/bold cyan]")

    if processed_chunks_file.exists():
        console.print(f"[yellow]Found existing processed chunks at {processed_chunks_file}[/yellow]")
        if not Confirm.ask("Reprocess all documents?", default=False):
            console.print("[green]Using existing processed chunks[/green]")
            chunks = DocumentProcessor.load_processed_chunks(processed_chunks_file)
        else:
            processor = DocumentProcessor(chunk_size=512, overlap=50)
            chunks = processor.process_all_documents(epstein_files_dir, processed_chunks_file)
    else:
        processor = DocumentProcessor(chunk_size=512, overlap=50)
        chunks = processor.process_all_documents(epstein_files_dir, processed_chunks_file)

    console.print(f"[green]✓[/green] Processed {len(chunks)} chunks")

    # Step 2: Download model
    console.print("\n[bold cyan]Step 2: Downloading Language Model[/bold cyan]")

    llm = LLM(model_dir)
    if llm.model_path.exists():
        console.print(f"[green]✓[/green] Model already exists at {llm.model_path}")
    else:
        try:
            llm.download_model()
            console.print(f"[green]✓[/green] Model downloaded successfully")
        except Exception as e:
            console.print(f"[bold red]Error downloading model:[/bold red] {e}")
            sys.exit(1)

    # Step 3: Build vector index
    console.print("\n[bold cyan]Step 3: Building Vector Search Index[/bold cyan]")

    try:
        vector_store = VectorStore(chroma_db_path)
        vector_store.initialize()
        vector_store.build_index(chunks, batch_size=100)

        stats = vector_store.get_stats()
        console.print(f"[green]✓[/green] Vector index built with {stats['total_chunks']} chunks")

    except Exception as e:
        console.print(f"[bold red]Error building index:[/bold red] {e}")
        sys.exit(1)

    # Success message
    console.print(Panel(
        "[bold green]Initialization Complete![/bold green]\n\n"
        "The chatbot is now ready to use.\n\n"
        "Start the chatbot with:\n"
        "[yellow]python -m src.chatbot[/yellow]\n\n"
        "Or if installed with uv:\n"
        "[yellow]epstein-chat[/yellow]",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
