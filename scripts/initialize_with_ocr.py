"""Initialization script with OCR support for better text extraction."""

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processor_ocr import DocumentProcessorOCR
from src.vector_store import VectorStore
from src.llm import LLM


def main():
    """Main initialization function with OCR support."""
    console = Console()

    console.print(Panel(
        "[bold cyan]Epstein Files Chatbot - Initialization with OCR[/bold cyan]\n\n"
        "This script will:\n"
        "1. Process PDF documents with OCR to extract text from images\n"
        "2. Download the Llama 3.2 3B model (~2GB) if needed\n"
        "3. Build the vector search index\n\n"
        "[yellow]OCR will significantly improve text extraction but takes longer.[/yellow]\n"
        "[yellow]Estimated time: 20-30 minutes (vs 5-10 without OCR)[/yellow]",
        border_style="cyan"
    ))

    project_root = Path(__file__).parent.parent
    epstein_files_dir = project_root / "epstein_files"
    processed_chunks_file = project_root / "processed_chunks_ocr.json"  # Different file
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

    # Check OCR availability
    try:
        import pytesseract
        console.print("[green]✓ pytesseract installed[/green]")
    except ImportError:
        console.print(Panel(
            "[bold yellow]Warning:[/bold yellow] pytesseract not installed\n\n"
            "To enable OCR, install:\n"
            "  pip install pytesseract pillow\n\n"
            "Also install Tesseract OCR engine:\n"
            "  Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
            "  macOS: brew install tesseract\n"
            "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n\n"
            "Continue without OCR?",
            border_style="yellow"
        ))
        if not Confirm.ask("Continue without OCR?", default=False):
            sys.exit(1)

    # Step 1: Process documents with OCR
    console.print("\n[bold cyan]Step 1: Processing PDF Documents with OCR[/bold cyan]")

    if processed_chunks_file.exists():
        console.print(f"[yellow]Found existing OCR-processed chunks at {processed_chunks_file}[/yellow]")
        if not Confirm.ask("Reprocess all documents with OCR?", default=False):
            console.print("[green]Using existing processed chunks[/green]")
            chunks = DocumentProcessorOCR.load_processed_chunks(processed_chunks_file)
        else:
            processor = DocumentProcessorOCR(chunk_size=512, overlap=50, use_ocr=True)
            chunks = processor.process_all_documents(epstein_files_dir, processed_chunks_file)
    else:
        processor = DocumentProcessorOCR(chunk_size=512, overlap=50, use_ocr=True)
        chunks = processor.process_all_documents(epstein_files_dir, processed_chunks_file)

    console.print(f"[green]✓[/green] Processed {len(chunks)} chunks")

    # Compare with non-OCR version if it exists
    old_chunks_file = project_root / "processed_chunks.json"
    if old_chunks_file.exists():
        import json
        with open(old_chunks_file) as f:
            old_chunks = json.load(f)

        console.print(f"\n[cyan]Comparison:[/cyan]")
        console.print(f"  Without OCR: {len(old_chunks)} chunks")
        console.print(f"  With OCR:    {len(chunks)} chunks")
        console.print(f"  Difference:  +{len(chunks) - len(old_chunks)} chunks ({(len(chunks) / len(old_chunks) - 1) * 100:.1f}% more)")

    # Step 2: Download model (if needed)
    console.print("\n[bold cyan]Step 2: Language Model[/bold cyan]")

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
        "[bold green]Initialization Complete with OCR![/bold green]\n\n"
        "The chatbot now has access to text extracted from images.\n\n"
        "Start the chatbot with:\n"
        "[yellow]python -m src.chatbot[/yellow]\n\n"
        "Or:\n"
        "[yellow]./run_chatbot.sh[/yellow]\n\n"
        f"[dim]OCR-processed chunks saved to: {processed_chunks_file}[/dim]",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
