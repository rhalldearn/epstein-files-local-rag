"""Background initialization script with progress tracking and ETA."""

import sys
import time
import signal
from pathlib import Path
from datetime import datetime, timedelta
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.prompt import Confirm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processor import DocumentProcessor
from src.document_processor_ocr import DocumentProcessorOCR
from src.vector_store import VectorStore
from src.llm import LLM


class BackgroundProcessor:
    """Handles background processing with progress tracking."""

    def __init__(self, use_ocr: bool = False):
        self.console = Console()
        self.use_ocr = use_ocr
        self.interrupted = False

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals gracefully."""
        self.interrupted = True
        self.console.print("\n[yellow]Interrupt received. Saving checkpoint and exiting...[/yellow]")

    def process_documents(self, epstein_files_dir: Path, processed_chunks_file: Path):
        """Process documents with progress tracking."""
        self.console.print(Panel(
            f"[bold cyan]Processing PDF Documents[/bold cyan]\n"
            f"Mode: {'OCR Enabled' if self.use_ocr else 'Standard'}\n"
            f"Checkpoint interval: Every 100 files\n"
            f"Press Ctrl+C to stop gracefully",
            border_style="cyan"
        ))

        # Initialize processor
        if self.use_ocr:
            processor = DocumentProcessorOCR(chunk_size=512, overlap=50, checkpoint_interval=100)
        else:
            processor = DocumentProcessor(chunk_size=512, overlap=50, checkpoint_interval=100)

        # Start processing
        start_time = time.time()
        try:
            chunks = processor.process_all_documents(epstein_files_dir, processed_chunks_file, resume=True)
            elapsed = time.time() - start_time

            self.console.print(f"\n[green]✓[/green] Processing complete!")
            self.console.print(f"Total chunks: {len(chunks)}")
            self.console.print(f"Time elapsed: {str(timedelta(seconds=int(elapsed)))}")

            return chunks

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Processing interrupted. Progress saved to checkpoint.[/yellow]")
            self.console.print("Run this script again to resume from where you left off.")
            sys.exit(0)
        except Exception as e:
            self.console.print(f"\n[red]Error during processing:[/red] {e}")
            self.console.print("[yellow]Progress has been saved. You can resume by running this script again.[/yellow]")
            sys.exit(1)

    def check_status(self, processed_chunks_file: Path):
        """Check current processing status."""
        checkpoint_path = processed_chunks_file.parent / f"{processed_chunks_file.stem}_checkpoint.json"

        if not checkpoint_path.exists():
            self.console.print("[yellow]No checkpoint found. Processing hasn't started or has completed.[/yellow]")
            return None

        import json
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)

        # Get total PDF count
        project_root = Path(__file__).parent.parent
        epstein_files_dir = project_root / "epstein_files"
        total_pdfs = len(list(epstein_files_dir.rglob("*.pdf")))

        processed_count = len(checkpoint["processed_files"])
        remaining = total_pdfs - processed_count
        progress_pct = (processed_count / total_pdfs * 100) if total_pdfs > 0 else 0

        self.console.print(Panel(
            f"[bold cyan]Processing Status[/bold cyan]\n\n"
            f"Total PDFs: {total_pdfs:,}\n"
            f"Processed: {processed_count:,}\n"
            f"Remaining: {remaining:,}\n"
            f"Progress: {progress_pct:.1f}%\n"
            f"Chunks so far: {checkpoint['total_chunks']:,}\n"
            f"Last saved: {checkpoint.get('last_saved', 'Unknown')}",
            border_style="cyan"
        ))

        return checkpoint


def main():
    """Main function."""
    console = Console()

    # Parse arguments
    use_ocr = "--ocr" in sys.argv
    check_only = "--status" in sys.argv

    project_root = Path(__file__).parent.parent
    epstein_files_dir = project_root / "epstein_files"
    processed_chunks_file = project_root / "processed_chunks.json"
    chroma_db_path = project_root / "chroma_db"
    model_dir = project_root / "models"

    processor = BackgroundProcessor(use_ocr=use_ocr)

    # Check status only
    if check_only:
        processor.check_status(processed_chunks_file)
        sys.exit(0)

    # Check if epstein_files directory exists
    if not epstein_files_dir.exists():
        console.print(Panel(
            f"[bold red]Error:[/bold red] Could not find epstein_files directory at:\n"
            f"{epstein_files_dir}\n\n"
            "Please ensure the PDF files are downloaded first.",
            border_style="red"
        ))
        sys.exit(1)

    # Show welcome message
    console.print(Panel(
        "[bold cyan]Epstein Files Chatbot - Background Initialization[/bold cyan]\n\n"
        "This script will:\n"
        "1. Process PDF documents and extract text (with checkpoints)\n"
        "2. Download the Llama 3.2 3B model (~2GB)\n"
        "3. Build the vector search index\n\n"
        f"[yellow]Mode: {'OCR Enabled (slower, more accurate)' if use_ocr else 'Standard (faster)'}[/yellow]\n"
        f"[yellow]Expected time: 7-15 hours for 43,383 PDFs[/yellow]\n\n"
        "Features:\n"
        "• Progress saved every 100 files\n"
        "• Can be interrupted with Ctrl+C and resumed later\n"
        "• Check status anytime with: python -m scripts.initialize_background --status",
        border_style="cyan"
    ))

    # Step 1: Process documents
    console.print("\n[bold cyan]Step 1: Processing PDF Documents[/bold cyan]")

    # Check for existing checkpoint
    checkpoint_path = processed_chunks_file.parent / f"{processed_chunks_file.stem}_checkpoint.json"
    if checkpoint_path.exists():
        processor.check_status(processed_chunks_file)
        if not Confirm.ask("\nResume from checkpoint?", default=True):
            console.print("[yellow]Starting fresh...[/yellow]")
            checkpoint_path.unlink()
            temp_file = processed_chunks_file.parent / f"{processed_chunks_file.stem}_temp.jsonl"
            if temp_file.exists():
                temp_file.unlink()

    elif processed_chunks_file.exists():
        console.print(f"[yellow]Found existing processed chunks at {processed_chunks_file}[/yellow]")
        if not Confirm.ask("Reprocess all documents?", default=False):
            console.print("[green]Using existing processed chunks[/green]")
            chunks = DocumentProcessor.load_processed_chunks(processed_chunks_file)
        else:
            chunks = processor.process_documents(epstein_files_dir, processed_chunks_file)
    else:
        chunks = processor.process_documents(epstein_files_dir, processed_chunks_file)

    # Load chunks if we skipped processing
    if 'chunks' not in locals():
        chunks = DocumentProcessor.load_processed_chunks(processed_chunks_file)

    console.print(f"[green]✓[/green] Processed {len(chunks):,} chunks")

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
        console.print(f"[green]✓[/green] Vector index built with {stats['total_chunks']:,} chunks")

    except Exception as e:
        console.print(f"[bold red]Error building index:[/bold red] {e}")
        sys.exit(1)

    # Success message
    console.print(Panel(
        "[bold green]Initialization Complete![/bold green]\n\n"
        "The chatbot is now ready to use.\n\n"
        "Start the chatbot with:\n"
        "[yellow]./run_chatbot.sh[/yellow]\n"
        "or\n"
        "[yellow]python -m src.chatbot[/yellow]",
        border_style="green"
    ))


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python -m scripts.initialize_background [OPTIONS]")
        print("\nOptions:")
        print("  --ocr           Enable OCR for better text extraction (slower)")
        print("  --status        Check current processing status")
        print("  --help, -h      Show this help message")
        print("\nExamples:")
        print("  # Start/resume processing without OCR")
        print("  python -m scripts.initialize_background")
        print("\n  # Start/resume processing with OCR")
        print("  python -m scripts.initialize_background --ocr")
        print("\n  # Check status")
        print("  python -m scripts.initialize_background --status")
        print("\n  # Run in background (Linux/Mac)")
        print("  nohup python -m scripts.initialize_background > processing.log 2>&1 &")
        sys.exit(0)

    main()
