#!/usr/bin/env python3
"""Test script to compare text extraction with and without OCR."""

import sys
from pathlib import Path
import fitz
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.document_processor_ocr import DocumentProcessorOCR
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


def test_extraction(pdf_path: Path):
    """Compare text extraction with and without OCR."""
    console = Console()

    console.print(f"\n[bold cyan]Testing: {pdf_path.name}[/bold cyan]\n")

    # Method 1: Standard extraction (current method)
    doc = fitz.open(pdf_path)
    total_text_standard = ""
    images_count = 0

    for page in doc:
        total_text_standard += page.get_text()
        images_count += len(page.get_images())

    doc.close()

    console.print(f"[yellow]Standard extraction:[/yellow]")
    console.print(f"  Images found: {images_count}")
    console.print(f"  Text extracted: {len(total_text_standard)} characters")
    console.print(f"  Preview: {total_text_standard[:200]}...")

    # Method 2: With OCR (if available)
    if OCR_AVAILABLE:
        console.print(f"\n[yellow]OCR extraction:[/yellow]")
        try:
            processor = DocumentProcessorOCR(use_ocr=True)
            result = processor.extract_text_from_pdf(pdf_path)

            if result:
                total_text_ocr = ""
                for page in result['pages']:
                    total_text_ocr += page['text']

                console.print(f"  Text extracted: {len(total_text_ocr)} characters")
                console.print(f"  Improvement: +{len(total_text_ocr) - len(total_text_standard)} characters ({(len(total_text_ocr) / max(len(total_text_standard), 1)):.1f}x)")
                console.print(f"  Preview: {total_text_ocr[:200]}...")
            else:
                console.print("  [red]Failed to extract[/red]")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
    else:
        console.print(f"\n[red]OCR not available. Install with:[/red]")
        console.print("  pip install pytesseract pillow")
        console.print("  sudo apt-get install tesseract-ocr  # Linux")


def main():
    """Test OCR on sample PDFs."""
    console = Console()

    console.print(Panel(
        "[bold cyan]OCR Extraction Test[/bold cyan]\n\n"
        "This script compares text extraction with and without OCR\n"
        "to show how much additional text can be recovered from images.",
        border_style="cyan"
    ))

    # Find some PDFs
    epstein_files_dir = Path("epstein_files")
    if not epstein_files_dir.exists():
        console.print("[red]Error: epstein_files directory not found[/red]")
        sys.exit(1)

    pdf_files = list(epstein_files_dir.rglob("*.pdf"))

    if not pdf_files:
        console.print("[red]No PDF files found[/red]")
        sys.exit(1)

    # Test first 3 PDFs
    console.print(f"\nFound {len(pdf_files)} PDFs. Testing first 3...\n")

    for pdf_path in pdf_files[:3]:
        test_extraction(pdf_path)
        console.print("\n" + "=" * 80 + "\n")

    # Summary
    console.print(Panel(
        "[bold green]To enable OCR for all PDFs:[/bold green]\n\n"
        "1. Install dependencies:\n"
        "   [yellow]pip install pytesseract pillow[/yellow]\n"
        "   [yellow]sudo apt-get install tesseract-ocr[/yellow]\n\n"
        "2. Run initialization with OCR:\n"
        "   [yellow]python -m scripts.initialize_with_ocr[/yellow]\n\n"
        "[dim]This will take longer but extract significantly more text[/dim]",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
