"""Test script to verify the chatbot setup."""

import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all required modules can be imported."""
    console = Console()
    console.print("\n[bold cyan]Testing Module Imports...[/bold cyan]\n")

    modules = [
        ("PyMuPDF (fitz)", "fitz"),
        ("sentence-transformers", "sentence_transformers"),
        ("ChromaDB", "chromadb"),
        ("Rich", "rich"),
        ("tqdm", "tqdm"),
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("requests", "requests"),
    ]

    all_ok = True
    for name, module in modules:
        try:
            __import__(module)
            console.print(f"[green]✓[/green] {name}")
        except ImportError as e:
            console.print(f"[red]✗[/red] {name}: {e}")
            all_ok = False

    # Test llama-cpp-python separately
    try:
        from llama_cpp import Llama
        console.print(f"[green]✓[/green] llama-cpp-python")

        # Check for CUDA support
        try:
            import torch
            if torch.cuda.is_available():
                console.print(f"  [green]CUDA available: {torch.cuda.get_device_name(0)}[/green]")
            else:
                console.print(f"  [yellow]CUDA not available - will use CPU[/yellow]")
        except:
            pass

    except ImportError as e:
        console.print(f"[red]✗[/red] llama-cpp-python: {e}")
        console.print(f"  [yellow]Install with: CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python[/yellow]")
        all_ok = False

    return all_ok


def test_file_structure():
    """Test that required files and directories exist."""
    console = Console()
    console.print("\n[bold cyan]Testing File Structure...[/bold cyan]\n")

    project_root = Path(__file__).parent.parent

    required_files = [
        "src/__init__.py",
        "src/document_processor.py",
        "src/vector_store.py",
        "src/llm.py",
        "src/rag.py",
        "src/chatbot.py",
        "scripts/__init__.py",
        "scripts/initialize.py",
        "pyproject.toml",
        ".python-version",
        "README_CHATBOT.md",
    ]

    required_dirs = [
        "src",
        "scripts",
        "epstein_files",
    ]

    all_ok = True

    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            console.print(f"[green]✓[/green] {file_path}")
        else:
            console.print(f"[red]✗[/red] {file_path} - missing")
            all_ok = False

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            console.print(f"[green]✓[/green] {dir_path}/")
        else:
            console.print(f"[red]✗[/red] {dir_path}/ - missing")
            all_ok = False

    return all_ok


def test_pdf_files():
    """Test that PDF files are present."""
    console = Console()
    console.print("\n[bold cyan]Testing PDF Files...[/bold cyan]\n")

    project_root = Path(__file__).parent.parent
    epstein_files_dir = project_root / "epstein_files"

    if not epstein_files_dir.exists():
        console.print(f"[red]✗[/red] epstein_files directory not found")
        return False

    pdf_files = list(epstein_files_dir.rglob("*.pdf"))
    console.print(f"[green]✓[/green] Found {len(pdf_files)} PDF files")

    if len(pdf_files) == 0:
        console.print(f"[yellow]⚠[/yellow] No PDF files found - run download script first")
        return False

    # Show datasets
    datasets = set()
    for pdf in pdf_files:
        datasets.add(pdf.parent.name)

    console.print(f"[cyan]Datasets found:[/cyan] {', '.join(sorted(datasets))}")

    return True


def test_initialization_status():
    """Check if chatbot has been initialized."""
    console = Console()
    console.print("\n[bold cyan]Checking Initialization Status...[/bold cyan]\n")

    project_root = Path(__file__).parent.parent

    checks = [
        ("Processed chunks", project_root / "processed_chunks.json"),
        ("ChromaDB", project_root / "chroma_db"),
        ("Models directory", project_root / "models"),
    ]

    for name, path in checks:
        if path.exists():
            if path.is_file():
                size = path.stat().st_size / (1024 * 1024)
                console.print(f"[green]✓[/green] {name}: {size:.2f} MB")
            else:
                console.print(f"[green]✓[/green] {name}: exists")
        else:
            console.print(f"[yellow]○[/yellow] {name}: not initialized")

    # Check for model file
    model_file = project_root / "models" / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    if model_file.exists():
        size = model_file.stat().st_size / (1024 ** 3)
        console.print(f"[green]✓[/green] Model file: {size:.2f} GB")
    else:
        console.print(f"[yellow]○[/yellow] Model file: not downloaded")


def main():
    """Run all tests."""
    console = Console()

    console.print("\n[bold]Epstein Files Chatbot - Setup Test[/bold]\n")

    tests = [
        ("Module Imports", test_imports),
        ("File Structure", test_file_structure),
        ("PDF Files", test_pdf_files),
    ]

    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))

    # Always show initialization status
    test_initialization_status()

    # Summary
    console.print("\n[bold cyan]Test Summary[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Test")
    table.add_column("Status")

    for name, result in results:
        status = "[green]PASS[/green]" if result else "[red]FAIL[/red]"
        table.add_row(name, status)

    console.print(table)

    all_passed = all(result for _, result in results)

    if all_passed:
        console.print("\n[bold green]All tests passed![/bold green]")
        console.print("\n[cyan]Next steps:[/cyan]")
        console.print("1. Run initialization: [yellow]python -m scripts.initialize[/yellow]")
        console.print("2. Start chatbot: [yellow]python -m src.chatbot[/yellow]")
    else:
        console.print("\n[bold red]Some tests failed. Please fix the issues above.[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
