"""Main CLI interface for the Epstein Files chatbot."""

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.rag import RAGChatbot


def print_help(console: Console):
    """Print help information."""
    help_text = """[bold cyan]Available Commands:[/bold cyan]

[yellow]/help[/yellow]    - Show this help message
[yellow]/quit[/yellow]    - Exit the chatbot
[yellow]/reset[/yellow]   - Clear conversation history
[yellow]/sources[/yellow] - Show sources from last answer
[yellow]/info[/yellow]    - Show system information

Just type your question to get started!"""

    console.print(Panel(help_text, title="Help", border_style="cyan"))


def print_welcome(console: Console):
    """Print welcome message."""
    welcome = """[bold cyan]Epstein Files RAG Chatbot[/bold cyan]

Ask questions about the Epstein Files documents.
Type [yellow]/help[/yellow] for available commands or just ask a question!"""

    console.print(Panel(welcome, border_style="cyan"))


def print_system_info(chatbot: RAGChatbot, console: Console):
    """Print system information."""
    info = chatbot.get_system_info()

    table = Table(title="System Information", show_header=True, header_style="bold cyan")
    table.add_column("Component", style="cyan")
    table.add_column("Details", style="white")

    # Vector Store
    vs_info = info.get("vector_store", {})
    table.add_row(
        "Vector Store",
        f"{vs_info.get('total_chunks', 0)} chunks indexed\n"
        f"Model: {vs_info.get('embedding_model', 'N/A')}\n"
        f"Dimensions: {vs_info.get('embedding_dimension', 'N/A')}"
    )

    # LLM
    llm_info = info.get("llm", {})
    table.add_row(
        "Language Model",
        f"{llm_info.get('model_name', 'N/A')}\n"
        f"Size: {llm_info.get('model_size', 'N/A')}\n"
        f"Quantization: {llm_info.get('quantization', 'N/A')}\n"
        f"GPU Layers: {llm_info.get('gpu_layers', 'N/A')}"
    )

    # Conversation
    table.add_row(
        "Conversation",
        f"{info.get('conversation_length', 0)} questions asked"
    )

    console.print(table)


def main():
    """Main entry point for the chatbot CLI."""
    console = Console()

    # Check if data is initialized
    project_root = Path(__file__).parent.parent
    chroma_db_path = project_root / "chroma_db"
    model_dir = project_root / "models"
    processed_chunks_file = project_root / "processed_chunks.json"

    if not processed_chunks_file.exists() or not chroma_db_path.exists():
        console.print(Panel(
            "[bold red]Error: Chatbot not initialized[/bold red]\n\n"
            "Please run the initialization script first:\n"
            "[yellow]python -m scripts.initialize[/yellow]\n\n"
            "This will process documents, download the model, and build the vector index.",
            border_style="red"
        ))
        sys.exit(1)

    # Initialize chatbot
    try:
        chatbot = RAGChatbot(chroma_db_path, model_dir)
        chatbot.initialize()
    except Exception as e:
        console.print(f"[bold red]Error initializing chatbot:[/bold red] {e}")
        sys.exit(1)

    print_welcome(console)
    console.print()

    # Main interaction loop
    while True:
        try:
            # Get user input
            question = Prompt.ask("\n[bold green]Your question[/bold green]").strip()

            if not question:
                continue

            # Handle commands
            if question.startswith("/"):
                command = question.lower()

                if command == "/quit" or command == "/exit":
                    console.print("\n[cyan]Goodbye![/cyan]")
                    break

                elif command == "/help":
                    print_help(console)

                elif command == "/reset":
                    chatbot.reset_conversation()

                elif command == "/sources":
                    sources = chatbot.get_last_sources()
                    if not sources:
                        console.print("[yellow]No sources available. Ask a question first.[/yellow]")
                    else:
                        console.print("\n[bold yellow]Sources from last answer:[/bold yellow]")
                        for i, source in enumerate(sources, 1):
                            metadata = source.get("metadata", {})
                            console.print(f"\n[bold cyan]Source {i}:[/bold cyan]")
                            console.print(f"File: {metadata.get('file_name', 'Unknown')}")
                            console.print(f"Page: {metadata.get('page_num', '?')}")
                            console.print(f"Dataset: {metadata.get('dataset', 'Unknown')}")
                            console.print(f"\n[dim]{source['text'][:300]}...[/dim]")

                elif command == "/info":
                    print_system_info(chatbot, console)

                else:
                    console.print(f"[red]Unknown command: {command}[/red]")
                    console.print("[yellow]Type /help for available commands[/yellow]")

                continue

            # Process question
            try:
                answer, sources = chatbot.answer_question(question)
                console.print()
                chatbot.format_answer_with_sources(answer, sources)

            except Exception as e:
                console.print(f"[bold red]Error processing question:[/bold red] {e}")

        except KeyboardInterrupt:
            console.print("\n\n[cyan]Use /quit to exit[/cyan]")
            continue

        except EOFError:
            console.print("\n[cyan]Goodbye![/cyan]")
            break


if __name__ == "__main__":
    main()
