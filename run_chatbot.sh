#!/bin/bash
# Quick start script for the Epstein Files chatbot

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Find the right Python - prefer venv
if [ -f "$SCRIPT_DIR/.venv/bin/python3" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python3"
elif [ -f "$SCRIPT_DIR/venv/bin/python3" ]; then
    PYTHON="$SCRIPT_DIR/venv/bin/python3"
else
    PYTHON="python3"
fi

echo "Epstein Files RAG Chatbot"
echo "=========================="
echo ""
echo "Using Python: $PYTHON"
echo ""

# Check if initialized
if [ ! -f "processed_chunks.json" ] || [ ! -d "chroma_db" ]; then
    echo "⚠️  Chatbot not initialized. Running initialization..."
    echo ""
    $PYTHON -m scripts.initialize
    echo ""
fi

echo "Starting chatbot..."
echo ""
$PYTHON -m src.chatbot
