#!/bin/bash
# Quick start script for the Epstein Files chatbot

set -e

echo "Epstein Files RAG Chatbot"
echo "=========================="
echo ""

# Check if initialized
if [ ! -f "processed_chunks.json" ] || [ ! -d "chroma_db" ]; then
    echo "⚠️  Chatbot not initialized. Running initialization..."
    echo ""
    python3 -m scripts.initialize
    echo ""
fi

echo "Starting chatbot..."
echo ""
python3 -m src.chatbot
