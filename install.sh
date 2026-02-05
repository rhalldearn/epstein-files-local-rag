#!/bin/bash
# Automated installation script for Epstein Files RAG Chatbot

set -e

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Epstein Files RAG Chatbot - Installer   ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════╝${NC}"
echo ""

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
if ! command -v python3.11 &> /dev/null; then
    echo -e "${RED}✗ Python 3.11+ not found${NC}"
    echo "Please install Python 3.11 or higher"
    exit 1
fi
echo -e "${GREEN}✓ Python found: $(python3.11 --version)${NC}"

# Check NVIDIA GPU
echo -e "\n${YELLOW}Checking for NVIDIA GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    echo -e "${GREEN}✓ GPU found: $GPU_INFO${NC}"
else
    echo -e "${YELLOW}⚠ nvidia-smi not found - GPU acceleration may not work${NC}"
fi

# Choose installation method
echo -e "\n${CYAN}Choose installation method:${NC}"
echo "1) UV (Recommended - faster)"
echo "2) pip (Traditional)"
read -p "Enter choice [1-2]: " CHOICE

if [ "$CHOICE" = "1" ]; then
    echo -e "\n${YELLOW}Installing with UV...${NC}"

    # Check if UV is installed
    if ! command -v uv &> /dev/null; then
        echo -e "${YELLOW}Installing UV...${NC}"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    fi

    echo -e "${GREEN}✓ UV installed${NC}"

    # Install dependencies
    echo -e "\n${YELLOW}Installing dependencies...${NC}"
    uv sync

    # Install llama-cpp-python with CUDA
    echo -e "\n${YELLOW}Installing llama-cpp-python with CUDA support...${NC}"
    echo -e "${CYAN}This may take several minutes...${NC}"
    CMAKE_ARGS="-DLLAMA_CUDA=on" uv pip install llama-cpp-python --force-reinstall --no-cache-dir

elif [ "$CHOICE" = "2" ]; then
    echo -e "\n${YELLOW}Installing with pip...${NC}"

    # Create virtual environment
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python3.11 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    echo -e "${GREEN}✓ Virtual environment activated${NC}"

    # Install dependencies
    echo -e "\n${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements-chatbot.txt

    # Install llama-cpp-python with CUDA
    echo -e "\n${YELLOW}Installing llama-cpp-python with CUDA support...${NC}"
    echo -e "${CYAN}This may take several minutes...${NC}"
    CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

else
    echo -e "${RED}Invalid choice${NC}"
    exit 1
fi

# Run setup test
echo -e "\n${YELLOW}Running setup verification...${NC}"
python -m scripts.test_setup

# Check if initialization is needed
if [ ! -f "processed_chunks.json" ] || [ ! -d "chroma_db" ]; then
    echo -e "\n${YELLOW}Initialization required${NC}"
    echo "This will:"
    echo "  - Process 525 PDF files"
    echo "  - Download Llama 3.2 3B model (~2GB)"
    echo "  - Build vector search index"
    echo ""
    read -p "Run initialization now? (y/n): " INIT_CHOICE

    if [ "$INIT_CHOICE" = "y" ] || [ "$INIT_CHOICE" = "Y" ]; then
        echo -e "\n${YELLOW}Starting initialization...${NC}"
        echo -e "${CYAN}This will take 5-10 minutes${NC}"
        python -m scripts.initialize
    else
        echo -e "\n${YELLOW}Initialization skipped${NC}"
        echo "Run later with: ${CYAN}python -m scripts.initialize${NC}"
    fi
fi

# Success message
echo -e "\n${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         Installation Complete!             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}To start the chatbot:${NC}"
echo -e "  ${YELLOW}./run_chatbot.sh${NC}"
echo ""
echo -e "or"
echo ""
echo -e "  ${YELLOW}python -m src.chatbot${NC}"
echo ""
echo -e "${CYAN}For help:${NC}"
echo -e "  ${YELLOW}cat QUICKSTART.md${NC}"
echo ""
