#!/bin/bash

# Background processing helper script for Epstein Files RAG

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Determine which Python to use
if [ -f ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
elif [ -n "$VIRTUAL_ENV" ]; then
    PYTHON="python3"
else
    PYTHON="python3"
fi

LOG_FILE="processing.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

show_help() {
    echo "Usage: ./run_processing.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start       Start processing in foreground"
    echo "  background  Start processing in background"
    echo "  status      Check processing status"
    echo "  logs        Follow processing logs (background mode only)"
    echo "  stop        Stop background processing"
    echo ""
    echo "Options:"
    echo "  --ocr       Enable OCR for better text extraction (slower)"
    echo ""
    echo "Examples:"
    echo "  ./run_processing.sh start"
    echo "  ./run_processing.sh background --ocr"
    echo "  ./run_processing.sh status"
    echo "  ./run_processing.sh logs"
}

check_status() {
    $PYTHON -m scripts.initialize_background --status
}

start_foreground() {
    OCR_FLAG=""
    if [[ "$1" == "--ocr" ]]; then
        OCR_FLAG="--ocr"
        echo -e "${CYAN}Starting processing with OCR enabled...${NC}"
    else
        echo -e "${CYAN}Starting processing (standard mode)...${NC}"
    fi

    $PYTHON -m scripts.initialize_background $OCR_FLAG
}

start_background() {
    OCR_FLAG=""
    if [[ "$1" == "--ocr" ]]; then
        OCR_FLAG="--ocr"
        echo -e "${CYAN}Starting background processing with OCR enabled...${NC}"
    else
        echo -e "${CYAN}Starting background processing (standard mode)...${NC}"
    fi

    # Check if already running
    if pgrep -f "scripts.initialize_background" > /dev/null; then
        echo -e "${YELLOW}Processing appears to be already running.${NC}"
        echo "Check with: ./run_processing.sh status"
        exit 1
    fi

    nohup $PYTHON -m scripts.initialize_background $OCR_FLAG > "$LOG_FILE" 2>&1 &
    PID=$!

    echo -e "${GREEN}Processing started in background (PID: $PID)${NC}"
    echo "Log file: $LOG_FILE"
    echo ""
    echo "Commands:"
    echo "  Check status: ./run_processing.sh status"
    echo "  View logs:    ./run_processing.sh logs"
    echo "  Stop:         ./run_processing.sh stop"
}

follow_logs() {
    if [[ ! -f "$LOG_FILE" ]]; then
        echo -e "${RED}Log file not found: $LOG_FILE${NC}"
        echo "Processing may not have started yet."
        exit 1
    fi

    echo -e "${CYAN}Following logs (Ctrl+C to exit)...${NC}"
    tail -f "$LOG_FILE"
}

stop_processing() {
    echo -e "${YELLOW}Stopping background processing...${NC}"

    if pgrep -f "scripts.initialize_background" > /dev/null; then
        pkill -SIGINT -f "scripts.initialize_background"
        echo -e "${GREEN}Stop signal sent. Processing will save checkpoint and exit.${NC}"
        echo "Check status with: ./run_processing.sh status"
    else
        echo -e "${YELLOW}No processing found running.${NC}"
    fi
}

# Main command dispatcher
case "${1:-}" in
    start)
        start_foreground "$2"
        ;;
    background|bg)
        start_background "$2"
        ;;
    status)
        check_status
        ;;
    logs|log)
        follow_logs
        ;;
    stop)
        stop_processing
        ;;
    --help|-h|help|"")
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
