#!/bin/bash
echo ""
echo " ============================================"
echo "  TeamSync — Student Compatibility Classifier"
echo " ============================================"
echo ""

cd "$(dirname "$0")"

echo "[1/3] Checking Python..."
if ! command -v python3 &>/dev/null; then
    echo " ERROR: python3 not found. Install from https://python.org"
    exit 1
fi

echo "[2/3] Installing dependencies..."
cd backend
pip3 install -r requirements.txt --quiet

echo "[3/3] Starting Flask server..."
echo ""
echo " Open your browser at: http://localhost:5000"
echo " Press Ctrl+C to stop."
echo ""
python3 app.py
