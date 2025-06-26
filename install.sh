#!/bin/bash

# Installation script for NarrateLM

echo "Installing NarrateLM dependencies..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is required but not installed."
    exit 1
fi

# Install dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Make the script executable
chmod +x narratelm.py

echo "Installation complete!"
echo ""
echo "Usage:"
echo "  python3 narratelm.py <epub_file> <output_directory> --api-key <your_gemini_api_key>"
echo ""
echo "Or set your API key as an environment variable:"
echo "  export GEMINI_API_KEY='your_api_key_here'"
echo "  python3 narratelm.py <epub_file> <output_directory>"
echo ""
echo "For more information, see README.md"
