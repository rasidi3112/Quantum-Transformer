#!/bin/bash
# Quantum Transformer Setup Script

set -e

echo "=========================================="
echo "Quantum Transformer Application Setup"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install package in development mode
echo "Installing Quantum Transformer..."
pip install -e ".[dev]"

# Install additional quantum backends (optional)
echo ""
echo "Install optional quantum backends?"
echo "1) IBM Qiskit"
echo "2) Amazon Braket"
echo "3) All backends"
echo "4) Skip"
read -p "Choice [4]: " choice

case $choice in
    1) pip install ".[ibm]" ;;
    2) pip install ".[braket]" ;;
    3) pip install ".[all]" ;;
    *) echo "Skipping optional backends" ;;
esac

# Run tests
echo ""
read -p "Run tests? [y/N]: " run_tests
if [[ $run_tests == "y" || $run_tests == "Y" ]]; then
    pytest tests/ -v
fi

echo ""
echo "=========================================="
echo "Installation complete!"
echo ""
echo "Activate environment: source venv/bin/activate"
echo "Run CLI: quantum-transformer --help"
echo "Run info: quantum-transformer info"
echo "=========================================="
