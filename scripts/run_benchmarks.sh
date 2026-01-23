#!/bin/bash
# Run Quantum Transformer Benchmarks

set -e

echo "Quantum Transformer Benchmark Suite"
echo "=============================================="

# Default parameters
MOLECULES=${1:-"H2 LiH"}
N_QUBITS=${2:-4}
N_LAYERS=${3:-4}
OUTPUT_DIR="./experiments/benchmarks/results"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run benchmarks
echo "Molecules: $MOLECULES"
echo "Qubits: $N_QUBITS"
echo "Layers: $N_LAYERS"
echo "Output: $OUTPUT_DIR"
echo ""

python3 -c "
from experiments.benchmarks.benchmark_suite import BenchmarkSuite

suite = BenchmarkSuite(
    molecules='${MOLECULES}'.split(),
    output_dir='${OUTPUT_DIR}',
)

results = suite.run(
    n_qubits=${N_QUBITS},
    n_layers=${N_LAYERS},
    verbose=True,
)

print('\\n✓ Quantum Transformer benchmarks complete!')
print(f'Results saved to: ${OUTPUT_DIR}')
"
