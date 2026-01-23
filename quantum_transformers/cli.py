import argparse
import sys

def main():

    parser = argparse.ArgumentParser(
        prog="qgenesis",
        description="Quantum Transformer: Pure Quantum Architecture for Molecular Intelligence",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument(
        "--molecules",
        nargs="+",
        default=["H2", "LiH"],
        help="Molecules to benchmark",
    )
    bench_parser.add_argument(
        "--output",
        default="./benchmark_results",
        help="Output directory",
    )

    train_parser = subparsers.add_parser("train", help="Train Quantum Transformer")
    train_parser.add_argument(
        "--config",
        required=True,
        help="Path to config YAML",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs",
    )
    train_parser.add_argument(
        "--n-qubits",
        type=int,
        default=4,
        help="Number of qubits",
    )
    train_parser.add_argument(
        "--n-layers",
        type=int,
        default=4,
        help="Number of transformer layers",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict molecular properties")
    predict_parser.add_argument(
        "--smiles",
        required=True,
        help="SMILES string of molecule",
    )
    predict_parser.add_argument(
        "--checkpoint",
        help="Model checkpoint path",
    )

    info_parser = subparsers.add_parser("info", help="Show package info")

    args = parser.parse_args()

    if args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "train":
        run_training(args)
    elif args.command == "predict":
        run_prediction(args)
    elif args.command == "info":
        show_info()
    else:
        parser.print_help()

def run_benchmark(args):

    print(f"Running Quantum Transformer benchmarks for: {args.molecules}")
    print(f"Output: {args.output}")

def run_training(args):

    import yaml

    print(f"Loading config from: {args.config}")
    print(f"Training Quantum Transformer for {args.epochs} epochs")
    print(f"  - Qubits: {args.n_qubits}")
    print(f"  - Layers: {args.n_layers}")

def run_prediction(args):

    print(f"Predicting properties for: {args.smiles}")

    from quantum_transformers.molecular import SMILESTokenizer

    tokenizer = SMILESTokenizer()
    tokens = tokenizer.tokenize(args.smiles)
    print(f"Tokens: {tokens}")

def show_info():

    from quantum_transformers import get_info

    info = get_info()

    print("\n" + "="*60)
    print("Quantum Transformer: Pure Quantum Architecture for Molecular Intelligence")
    print("="*60)

    for key, value in info.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for v in value:
                print(f"    - {v}")
        else:
            print(f"  {key}: {value}")

    print("="*60 + "\n")

if __name__ == "__main__":
    main()
