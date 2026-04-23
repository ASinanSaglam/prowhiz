"""Evaluate a trained model on the test set.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt \\
        --split data/splits/test.txt --processed data/processed/
    prowhiz-eval --help
"""
from prowhiz.cli.evaluate import main

if __name__ == "__main__":
    main()
