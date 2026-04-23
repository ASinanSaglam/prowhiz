"""Predict binding free energy (dG) for a single mmCIF file.

Usage:
    python scripts/predict.py --input path/to/complex.cif --ligand ATP --checkpoint best.pt
    prowhiz-predict --input 1abc.cif --ligand ATP --model-index 0

Run prowhiz-predict --help for full options.
"""
from prowhiz.cli.predict import main

if __name__ == "__main__":
    main()
