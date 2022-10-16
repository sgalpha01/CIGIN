"""
CIGIN : Chemically Interpretable Graph Interaction Network
"""

import argparse

import sys


import torch

from rdkit import Chem
from cigin_app.models import Cigin


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_solv_free_energy(model, solute, solvent):
    """Get the solvation free energy"""

    solv_free_energy, interaction_map = model(solute, solvent)
    return solv_free_energy, interaction_map


def load_model():
    """Load the model"""

    model = Cigin().to(DEVICE)
    model.load_state_dict(torch.load("data/cigin.tar"))
    model.eval()
    return model


def arg_parser():
    """Read arguments from command line"""

    parser = argparse.ArgumentParser(
        description="""Cigin: A GNN-based model for predicting
        solvation free energies of small molecules"""
    )
    parser.add_argument("--solute", type=str, help="Solute SMILES", required=True)
    parser.add_argument(
        "--solvent", type=str, help="Solvent SMILES (defaults to water)", default="O"
    )
    return parser


def main():
    """Entypoint function"""
    parser = arg_parser()
    solute = parser.parse_args().solute
    solvent = parser.parse_args().solvent
    if not solute:
        print("Please provide a solute SMILES")
        sys.exit(0)
    if not solvent:
        solvent = "O"
    if not Chem.MolFromSmiles(solute):
        print(f"Invalid solute SMILES - {solute}")
        sys.exit(0)
    if not Chem.MolFromSmiles(solvent):
        print(f"Invalid solvent SMILES - {solvent}")
        sys.exit(0)

    model = load_model()

    try:
        delta_g, interaction_map = get_solv_free_energy(model, solute, solvent)
    except Exception as err:
        print(err)
        sys.exit(0)

    print("Predicted free energy of solvation: ", str(delta_g.item()))

    # Applying min max normalization across the interaction map
    min_value = min(interaction_map.flatten())
    max_value = max(interaction_map.flatten())
    interaction_map = (interaction_map - min_value) / (max_value - min_value)

if __name__ == "__main__":
    main()
