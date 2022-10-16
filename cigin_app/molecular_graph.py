"""Helper code to create molecular graph"""

import warnings
from collections import OrderedDict

import numpy as np
import torch
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdMolDescriptors as rdDesc

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog("rdApp.error")
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def one_of_k_encoding(feature, allowable_set):
    """Function to get one hot encoding"""

    if feature not in allowable_set:
        raise Exception(f"input {feature} not in allowable set{allowable_set}")
    return list(map(lambda s: feature == s, allowable_set))


def one_of_k_encoding_unk(feature, allowable_set):
    """Maps inputs not in the allowable set to the last element."""

    if feature not in allowable_set:
        feature = allowable_set[-1]

    return list(map(lambda s: feature == s, allowable_set))


def bond_features(bond, use_chirality=True, bond_length=None):
    """Bond level features from rdkit bond object"""

    bont_type = bond.GetBondType()
    bond_feats = [
        bont_type == Chem.rdchem.BondType.SINGLE,
        bont_type == Chem.rdchem.BondType.DOUBLE,
        bont_type == Chem.rdchem.BondType.TRIPLE,
        bont_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]
    if bond_length is not None:
        bond_feats = bond_feats + [bond_length]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
        )
    return np.array(bond_feats)


def atom_features(atom, stereo, features, bool_id_feat=False, explicit_h=False):
    """Atom level features from rdkit's atom object"""
    if bool_id_feat:
        return np.array([])

    results = (
        one_of_k_encoding_unk(
            atom.GetSymbol(), ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "Si"]
        )
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4])
        + one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1])
        + one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
        + one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1])
        + one_of_k_encoding_unk(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
            ],
        )
        + [int(i) for i in list(f"{features:06b}")]
    )

    if not explicit_h:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    try:
        results = (
            results
            + one_of_k_encoding_unk(stereo, ["R", "S"])
            + [atom.HasProp("_ChiralityPossible")]
        )
    except Exception:
        results = results + [False, False] + [atom.HasProp("_ChiralityPossible")]

    return np.array(results)


def construct_molecular_graph(molecule):

    """Constructs molecular graph from rdkit's molecule object"""

    edges = OrderedDict({})
    nodes = OrderedDict({})

    molecule = Chem.MolFromSmiles(molecule)
    stereo = Chem.FindMolChiralCenters(molecule)
    features = rdDesc.GetFeatureInvariants(molecule)
    chiral_centers = [0] * molecule.GetNumAtoms()
    for i in stereo:
        chiral_centers[i[0]] = i[1]
    for i in range(0, molecule.GetNumAtoms()):
        atom_i = molecule.GetAtomWithIdx(i)
        nodes[i] = torch.FloatTensor(
            atom_features(atom_i, chiral_centers[i], features[i]).astype(np.float64)
        ).to(DEVICE)
        for j in range(0, molecule.GetNumAtoms()):
            e_ij = molecule.GetBondBetweenAtoms(i, j)

            if e_ij is not None:
                e_ij = map(
                    lambda feature: 1 if feature is True else 0,
                    bond_features(e_ij).tolist(),
                )  # ADDED edge feat
                e_ij = torch.FloatTensor(list(e_ij)).to(DEVICE)
                # atom_j = molecule.GetAtomWithIdx(j)
                if i not in edges:
                    edges[i] = []
                edges[i].append((e_ij, j))

    return edges, nodes
