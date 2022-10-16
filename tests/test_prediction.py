import pytest
from cigin_app.run import get_solv_free_energy, load_model
from tests.inputs import solute, solvent, delta_g, interact_map


def test_1():
    """Load model"""
    model = load_model()
    prediction, _ = get_solv_free_energy(model, solute, solvent)
    assert prediction == delta_g


def test_2():
    """Load model"""
    model = load_model()
    _, interaction_map = get_solv_free_energy(model, solute, solvent)
    assert interaction_map.tolist() == interact_map
