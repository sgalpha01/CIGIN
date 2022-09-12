import pytest
from inputs.sample_json_data import data_two
from inputs.test_input import solvent, solute, data_two_here
from app.worker import response ,predictions, response_two, predictions_two
@pytest.mark.anyio

async def test_predict_json():
    res = await predictions_two(solute)
    response_final = response_two
    assert response_final == data_two

async def test_predict():
    res = await predictions(solute, solvent)
    response_final = response
    assert response_final == data_two_here