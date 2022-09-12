from httpx import AsyncClient
import pytest
from app.main import app
from model.model import predictions,response
from inputs.test_input import data, data_two_here, solute, solvent


@pytest.mark.anyio
async def test_root():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get(f"/predict?solute={solute}&solvent={solvent}")
        response_two = await ac.get('/predict_solubility')
    assert response.status_code == 200
    assert response_two.status_code == 200
    assert response.json() == data
    assert response_two.json() == data_two_here


async def test_predict():
    res = await predictions(solute, solvent)
    response_final = response
    assert response_final == data_two_here

    