import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rdkit import Chem

from cigin_app.models import Cigin

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def predictions(solute, solvent):
    response = {}
    solute = solute.upper()
    solvent = solvent.upper()
    mol = Chem.MolFromSmiles(solute)
    mol = Chem.AddHs(mol)
    solute = Chem.MolToSmiles(mol)

    mol = Chem.MolFromSmiles(solvent)
    mol = Chem.AddHs(mol)
    solvent = Chem.MolToSmiles(mol)

    model = load_model()

    delta_g, interaction_map = model(solute, solvent)
    response["interaction_map"] = (interaction_map.detach().numpy()).tolist()
    response["solvation"] = delta_g.item()
    return response


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/prediction")
def prediction(solute, solvent):
    try:
        response = predictions(solute, solvent)
    except Exception as err:
        print(err)
        response = {"error": "Invalid input"}
        response["interaction_map"] = []
        response["solvation"] = "No response due to invalid input"
    return {"prediction": response}
