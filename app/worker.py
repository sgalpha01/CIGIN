import os
import queue
from celery import Celery
from rdkit import Chem
import torch
from model.model import get_graph_from_smile, model, device, data, key_attach

celery = Celery(__name__)
celery.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379")
celery.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379")

response = {}
def predictions(solute, solvent):
    # response.clear()
    m = Chem.MolFromSmiles(solute,sanitize=False)
    n = Chem.MolFromSmiles(solvent,sanitize=False)
    if (m == None or n == None):
      response['predictions']= 'invalid SMILES'
      print('invalid SMILES')
    else:
      mol = Chem.MolFromSmiles(solute)
      mol = Chem.AddHs(mol)
      solute = Chem.MolToSmiles(mol)
      solute_graph = get_graph_from_smile(solute)
      mol = Chem.MolFromSmiles(solvent)
      mol = Chem.AddHs(mol)
      solvent = Chem.MolToSmiles(mol)
      solvent_graph = get_graph_from_smile(solvent)
      delta_g, interaction_map =  model([solute_graph.to(device), solvent_graph.to(device)])
      interaction_map_one = torch.trunc(interaction_map)
      response["interaction_map"] = (interaction_map_one.detach().numpy()).tolist()
      response["predictions"] = delta_g.item()  

response_two = {}
def predictions_two(solute):
    response_two.clear()
    m = Chem.MolFromSmiles(solute,sanitize=False)
    if (m == None):
      response_two['predictions']= 'invalid SMILES'
      print('invalid SMILES')
    else:
        for i in data:
            delta_g, interaction_map =  model([get_graph_from_smile(Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(solute)))).to(device), get_graph_from_smile(Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(i)))).to(device)])
            response_two[i] = delta_g.item() 

def attach_drug_name():
    return {val:{k:v} for val,(k,v) in zip(key_attach, response_two.items())}


@celery.task(name='create_task')
def create_task(solute, solvent):
    predictions(solute, solvent)
    return {'result': response}

@celery.task(name='create_task_two')
def create_task_two(solute):
    predictions_two(solute)
    return {'result': attach_drug_name()}