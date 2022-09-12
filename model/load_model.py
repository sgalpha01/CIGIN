import json
import torch

# load the list 0f 1706 drug molecules
data_first = json.load(open('inputs/drug_set.json'))

# load model here 
state = torch.load('inputs/best_model.tar',map_location=torch.device('cpu'))
