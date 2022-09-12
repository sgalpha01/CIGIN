import json

def json_data_func(file):
    f = open(file)
    data = json.load(f)
    json_data =[]
    for key in data:
        json_data.append(data[key]['smiles'])    
    return json_data

