# CIGIN (chemically Interpretable Graph Interaction Network)

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10SZqCpc8wp0sUU-TGhhLGZ6Vmy3ihPg0?usp=sharing)

## Abstract

Solubility of drug molecules is related to pharmacokinetic properties such as absorption and distribution, which affects the amount of drug that is available in the body for its action Computational or experimental evaluation of solvation free energies of drug-like molecules solute that quantify solubilities is an arduous task and hence development of reliable computationally tractable models is sought after in drug discovery tasks in pharmaceutical industry. Here, we report a novel method based on graph neural network to predict solvation free energies. Previous studies considered only the solute for solvation free energy prediction and ignored the nature of the solvent, limiting their practical applicability. The proposed model is an end-to-end framework comprising three phases namely, message passing, interaction and prediction phases. In the first phase, message passing neural network was used to compute inter-atomic interaction within both solute and solvent molecules represented as molecular graphs. In the interaction phase, features from the preceding step is used to calculate a solute-solvent interaction map, since the solvation free energy depends on how (un)favorable the solute and solvent molecules interact with each other. The calculated interaction map that captures the solute-solvent interactions along with the features from the message passing phase is used to predict the solvation free energies in the final phase. The model predicts solvation free energies involving a large number of solvents with high accuracy. We also show that the interaction map captures the electronic and steric factors that govern the solubility of drug-like molecules and hence is chemically interpretable.

## Documentation

### File Directory

```
├── app
│   └─── __init__.py
│   └─── main.py
├── inputs
│   └─── best_model.tar
│   └─── drug_set.json
│   └─── sample_json_data.py
│   └─── test_input.py
│   └─── unwanted_smiles.py
├── model
│   └─── __init__.py
│   └─── load_model.py
│   └─── model.py
│   └─── predict_json.py
├── tests
│   └─── __init__.py
│   └─── test_solubility_json.py
│   └─── test_solubility.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── entrypoint.sh
├── gunicorn.config.py
├── Procfile
├── requirements.txt
├── README.md

```

- `App` folder contains `main.py` where we handle api calling and posts responses
- `inputs` folder contains all the inputs required for the model to work
- `model` folder contains `model.py` file where all the machine learning code is present. `load_model.py` contains the code to load the model. If you want to use a custom model load the model in `inputs` folder and provide the path in `load_model.py`. `predict_json.py` contains the code to load the load all the similes in `inputs/drug_set.json` file and converts them to do predictions.
- `tests` folder contains the `test_solubility_json.py` and `test_solubility.py` to do unit tests
- `Dockerfile` used to deploy this in a container
- `entrypoint.sh` conatins the starting command to start the fastapi in docker after deploying and run background workers
- `requirements.txt` contains all the required modules for this repository
