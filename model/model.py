import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from dgl import DGLGraph
from dgl.nn.pytorch import Set2Set, NNConv, GATConv
from rdkit import Chem, RDLogger,rdBase
from rdkit.Chem import rdMolDescriptors as rdDesc
import numpy as np
import warnings
from inputs.unwanted_smiles import unwanted_smiles
from model.load_model import data_first, state

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class GatherModel(nn.Module):
    def __init__(self,
                 node_input_dim=42,
                 edge_input_dim=10,
                 node_hidden_dim=42,
                 edge_hidden_dim=128,
                 num_step_message_passing=6,
                 gather="mpnn",
                 n_heads=3):
        super(GatherModel, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        self.gather = gather
        self.set2set = Set2Set(node_hidden_dim, 2, 1) 
        if self.gather == "mpnn":
        	self.message_layer = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
	        edge_network = nn.Sequential(
	            nn.Linear(edge_input_dim, edge_hidden_dim), nn.ReLU(),
	            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim))
	        self.conv = NNConv(in_feats=node_hidden_dim,
	                           out_feats=node_hidden_dim,
	                           edge_func=edge_network,
	                           aggregator_type='sum',
                               residual=True
                                )
        	self.gru = nn.GRU(node_hidden_dim, node_hidden_dim)
        elif self.gather == "gat":
        	self.n_heads = n_heads  
        	self.gat =  GATConv(node_hidden_dim,node_hidden_dim,self.n_heads)

    def forward(self, g, n_feat, e_feat):

        init = n_feat.clone()
        out = F.relu(self.lin0(n_feat))
        if self.gather == "mpnn":
            h = out.unsqueeze(0)                           
            for i in range(self.num_step_message_passing):
                m = torch.relu(self.conv(g, out, e_feat))
                out = self.message_layer(torch.cat([m, out],dim=1))
            return out + init


class CIGINModel(nn.Module):
    
    def __init__(self,
                 node_input_dim=42,
                 edge_input_dim=10,
                 node_hidden_dim=42,
                 edge_hidden_dim=42,
                 num_step_message_passing=8,
                 interaction='dot',
                 gather='mpnn'):
        super(CIGINModel, self).__init__()
        
        self.node_input_dim = node_input_dim
        self.node_hidden_dim =  node_hidden_dim
        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_step_message_passing = num_step_message_passing
        self.gather = gather
        self.interaction = interaction

        self.solute_gather = GatherModel(self.node_input_dim,self.edge_input_dim,
                              self.node_hidden_dim,self.edge_input_dim,
                              self.num_step_message_passing, 
                              self.gather, 3)
        self.solvent_gather = GatherModel(self.node_input_dim,self.edge_input_dim,
                              self.node_hidden_dim,self.edge_input_dim,
                              self.num_step_message_passing, 
                              self.gather, 3)

        self.fc1 = nn.Linear(8*self.node_hidden_dim,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,1)
        
        self.imap = nn.Linear(80,1)
        self.num_step_set2set=2
        self.num_layer_set2set=1
        self.set2set_solute = Set2Set(2*node_hidden_dim, self.num_step_set2set, self.num_layer_set2set)
        self.set2set_solvent = Set2Set(2*node_hidden_dim, self.num_step_set2set, self.num_layer_set2set)


        
    def forward(self, data):

        solute = data[0]
        solvent = data[1]
        
        solute_features = self.solute_gather(solute, solute.ndata['x'].float(), solute.edata['w'].float())
        solvent_features = self.solvent_gather(solvent, solvent.ndata['x'].float(), solvent.edata['w'].float())

        if 'dot' not in self.interaction:
            X1 = solute_features.unsqueeze(0)
            Y1= solvent_features.unsqueeze(1)
            X2 = X1.repeat(solvent_features.shape[0],1,1)
            Y2 = Y1.repeat(1,solute_features.shape[0],1)
            Z = torch.cat([X2,Y2],-1)

            if self.interaction == 'general':
                interaction_map = self.imap(Z).squeeze(2)
            if self.interaction == 'tanh-general':
                interaction_map = torch.tanh(self.imap(Z)).squeeze(2)

            ret_interaction_map = torch.clone(interaction_map)

        elif 'dot' in self.interaction :
            interaction_map = torch.mm(solute_features, solvent_features.t())
            if 'scaled' in self.interaction:
                interaction_map = interaction_map/(np.sqrt(self.node_hidden_dim))

            ret_interaction_map = torch.clone(interaction_map)
            interaction_map = torch.tanh(interaction_map)
        
        solvent_prime = torch.mm(interaction_map.t(), solute_features)
        solute_prime = torch.mm(interaction_map, solvent_features)

        solute_features = torch.cat((solute_features, solute_prime), dim=1)
        solvent_features = torch.cat((solvent_features, solvent_prime), dim=1)
        
        solute_features = self.set2set_solute(solute, solute_features)
        solvent_features = self.set2set_solvent(solvent, solvent_features)

        final_features = torch.cat((solute_features,solvent_features),1)
        predictions = torch.relu(self.fc1(final_features))
        predictions = torch.relu(self.fc2(predictions))
        predictions =  self.fc3(predictions)
        return predictions, ret_interaction_map

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]        
    return list(map(lambda s: x == s, allowable_set))

def get_atom_features(atom, stereo, features, explicit_H=False):

    """
    Method that computes atom level features from rdkit atom object
    :param atom: rdkit atom object
    :return: atom features, 1d numpy array
    """
    # todo: take list  of all possible atoms
    possible_atoms = ['C','N','O','S','F','P','Cl','Br','I','Si']
    atom_features  = one_of_k_encoding_unk(atom.GetSymbol(),possible_atoms)
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
    atom_features += one_of_k_encoding(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) 
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D])
    atom_features += [int(i) for i in list("{0:06b}".format(features))]

    #todo: add aromacity,acceptor,donor and chirality
    if not explicit_H:
        atom_features += one_of_k_encoding_unk(atom.GetTotalNumHs(),[0, 1, 2, 3, 4])

    try:
        atom_features += one_of_k_encoding_unk(stereo,['R', 'S']) 
        atom_features += [atom.HasProp('_ChiralityPossible')]
    except Exception as e:
        
        atom_features +=  [False, False
                          ] + [atom.HasProp('_ChiralityPossible')]
        
    return np.array(atom_features)

def get_bond_features(bond):
    
    """
    Method that computes bond level features from rdkit bond object
    :param bond: rdkit bond object
    :return: bond features, 1d numpy array
    """
    
    bond_type = bond.GetBondType()
    bond_feats = [
      bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
      bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
      bond.GetIsConjugated(),
      bond.IsInRing()
    ]
    bond_feats += one_of_k_encoding_unk(str(bond.GetStereo()),["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])

    return np.array(bond_feats)

def get_graph_from_smile(molecule):
    
    """
    Method that constructs a molecular graph with nodes being the atoms
    and bonds being the edges.
    :param molecule: SMILE sequence
    :return: DGL graph object, Node features and Edge features
    """

    G = DGLGraph()
    molecule = Chem.MolFromSmiles(molecule)
    features = rdDesc.GetFeatureInvariants(molecule)
    
    stereo = Chem.FindMolChiralCenters(molecule)
    chiral_centers = [0]* molecule.GetNumAtoms()
    for i in stereo:
        chiral_centers[i[0]] = i[1]
        
    G.add_nodes(molecule.GetNumAtoms())
    node_features = []
    edge_features = []
    for i in range(molecule.GetNumAtoms()):

        atom_i = molecule.GetAtomWithIdx(i)
        atom_i_features =  get_atom_features(atom_i,chiral_centers[i],features[i])
        node_features.append(atom_i_features)
        
        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                G.add_edge(i,j)
                bond_features_ij = get_bond_features(bond_ij)
                edge_features.append(bond_features_ij)
                
    G.ndata['x'] = torch.FloatTensor(node_features)
    G.edata['w'] = torch.FloatTensor(edge_features)
    return G



def get_len_matrix(len_list):
    len_list = np.array(len_list)
    max_nodes = np.sum(len_list)
    curr_sum = 0
    len_matrix = []
    for l in len_list:
        curr = np.zeros(max_nodes)
        curr[curr_sum:curr_sum+l] = 1
        len_matrix.append(curr)
        curr_sum += l
    return np.array(len_matrix)
    
class Dataclass(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # solute_file = 'mol_files/'+self.dataset.loc[idx]['FileHandle'] +'.mol'
        # solute = Chem.MolFromMolFile(solute_file) 
        # solute=Chem.MolToSmiles(solute)
        solute = self.dataset.loc[idx]['SoluteSMILES']
        mol = Chem.MolFromSmiles(solute)
        mol = Chem.AddHs(mol)
        solute = Chem.MolToSmiles(mol)
        solute_graph = get_graph_from_smile(solute)
        
        solvent = self.dataset.loc[idx]['SolventSMILES']
        mol = Chem.MolFromSmiles(solvent)
        mol = Chem.AddHs(mol)
        solvent = Chem.MolToSmiles(mol)
        
        solvent_graph = get_graph_from_smile(solvent)
        delta_g = self.dataset.loc[idx]['DeltaGsolv']
        return [solute_graph, solvent_graph]


state["solute_gather.conv.edge_func.0.weight"] = state["solute_gather.conv.edge_nn.0.weight"]
state["solute_gather.conv.edge_func.0.bias"] = state["solute_gather.conv.edge_nn.0.bias"]
state["solute_gather.conv.edge_func.2.weight"] = state["solute_gather.conv.edge_nn.2.weight"]
state["solute_gather.conv.edge_func.2.bias"] = state["solute_gather.conv.edge_nn.2.bias"]
state["solvent_gather.conv.edge_func.0.weight"] = state["solvent_gather.conv.edge_nn.0.weight"]
state["solvent_gather.conv.edge_func.0.bias"] = state["solvent_gather.conv.edge_nn.0.bias"]
state["solvent_gather.conv.edge_func.2.weight"] = state["solvent_gather.conv.edge_nn.2.weight"]
state["solvent_gather.conv.edge_func.2.bias"] = state["solvent_gather.conv.edge_nn.2.bias"]
 
del state["solute_gather.conv.edge_nn.0.weight"]
del state["solute_gather.conv.edge_nn.0.bias"]
del state["solute_gather.conv.edge_nn.2.weight"]
del state["solute_gather.conv.edge_nn.2.bias"]
del state["solvent_gather.conv.edge_nn.0.weight"]
del state["solvent_gather.conv.edge_nn.0.bias"]
del state["solvent_gather.conv.edge_nn.2.weight"]
del state["solvent_gather.conv.edge_nn.2.bias"]

model= CIGINModel().to(device)
model.load_state_dict(state,strict=True)
model.eval()

# def attach_drug_name():
#     return {val:{k:v} for val,(k,v) in zip(key_attach, response_two.items())}

# response = {}
# async def predictions(solute, solvent):
#     response.clear()
#     m = Chem.MolFromSmiles(solute,sanitize=False)
#     n = Chem.MolFromSmiles(solvent,sanitize=False)
#     if (m == None or n == None):
#       response['predictions']= 'invalid SMILES'
#       print('invalid SMILES')
#     else:
#       mol = Chem.MolFromSmiles(solute)
#       mol = Chem.AddHs(mol)
#       solute = Chem.MolToSmiles(mol)
#       solute_graph = get_graph_from_smile(solute)
#       mol = Chem.MolFromSmiles(solvent)
#       mol = Chem.AddHs(mol)
#       solvent = Chem.MolToSmiles(mol)
#       solvent_graph = get_graph_from_smile(solvent)
#       delta_g, interaction_map =  model([solute_graph.to(device), solvent_graph.to(device)])
#       interaction_map_one = torch.trunc(interaction_map)
#       response["interaction_map"] = (interaction_map_one.detach().numpy()).tolist()
#       response["predictions"] = delta_g.item()


# response_two = {}
# async def predictions_two(solute):
#     response_two.clear()
#     m = Chem.MolFromSmiles(solute,sanitize=False)
#     if (m == None):
#       response_two['predictions']= 'invalid SMILES'
#       print('invalid SMILES')
#     else:
#         for i in data:
#             delta_g, interaction_map =  model([get_graph_from_smile(Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(solute)))).to(device), get_graph_from_smile(Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(i)))).to(device)])
#             response_two[i] = delta_g.item()

keys = []
for key in unwanted_smiles:
  keys.append(key)

data_two = dict([(key, val) for key, val in 
           data_first.items() if key not in keys])

data =[]
for key in data_two:
  data.append(data_two[key]['smiles'])  

key_attach = []
for key in data_two:
  key_attach.append(key)