'''
The setup of the adjacency matrices, atomic feature vectors, and reaction feature vectors.

BE SURE TO BE USING NETWORKX VER. 1.11!!

'''
import os
import numpy as np
import networkx as nx
import torch
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from torch.utils.data import Dataset, DataLoader


def rxn_encoding(df_row, unique_reagents, unique_oxidants, unique_solvents, unique_acids, unique_additives):
    '''
    Creating the reaction vector one-hot encoding. The current information
    we are one-hot encoding is: solvent, additive, oxidant, acid, radical mechanism.
        Args:
            df_row:
                The row (sliced with loc/iloc) of the train/test/val dataframe.

            unique_reagents (list of strings):
                A list which contains all the unique reagents possible in the dataframe.

            unique_solvents (list of strings):
                A list which contains all the unique solvents possible in the dataframe.

            unique_additives (list of strings):
                A list which contains all the unique additives possible in the dataframe.

            unique_oxidants (list of strings):
                A list which contains all the unique oxidants possible in the dataframe.

            unique_acids (list of strings):
                A list which contains all the unique acids possible in the dataframe.

        Returns:
            rxn (list):
                A one-hot encoded representation of the reaction.
                Size = 1 x rxn features length
    '''
    # Empty reaction vector.
    rxn = []
    # One hot encoding the reagents.
    rxn += [int(df_row['reagent'] == x) for x in unique_reagents]
    # One hot encoding the oxidants.
    rxn += [int(df_row['oxidant'] == x) for x in unique_oxidants]
    # One hot encoding the solvents.
    rxn += [int(df_row['solvent'] == x) for x in unique_solvents]
    # One hot encoding the acids.
    rxn += [int(df_row['acid'] == x) for x in unique_acids]
    # One hot encoding the additives.
    rxn += [int(df_row['additive'] == x) for x in unique_additives]
    return rxn

def make_graph(df_row):
    '''
    Creating the adjacency matricies for each bond type.
        Args:
            df_row:
                The row (sliced with loc/iloc) of the train/test/val dataframe.

        Returns:
            g_single (networkx graph):
                The single bond adjaceny matrix for the molecule.
            g_double (networkx graph):
                The double bond adjaceny matrix for the molecule.
            g_triple (networkx graph):
                The triple bond adjaceny matrix for the molecule.
            g_aromatic (networkx graph):
                The aromatic bond adjaceny matrix for the molecule.
    '''
    smiles = df_row['reactant']
    m = Chem.MolFromSmiles(smiles)

    # RDKit nonsense for finding the features of our molecule.
    fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    feats = factory.GetFeaturesForMol(m)

    g = nx.Graph()

    # Create nodes for each bond type.
    for i, atom in enumerate(m.GetAtoms()):
        # Note that RDKit goes in order of atom index, which is constant.

        g.add_node(i, a_type=atom.GetSymbol(), a_num=atom.GetAtomicNum(), acceptor=0, donor=0,
                   aromatic=atom.GetIsAromatic(), hybridization=atom.GetHybridization(),
                   num_h=atom.GetTotalNumHs())
    # Here 'Donor' and 'Acceptor' refer to hydrogen bond donors or acceptors.
    for i in range(len(feats)):
        if feats[i].GetFamily() == 'Donor':
            node_list = feats[i].GetAtomIds()
            for i in node_list:
                g.node[i]['donor'] = 1
        elif feats[i].GetFamily() == 'Acceptor':
            node_list = feats[i].GetAtomIds()
            for i in node_list:
                g.node[i]['acceptor'] = 1

    g_single = g.copy()
    g_double = g.copy()
    g_triple = g.copy()
    g_aromatic = g.copy()

    # Read Edges - no distances, split g based on bond type
    for i in range(len(m.GetAtoms())):
        for j in range(len(m.GetAtoms())):
            e_ij = m.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                if e_ij.GetBondType() == Chem.rdchem.BondType.SINGLE:
                    # Add bond to single.
                    g_single.add_edge(i, j, b_type=e_ij.GetBondType())

                elif e_ij.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                     # Add bond to double.
                     g_double.add_edge(i, j, b_type=e_ij.GetBondType())

                elif e_ij.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                    # Add bond to triple.
                    g_triple.add_edge(i, j, b_type=e_ij.GetBondType())

                elif e_ij.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                    # Add bond to aromatic.
                    g_aromatic.add_edge(i, j, b_type=e_ij.GetBondType())
            else:
                pass
    return g_single, g_double, g_triple, g_aromatic

def get_h(graph, longest_molecule, atom_list):
    '''
    Finding the starting atomic features vectors.
        Args:
            graph (networkx graph):
                A graph generated by make_g().

            longest_molecule (int):
                The length of the longest molecule in the entire pre-split dataframe.

            atom_list (list):
                The list of unique atomis in the entire pre-split dataframe.

        Returns:
            h (list):
                A list of atomic features for each atom in a molecule padded
                with 0's to the largest molecule.
    '''
    h = []
    # Generating the hidden features vector for each atom in the molecule
    for n, d in graph.nodes_iter(data=True):
        h_t = []
        h_t += [int(d['a_type'] == x) for x in atom_list]
        # Atomic number
        h_t.append(d['a_num'])
        # Acceptor
        h_t.append(d['acceptor'])
        # Donor
        h_t.append(d['donor'])
        # Aromatic
        h_t.append(int(d['aromatic']))
        # Hybradization
        h_t += [int(d['hybridization'] == x) for x in
                [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3]]
        # Implicit number of hydrogens
        h_t.append(d['num_h'])
        # Universal node one-hot.
        h_t.append(0)
        h.append(h_t)
    # Appending the universal node as the n+1th node.
    #h.append([0] * (len(h[0]) -1) + [1] )
    # Padding h with 0's to match longest molecule size.
    if len(h) < longest_molecule:
        atom_discrepancy = longest_molecule - len(h)
        for i in range(atom_discrepancy):
            h.append([0] * (len(h[0])))
    return h # h now padded to longest molecule size.

def universal_node_graph(molecule_size, longest_molecule):
    '''
    Generates the adjacency matrix for the universal node - the
    node which has a bond to every other node (atom) in the graph
    to help deal with long-range interactions.
        Args:
            molecule_size (int):
                The number of heavy atoms in the molecule. If a molecule
                has n atoms then node n+1 is the universal atom. Note that
                this means that everything must be padded to longest_molecule + 1.
            longest_molecule (int):
                The length everything needs to be padded to, technically longest molecule + 1.
        Returns:
            g_universal (numpy array):
                The adjacency matrix for the universal node padded out to the longest molecule + 1.
    '''
    g_universal = np.zeros(longest_molecule*longest_molecule).reshape((longest_molecule, longest_molecule))
    for i in range(molecule_size):
        g_universal[i, molecule_size] = 1
        g_universal[molecule_size, i] = 1

    return g_universal



def graph_padding(graph, longest_molecule):
    '''
    Padding an adjacency matrix to N x N where N is the size of the largest molecule.
        Args:
            graph (numpy array):
                An adjacency matrix of a molecule after being reshaped and converted to a
                numpy array.

            longest_molecule (int):
                The length of the longest molecule in the entire pre-split dataframe.

        Returns:
            graph (numpy array):
                An adjacency matrix of a molecule padded out with 0's to longest molecule.
    '''
    g_length = len(graph)
    if g_length < longest_molecule:
        padding = longest_molecule - g_length
        # Forming the zero column we will add. Note that we need to make it
        # an array with dimensions len(graph) x 1
        col_padding = np.zeros([g_length, 1])

        # Appending the column padding vector to the graph padding times.
        padding_counter = 1
        while padding_counter <= padding:
            graph = np.append(graph, col_padding, axis=1)
            padding_counter += 1

        # Repeating the above for the rows. Note that the graph dimensions
        # should now be M x N where N is the longest molecule size.
        row_padding = np.zeros([1, longest_molecule])

        padding_counter = 1
        while padding_counter <= padding:
            graph = np.append(graph, row_padding, axis=0)
            padding_counter += 1
    else:
        pass
    return graph

def rxn_padding(rxn_vector, longest_molecule, nha):
    '''
    Padding of the rxn vector to [longest molecule x rxn features length].
    Note that we will not be adding the features vector to the padded atoms, only the
    atoms that are really in the molecule.
        Args:
            rxn_vector (list):
                A list with the rxn features vector for each molecule as determined by rxn_encoding().

            longest_molecule (int):
                The length of the longest molecule in the entire pre-split dataframe.

            nha (int):
                The number of heavy atoms in the corresponding molecule.

        Returns:
            padded_rxn (numpy array):
                An array of size longest molecule x reaction features length.
    '''
    padding = longest_molecule - nha
    padded_rxn = np.tile(rxn_vector, nha)

    # Adding the zeros if the molecule is shorter than longest molecule
    if padding != 0:
        zeros_padding = np.zeros([padding * len(rxn_vector)])
        padded_rxn = np.append(padded_rxn, zeros_padding, axis=0)

    # Reshaping the 1D array.
    padded_rxn = padded_rxn.reshape([longest_molecule, len(rxn_vector)])

    return padded_rxn

class Pfizer_Dataset(Dataset):
    '''
    A class for the Pfizer data as a dataframe.
        Args:
            df (dataframe):
                The training/testing/validation dataframe

            longest_molecule (int):
                The size of the longest molecule in the entire pre-split dataframe.

            atom_list (list of strings):
                A list of all the unique atoms in every molecule in the entire
                pre-split dataframe.

            unique_reagents (list of strings):
                A list which contains all the unique reagents possible in the dataframe.

            unique_solvents (list of strings):
                A list which contains all the unique solvents possible in the dataframe.

            unique_additives (list of strings):
                A list which contains all the unique additives possible in the dataframe.

            unique_oxidants (list of strings):
                A list which contains all the unique oxidants possible in the dataframe.

            unique_acids (list of strings):
                A list which contains all the unique acids possible in the dataframe.

        Returns:
            (g, h, rxn_vector), (Y_weights, Y) (tuple of tensors):
                Set up of model inputs and model outputs.
    '''
    def __init__(self, df, longest_molecule, atom_list, unique_reagents, unique_oxidants,
                 unique_solvents, unique_acids, unique_additives):
        self.df = df
        self.longest_molecule = longest_molecule
        self.atom_list = atom_list
        self.unique_reagents = unique_reagents
        self.unique_solvents = unique_solvents
        self.unique_additives = unique_additives
        self.unique_oxidants = unique_oxidants
        self.unique_acids = unique_acids

    def __len__(self):
        # length = number of rows
        return len(self.df)

    def __getitem__(self, index):
        # Retrieving Y from dataframe.
        Y = self.df.loc[index, 'Y']

        # Generating the adjacency matrices.
        g_single, g_double, g_triple, g_aromatic = make_graph(self.df.iloc[index])

        nha = self.df.loc[index, 'nha']
        g_universal = universal_node_graph(nha, self.longest_molecule).reshape((1, self.longest_molecule, self.longest_molecule))

        # Making h. Note that h is already padded to longest molecule.
        h = get_h(g_single, self.longest_molecule, self.atom_list)

        # Reshaping the graphs and turning them into numpy arrays.
        g = np.concatenate((nx.to_numpy_matrix(g_single), nx.to_numpy_matrix(g_double), nx.to_numpy_matrix(g_triple),
                            nx.to_numpy_matrix(g_aromatic)), axis=0)
        g = np.array(g).reshape([-1, nx.to_numpy_matrix(g_single).shape[0], nx.to_numpy_matrix(g_single).shape[1]])

        # Adding the graph padding.
        g_pad = np.zeros([g.shape[0], self.longest_molecule, self.longest_molecule])
        for i in range(g.shape[0]): # g.shape[0] = number of bond types.
            g_pad[i] = graph_padding(g[i], self.longest_molecule)

        # Finding the rxn features vector.
        rxn_vector = rxn_encoding(self.df.iloc[index], self.unique_reagents, self.unique_oxidants, self.unique_solvents,
                                  self.unique_acids, self.unique_additives)

        rxn_vector = rxn_padding(rxn_vector, self.longest_molecule, nha)

        # Converting everything to torch tensors.
        g = torch.tensor(g_pad)
        g_universal = torch.tensor(g_universal)
        g = torch.cat((g, g_universal), axis=0)

        h = torch.tensor(h)
        rxn_vector = torch.tensor(rxn_vector)
        nha = torch.tensor([nha])
        Y = torch.tensor(Y)

        # Must return 2 things - a tuple and a target.
        return (g, h, rxn_vector, nha), (Y)


def collate_y(batch):
    '''
    Collating function for DataLoader().
    We want the output of the Y_weights and Y to be of the form N x 1 where N = length of all Y_weights or Y in that batch.
    We want the output of all other tensors to be batch size x dimensions of input.
        Args:
            batch (tuple: inputs, target):
                The batch that the DataLoader loads.
        Returns:
            batch (de-tupled):
                Un-tupled g, h, rxn_vector with correct dimensions.
    '''
    # Decoupling the tuples.
    g, h, rxn_vector, nha = batch[0][0]
    Y = batch[0][1]

    batch_size = len(batch)

    for i in range(1, batch_size):
        # Concatonating all the g's together.
        g = torch.cat([g, batch[i][0][0]], 0)
        # Concatonating all the h's together.
        h = torch.cat([h, batch[i][0][1]], 0)
        # Concatonating all the rxn_vectors's together.
        rxn_vector = torch.cat([rxn_vector, batch[i][0][2]], 0)
        # Concatonating all the nha's together.
        nha = torch.cat([nha, batch[i][0][3]], 0)
        # Concatonating all the Y's together.
        Y = torch.cat([Y, batch[i][1]], 0)

    # Reshaping the g's to be of format batch size x num bond types x longest molecule x longest molecule.
    # Note that g has been formated so that the number of columns = longest molecule
    g = g.view([batch_size, -1, g.size()[1], g.size()[1]])
    # Reshaping the h's to be of format batch size x longest molecule x len h features (number of columns).
    h = h.view([batch_size, g.size()[2], -1])
    # Reshaping the rxn_vector's to be of format batch size x longest molecule x len rxn features
    rxn_vector = rxn_vector.view([batch_size, g.size()[2], -1])
    # Reshaping the nha's to be one long tensor.
    nha = nha.view([-1])
    # Reshaping the Y's to be one long tensor.
    Y = Y.view([-1])

    return g, h, rxn_vector, nha, Y