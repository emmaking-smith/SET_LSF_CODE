'''
The setup functions for running the NMR Pretraining.

Do things improve when standardizing it by subtracting the mean and dividing by the standard deviation?
'''

import numpy as np
from rdkit import Chem
import os
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import networkx as nx
from collections import Counter
import torch
from torch.utils.data import Dataset

def get_max_ppm(df):
    '''
    Finding the maximum ppm value in the dataset for scaling
        Args:
            df (dataframe):
                The un-split dataframe.

        Returns:
            max_ppm (float):
                The maximum ppm value.
    '''
    ppm_values = []

    for i in range(len(df)):
        ppm_values = ppm_values + list(df.loc[i, 'value'].values())

    max_ppm = max(ppm_values)
    return max_ppm

def get_mean_ppm(df):
    '''
    Finding the mean ppm value in the dataset for scaling
        Args:
            df (dataframe):
                The un-split dataframe.

        Returns:
            mean_ppm (float):
                The mean ppm value.
    '''
    ppm_values = []

    for i in range(len(df)):
        ppm_values = ppm_values + list(df.loc[i, 'value'].values())

    mean_ppm = np.mean(ppm_values)
    return mean_ppm

def get_ppm_std(df):
    '''
    Finding the ppm standard deviation in the dataset for scaling
        Args:
            df (dataframe):
                The un-split dataframe.

        Returns:
            mean_ppm (float):
                The mean ppm value.
    '''
    ppm_values = []

    for i in range(len(df)):
        ppm_values = ppm_values + list(df.loc[i, 'value'].values())

    ppm_std = np.std(ppm_values)
    return ppm_std

def atomic_symbols(df_row):
    '''
    Finding the atomic symbols for a given molecule.
        Args:
            df_row (dataframe row):
                A sliced row in the dataframe.

        Returns:
            atom_ids (list):
                The atomic symbols for the atom in question.
    '''
    atom_ids = []
    mol = Chem.RemoveHs(df_row['rdmol'])
    for atom in mol.GetAtoms():
        atom_ids.append(atom.GetSymbol())
    return(atom_ids)

def total_atom_types(df):
    '''
    Finding the unique atom types in the dataset.
        Args:
            df (dataframe):
                The un-split dataframe.

        Returns:
            unique_atoms (list):
                A sorted list of the unique atom types where
                the most common atom is entry 0.
    '''
    all_atoms = []
    # Create one long list of all the atoms in every molecule
    for atom_list in df['atom_id']:
        all_atoms = all_atoms + atom_list
    # Take only the unique atoms.
    ranked_atoms = Counter(all_atoms).most_common()
    unique_atoms = []
    for i in range(len(ranked_atoms)):
        unique_atoms.append(ranked_atoms[i][0])
    return unique_atoms

def atom_keys(df, molecule_id):
    '''
    Find the total atom keys used in the
    dictionary across duplicated entries.

    Args:
        df (dataframe):
            The dataframe with the duplicate entries.

        molecule_id (int):
            The molecule_id that corresponds to
            the id with multiple spectra entries

    Returns:
        keys (list):
            The unique list of atom numbers (dict keys) for the molecule of interest.
    '''
    duplicates = df.loc[df['molecule_id'] == molecule_id, ['value']]
    keys = []

    for i in range(len(duplicates)):
        keys_i = list(duplicates.iloc[i][0][0].keys())
        for k in keys_i:
            if k not in keys:
                keys.append(k)

    return keys

def average_nmr_shifts(df, molecule_id):
    '''
    In the case that a molecule has multiple associated spectra,
    average the ppm shift for each atom.

    Args:
        df (dataframe):
            The dataframe with the duplicate entries.

        molecule_id (int):
            The molecule_id that corresponds to
            the id with multiple spectra entries

    Returns:
        new_ppm_values (list of dict):
            A list containing the dictionary of averaged
            ppm shifts, tiled appropriately to replace
            all the entries in the dataframe.
    '''

    duplicates = df.loc[df['molecule_id'] == molecule_id, ['value']]
    # Find the associated spectra.
    atoms_numbers = atom_keys(df, molecule_id)

    new_ppm_values = []
    ppm_dict = {}

    # Averaging the ppm values for each spectra key.
    for key in atoms_numbers:
        average_index_values = []
        for i in range(len(duplicates)):
            try:
                key_attempt = duplicates.iloc[i][0][0][key]
            except:
                pass
            else:
                average_index_values.append(key_attempt)

        ppm_dict[key] = np.mean(average_index_values)

    # Appending new values to a list for later storage into a dataframe.
    new_ppm_values.append(ppm_dict)

    return np.tile(new_ppm_values, len(duplicates))

def remove_duplicates(df, molecule_id):
    '''
    Once the duplicates have been updated via average_nmr_shifts(),
    remove all but one entry.

    Args:
        df (dataframe):
            The dataframe with the duplicate entries.

        molecule_id (int):
            The molecule_id that corresponds to
            the id with multiple spectra entries

    Returns:
        df (dataframe):
            The dataframe without duplicate entries.
    '''

    # Finding the indicies.
    duplicate_idxs = df.loc[df['molecule_id'] == molecule_id].index

    # Drop all but one.
    df = df.drop(duplicate_idxs[1:])

    return df

def get_num_heavy_atoms(df, universal_node=True):
    '''
    Finding the longest molecule in the dataframe.
    If we are using a uni node, the longest molecule
    goes to longest molecule + 1.

    Args:
        df (dataframe):
            The dataframe from NMR paper.
        universal_node (bool):
            If a universal node being used or not.

    Returns:
        df (dataframe):
            The dataframe with an nha (number of heavy atoms)
            column.

        longest_molecule (int):
            The size of the longest molecule.
    '''
    nha = []

    for i in range(len(df)):
        m = Chem.RemoveHs(df.loc[i, 'rdmol'])
        nha.append(len(m.GetAtoms()))

    df['nha'] = nha
    longest_molecule = max(nha)

    if universal_node == True:
        longest_molecule = longest_molecule + 1

    return df, longest_molecule

def total_atom_numbers(atom_list):
    unique_atom_num = []

    for smiles in atom_list:
        mol = Chem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            unique_atom_num.append(atom.GetAtomicNum())
    return(unique_atom_num)

def get_y(df_row, longest_molecule, mean_Y, Y_std):
    '''
    Finding the Y from a single dataframe row.
    Standardize by subtracting the mean and dividing by standard deviation.

    Args:
        df_row (dataframe):
            The dataframe row.

        longest_molecule (int):
            The longest molecule in the entire dataset.

        mean_Y (float):
            The mean ppm value in the entire dataset.

        Y_std (float):
            The ppm standard deviation of the entire dataset.

    Returns:
        Y (numpy array):
            The corresponding array of NMR ppm shifts. The entry is 0 if
            it is not a carbon atom.
    '''

    spectra = df_row['value']
    Y = np.zeros(longest_molecule)

    # Note that the spectra keys correspond to the atom index of the molecule.
    # The hydrogens are always indexed last.
    keys = list(spectra.keys())
    for k in keys:
        Y[k] = (spectra[k] - mean_Y) / (Y_std)

    return Y

def df_split(df, ratio):
    '''
    Splitting the dataframe into training/validation/testing sets.
    Note that each row is a unique molecule
        Args:
            df (dataframe):
                The dataframe of interest.
            ratio (float):
                The percentage to be left out.

        Returns:
            larger_df, smaller_df (dataframes):
                The larger and smaller dataframe after splits.
    '''
    df_idxs = df.index

    np.random.seed(0)
    df_idxs = np.random.permutation(df_idxs)
    num_drop_idxs = int(np.floor(len(df_idxs) * ratio))

    drop_idxs = df_idxs[0:num_drop_idxs]
    keep_idxs = df_idxs[num_drop_idxs:]

    larger_df = df.loc[keep_idxs]
    smaller_df = df.loc[drop_idxs]

    return larger_df, smaller_df

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
    m = Chem.RemoveHs(df_row['rdmol'])

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

    # Padding h with 0's to match longest molecule size.
    if len(h) < longest_molecule:
        atom_discrepancy = longest_molecule - len(h)
        for i in range(atom_discrepancy):
            h.append([0] * (len(h[0])))
    return h # h now padded to longest molecule size.

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

####### THE DATASET SETUP #######

class NMR_Dataset(Dataset):
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

        Returns:
            (g, h), (Y) (tuple of tensors):
                Set up of model inputs and model outputs.
    '''
    def __init__(self, df, longest_molecule, atom_list, ):
        self.df = df
        self.longest_molecule = longest_molecule
        self.atom_list = atom_list

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

        # Converting everything to torch tensors.
        g = torch.tensor(g_pad)
        g_universal = torch.tensor(g_universal)
        g = torch.cat((g, g_universal), axis=0)

        h = torch.tensor(h)
        Y = torch.tensor(Y)

        # Must return 2 things
        return (g, h), (Y)

def collate_nmr(batch):
    '''
    Collating function for DataLoader().
    We want the output of the Y_weights and Y to be of the form N x 1 where N = length of all Y_weights or Y in that batch.
    We want the output of all other tensors to be batch size x dimensions of input.
        Args:
            batch (tuple: inputs, target):
                The batch that the DataLoader loads.
        Returns:
            batch (de-tupled):
                Un-tupled g, h, Y with correct dimensions.
    '''
    # Decoupling the tuples.
    g, h = batch[0][0]
    Y = batch[0][1]

    batch_size = len(batch)

    for i in range(1, batch_size):
        # Concatonating all the g's together.
        g = torch.cat([g, batch[i][0][0]], 0)
        # Concatonating all the h's together.
        h = torch.cat([h, batch[i][0][1]], 0)
        # Concatonating all the Y's together.
        Y = torch.cat([Y, batch[i][1]], 0)

    # Reshaping the g's to be of format batch size x num bond types x longest molecule x longest molecule.
    # Note that g has been formated so that the number of columns = longest molecule
    g = g.view([batch_size, -1, g.size()[1], g.size()[1]])
    # Reshaping the h's to be of format batch size x longest molecule x len h features (number of columns).
    h = h.view([batch_size, g.size()[2], -1])
    # Reshaping the Y's to be batch size x 1.
    Y = Y.view([batch_size, -1])
    return g, h, Y