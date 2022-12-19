'''
The setup of the dataframe including the splits into training, testing,
and validation dataframes.

Credit to Dr. Pau Riba & Dr. Anjan Dutta for the original MPNN code.
'''

import numpy as np
import pandas as pd
from rdkit import Chem
from collections import Counter

########### DATAFRAME PREPARATION ###########
'''
Finding the reaction site atom indices. Note that the
index corresponds to not only the correct atomic position
in the SMILES when inputted into RDKit but also the correct
atomic line in the xyz file (NOT USING CURRENTLY).

This function is for a single compound.
    Args:
        csv_data (dataframe): The pre-split dataframe.
'''
def make_labels(pickle_data, longest_molecule):
    reaction_sites = pickle_data['parent_centres']

    # Preparing the atomic-level vector for the label.
    # Y = [y1, y2, ..., yN] where N is the number
    # of heavy atoms and i is the atomic index of the molecule
    # in RDKit (canonical). yi = 0 if the atom does not
    # react and 1 if it does.

    # Padding Y to longest molecule length.
    Y = np.zeros(longest_molecule)
    # Note that RDKit's atom indexing also begins on 0.
    # We make sure that we have some reaction sites.
    if len(reaction_sites) > 0:
    # if pd.isna(reaction_sites) == False:
        # Some formatting needs to happen with reaction sites
        # if the molecule actually reacts.
        for site in reaction_sites:
        # for site in reaction_sites.split():
            Y[int(site)] = 1
    return Y

'''
Calculating the number of reaction
sites for a Y and one-hot encoding it.
[y0, y1, y2, y3, y4] where y0 = 1 if the
molecule doesn't react, y1 = 1 if the molecule
has one active site, y2 = 1 if the molecule has
2 active sites, y3 = 1 if the molecule as 3
active sites, and y4 = 1 if the molecule has 4+
active sites.
    Args:
        df_row (dataframe row): A single row
                                of the input dataframe.
'''
def one_hot_Y(df_row):
    Y = df_row['Y']
    num_sites = sum(Y)
    if num_sites > 4:
        num_sites = 4
    num_sites = [int(num_sites == x) for x in [0,1,2,3,4]]

    return num_sites
'''
Finding the number of heavy atoms for every molecule
in the dataframe.
    Args:
        df (dataframe): The pre-split dataframe.
'''
def num_atoms(df):
    molecules = df['reactant']
    nha = []
    for m in molecules:
        m = Chem.MolFromSmiles(m)
        num_heavy_atoms = m.GetNumHeavyAtoms()
        nha.append(num_heavy_atoms)
    df['nha'] = nha
    return df

'''
Finding atomic symbols - To be used in generating the list of unique atoms.
    Args:
        df_row (dataframe): A single row in the pre-split dataframe.
'''
def atomic_symbols(df_row):
    atom_ids = []
    smiles = df_row['reactant']
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom_ids.append(atom.GetSymbol())
    return(atom_ids)

'''
Finding the list of unique atoms, sorted by most common.
    Args:
        df (dataframe): The pre-split dataframe.
'''
def total_atom_types(df):
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


'''
Finding the list of unique atomic numbers.
    Args:
        atom_list (list): The list of unique atomic symbols.
'''
def total_atom_numbers(atom_list):
    unique_atom_num = []

    for smiles in atom_list:
        mol = Chem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            unique_atom_num.append(atom.GetAtomicNum())
    return(unique_atom_num)

########### DATAFRAME SPLITS ###########
'''
Select n most common molecules to leave out for validation.
For simplicity, we are using the reactant_id to select molecules.
    Args:
        df (dataframe): The pre-split dataframe.

        n (int): The number of compounds to be left out for
                 testing.
'''
def test_molecules(df, n):
    # Only leaving out molecules who reacted.
    reacted_df = df[pd.isna(df['parent_centres']) == False]

    all_molecule_ids = []
    for id in reacted_df['reactant_id']:
        all_molecule_ids.append(id)

    # Finding the n most common ids in the list.
    ranking = Counter(all_molecule_ids).most_common()
    test_ids = []

    # Appending the n most common ids to the validation list.
    for i in range(n):
        test_ids.append(ranking[i][0])

    # Splitting the dataframe into the non-test and test dataframes.
    test_df = pd.DataFrame()
    for i in test_ids:
        # Taking out the entry(ies) which correspond to the validation id(s).
        test_df_i = df[df['reactant_id'] == i]
        test_df = pd.concat([test_df, test_df_i])
        df = df[df['reactant_id'] != i]

    return test_df, df

'''
A function that takes out an scaffold based on the a given
Pfizer index. Using Felix's compounds so indices are:
1965, 1954, 913 (no WuXi LSF), 1995, and 2004
    Args:
        df (dataframe): The pre-split dataframe.

        index (int): The index corresponding to the
                     Pfizer scaffold we want left
                     out for testing.
'''
def pfizer_test_mol(df, index):

    test_df = df[df['reactant_id'] == index]
    df = df[df['reactant_id'] != index]

    # 913 has A LOT of entries, most of which
    # don't react (WuXi LSF). Keeping only
    # Pfizer's internal data ('LSF Radical').
    if index == 913:

        test_df = test_df[test_df['workflow'] == 'LSF Radical']

    return test_df, df


'''
Splitting the dataset after the test molecules have been removed
into testing and validation datasets. Similar to Felix, we are only leaving
out a single compound. This is the most common molecule after the validation
molecule(s).
    Args:
        df (dataframe): The dataframe after going through test_molecules().
'''
def train_val_split(df):
    # Only leaving out molecules who reacted.
    reacted_df = df[pd.isna(df['parent_centres']) == False]
    all_molecule_ids = []
    for id in reacted_df['reactant_id']:
        all_molecule_ids.append(id)

    # Finding the (next) most common id in the list.
    # Most common id(s) are in validation dataframe.
    val_id = Counter(all_molecule_ids).most_common()[0][0]

    # Splitting the dataframe into the training and validation sets.
    val_df = df[ df['reactant_id'] == val_id ]
    train_df = df[ df['reactant_id'] != val_id ]

    return train_df, val_df


