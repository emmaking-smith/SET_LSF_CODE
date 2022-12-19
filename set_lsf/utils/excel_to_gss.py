'''
This module is used to process the data to make it the standardized csv file,
ready for loading into the GNN module. Specifically, this module makes the
txt files for the Glasgow Subgraph Solver (GSS).

Commands for running the GSS along with the sed commands to clean up the
GSS output are included in the comments at the end of main().
'''
import argparse
import pandas as pd
from rdkit import Chem
import os
from pathlib import Path
def init_args():
     '''
     Setting up the input arguments.
     '''
     parser = argparse.ArgumentParser()
     parser.add_argument('--data_path', '-d', type=str)
     parser.add_argument('--save_path', '-s', type=str)
    # parser.add_argument('--gss_path', '-gss', type=str)
     return parser.parse_args()

def standardize_smiles(smiles):
    '''
    RDKit's canonicalization of SMILES strings.
    Args:
        smiles (str): The SMILES string.
    Retunrs:
        smiles (str): The canonical SMILES string.
    '''
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    return smiles

def atom_list(smiles):
    '''
    Making an atom list for the subgraph solver.
    Args:
        smiles (str): A SMILES string.
    Returns:
        file (list): A list-form version of the
        atom features section of the GSS file.
    '''
    mol = Chem.MolFromSmiles(smiles)
    file = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        line = str(atom.GetIdx()) + ',,' + str(symbol) + '\n'
        file.append(line)
    return file

def edge_list(smiles):
    '''
    Making the edge list for the subgraph solver.
    Args:
        smiles (str): A SMILES string.
    Returns:
        file (list): A list-form version of the
        edges section of the GSS file.
    '''
    mol = Chem.MolFromSmiles(smiles)

    file = []

    for i in range(len(mol.GetAtoms())):
        for j in range(len(mol.GetAtoms())):
            e_ij = mol.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                line = str(min(i,j)) + ',' + str(max(i,j)) + ',' + str(e_ij.GetBondType()) + '\n'
                if line not in file:
                    file.append(line)
    return file

def main():
    args = init_args()
    data_path = args.data_path
    save_path = args.save_path
    save_path = os.path.join(save_path, 'molecules')
    df = pd.read_excel(data_path)

    for i in range(len(df)):
        # Canonicalizing the SMILES strings - must be done for consistent mapping!
        sm = standardize_smiles(df.loc[i, 'reactant'])
        if pd.isna(df.loc[i, 'product']) == True or df.loc[i, 'product'] == 'NO STRUCTURE':
            p = standardize_smiles(df.loc[i, 'reactant'])
        else:
            p = standardize_smiles(df.loc[i, 'product'])

        # For GSS to work the subgraph (SM) must be the same size or smaller than the graph (P),
        # so the naming convention might be reversed.
        if len(Chem.MolFromSmiles(sm).GetAtoms()) > len(Chem.MolFromSmiles(p).GetAtoms()):
            temp = sm
            sm = p
            p = temp
        #path_sm = os.path.join('/gpfs/workspace/users/kingse01/gss/glasgow-subgraph-solver/molecules', str(i), str(i) + '_sm.txt')
        #path_p = os.path.join('/gpfs/workspace/users/kingse01/gss/glasgow-subgraph-solver/molecules', str(i), str(i) + '_p.txt')
        path_sm = os.path.join(save_path, str(i), str(i) + '_sm.txt')
        path_p = os.path.join(save_path, str(i), str(i) + '_p.txt')

        # Making the directories for the molecules based on index.
        Path(os.path.split(path_sm)[0]).mkdir(parents=True, exist_ok=True)

        edge_sm = edge_list(sm)
        atom_sm = atom_list(sm)
        combined_sm = edge_sm + atom_sm

        # Writing to file. Note that the subgraph (SM) must be the same size or smaller than the graph (P), so
        # the naming convention might be reversed.
        with open(path_sm, 'w+') as f:
            for line in combined_sm:
                f.write(line)

        edge_p = edge_list(p)
        atom_p = atom_list(p)
        combined_p = edge_p + atom_p

        # Writing to file.
        with open(path_p, 'w+') as f:
            for line in combined_p:
                f.write(line)

#### BASH COMMANDS ####
# export LD_LIBRARY_PATH=/home/kingse01/workspace/miniconda3/lib:$LD_LIBRARY_PATH (PATH TO LIBRARY)
# Generating the mapping: cd into save_path/molecules
    # for d in ./*/ ; do (cd "$d" && index=$(echo $d | cut -d '/' -f 2) && /gpfs/workspace/users/kingse01/gss/glasgow-subgraph-solver/glasgow_subgraph_solver --print-all-solutions --format csv $index'_sm.txt' $index'_p.txt' > $index'_map.txt') ; done
# Making the comma separated file:
    # for d in ./*/ ; do (cd "$d" && index=$(echo $d | cut -d '/' -f 2) && sed -n -e '/mapping/p' $index'_map.txt' | sed 's/->/,/g' | sed 's/mapping \=//g' | sed 's/[(]//g' | sed 's/[)]/\n/g' > $index'_map_df.csv') ; done
if __name__ == '__main__':
    main()