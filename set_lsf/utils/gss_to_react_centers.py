'''
This module is used to process the data to make it the standardized csv file,
ready for loading into the GNN module. Specifically, this module takes the
csv files from the GSS and converts them to reactive sites.
'''

import numpy as np
import pandas as pd
import os
from rdkit import Chem
import logging
from excel_to_gss import standardize_smiles
import argparse

def init_args():
     '''
     Setting up the input arguments.
     '''
     parser = argparse.ArgumentParser()
     parser.add_argument('--data_path', '-d', type=str)
     parser.add_argument('--save_path', '-s', type=str)
     parser.add_argument('--log_path', '-log', type=str)
     parser.add_argument('--gss_directory_path', '-gss', type=str)
     return parser.parse_args()

def find_react_centers(df_row_number, directory, sm_smiles, p_smiles, reversed=False):
    '''
    Finds the GSS mapping file and to identify the reactive centers.
    Args:
        df_row_number (int): The row number corresponding to the index
        for the GSS file.

        directory (str): The path to the directories for all the GSS
        map files.

        sm_smiles (str): The starting material SMILES string.

        p_smiles (str): The product SMILES string.

        reversed (bool): False (default) if SM is a subgraph of P. True
        if P is a subgraph of SM (i.e. dealkylation).
    Returns:
        cleaned_react_centers (array): The reactive centers - includes symmetric centers.
    '''
    # Converting the SMILES strings to molecules.
    sm = Chem.MolFromSmiles(sm_smiles) # May not necessarily be SM if SM is larger than P.
    p = Chem.MolFromSmiles(p_smiles)
    cleaned_react_centers = []
    path = os.path.join(directory, str(df_row_number), str(df_row_number) + '_map_df.csv')

    # Note that some subgraph matches failed. Make sure to log those.
    if os.stat(path).st_size == 0:
        logger.debug('GSS Failed. Check %d.', df_row_number)
    else:
        # Loading the *_map_df.csv file
        mapping = pd.read_csv(path, header=None, sep=',')

        mapping = mapping.rename(columns={mapping.keys().tolist()[0]: 'SM', mapping.keys().tolist()[1]: 'P'})
        num_atoms = len(sm.GetAtoms())
        react_centers = {}
        total_mappings = int(len(mapping) / num_atoms)

        for map in range(total_mappings):
            centers_per_map = []
            mapping_i = mapping[(map*num_atoms):((map+1)*num_atoms)]
            mapping_i = mapping_i.reset_index(drop=True)
            for i in range(len(mapping_i)):
                # check number of neighbors for atom.
                degree_sm = len(sm.GetAtomWithIdx(int(mapping_i.loc[i, 'SM'])).GetNeighbors())
                degree_p = len(p.GetAtomWithIdx(int(mapping_i.loc[i, 'P'])).GetNeighbors())
                if degree_sm != degree_p:
                    # If SM truly is SM, then append SM atom index, but when SM has been
                    # swapped for P due to SM being larger than P, append the P atom index
                    # (the true SM atom index).
                    if reversed == False:
                        centers_per_map.append(mapping.loc[i, 'SM'])
                    else:
                        centers_per_map.append(mapping.loc[i, 'P'])
            if len(centers_per_map) > 0:
                react_centers[map] = centers_per_map

        # Finding the lowest number of reactive centers in all the mappings - everything above that will be considered to
        # be a less possible subgraph.
        if len(react_centers) > 0:
            num_react_sites = [len(react_centers[key]) for key in react_centers]
            min_num_react_sites = min(num_react_sites)
            for key in react_centers:
                if len(react_centers[key]) == min_num_react_sites:
                    cleaned_react_centers += react_centers[key]
            # Logging if different mappings give different number of reactive sites.
            if len(np.unique(num_react_sites)) > 1:
                logger.debug('REACTIVE SITE MISMATCH FROM GLASGOW GRAPH SOLVER. CHECK %d', df_row_number)
        cleaned_react_centers = np.unique(cleaned_react_centers)
    return cleaned_react_centers

def main():
    args = init_args()
    data_path = args.data_path
    save_path = args.save_path
    log_path = args.log_path # Path for the log file.
    gss_directory_path = args.gss_directory_path # Path pointing to all the subgraph'd molecules
    
    #log_path = '/gpfs/workspace/users/kingse01/gss/glasgow_map_manual_inspection.log'
    logging.basicConfig(filename=os.path.join(log_path, 'gss_log.log'), format='%(asctime)s %(message)s', filemode='w')
    # Creating an object
    logger = logging.getLogger()

    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)
    
    df = pd.read_excel(data_path)

    parent_centres_GSS = []
    canon_sms = []
    canon_ps = []
    for i in range(len(df)):
        #directory = os.path.join('/gpfs/workspace/users/kingse01/ggs/glasgow-subgraph-solver/molecules')
        sm = standardize_smiles(df.loc[i, 'reactant'])
        canon_sms.append(sm)
        if pd.isna(df.loc[i, 'product']) == True or df.loc[i, 'product'] == 'NO STRUCTURE':
            p = standardize_smiles(df.loc[i, 'reactant'])
        else:
            p = standardize_smiles(df.loc[i, 'product'])
        canon_ps.append(p)

        # For GSS to work the subgraph (SM) must be the same size or smaller than the graph (P),
        # so the naming convention might be reversed.

        if len(Chem.MolFromSmiles(sm).GetAtoms()) > len(Chem.MolFromSmiles(p).GetAtoms()):
            react_centers = find_react_centers(i, gss_directory_path, p, sm, reversed=True)
        else:
            react_centers = find_react_centers(i, gss_directory_path, sm, p, reversed=False)

        parent_centres_GSS.append(react_centers)
    df['parent_centres'] = parent_centres_GSS
    df['reactant'] = canon_sms
    df['product'] = canon_ps

    df.to_pickle(save_path)
   
if __name__ == '__main__':
    main()