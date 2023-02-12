'''
Forming the features vector with the fukui indices.
'''
import numpy as np

def add_fukuis(df_row, h, longest_molecule):
    '''
    Adding in the fukui indices for each atom in features vector.
    '''
    h = np.array(h).reshape([longest_molecule, -1])
    padded_electro_fukuis, padded_nucleo_fukuis, padded_rad_fukuis = pad_fukuis(df_row, longest_molecule)
    padded_electro_fukuis = padded_electro_fukuis.reshape([-1, 1])
    padded_nucleo_fukuis = padded_nucleo_fukuis.reshape([-1, 1])
    padded_rad_fukuis = padded_rad_fukuis.reshape([-1, 1])
    h = np.concatenate((h, padded_electro_fukuis), axis=1)
    h = np.concatenate((h, padded_nucleo_fukuis), axis=1)
    h = np.concatenate((h, padded_rad_fukuis), axis=1)
    return h

def pad_fukuis(df_row, longest_molecule):
    '''
    Padding out fukui columns to be correct length for concatenation.
    '''
    electro_fukuis = np.array(df_row['electro_fukui'])
    nucleo_fukuis = np.array(df_row['nucleo_fukui'])
    rad_fukuis = np.array(df_row['rad_fukui'])
    padding_needed = np.zeros([longest_molecule - len(electro_fukuis)])
    padded_electro_fukuis = np.concatenate((electro_fukuis, padding_needed))
    padded_nucleo_fukuis = np.concatenate((nucleo_fukuis, padding_needed))
    padded_rad_fukuis = np.concatenate((rad_fukuis, padding_needed))
    return padded_electro_fukuis, padded_nucleo_fukuis, padded_rad_fukuis

