'''
The plug-and-play running new molecules on LSF. Input should be a dataframe with the correct columns.
'''

import pandas as pd
import numpy as np
import torch
import argparse

import utils.MPNN_setup as ms
import utils.NMR_Pretrain_Setup as nmr
import utils.df_setup as ds
from neural_nets.Final_LSF_Net import LSF_MPNN

device = "cuda" if torch.cuda.is_available() else "cpu"

def init_args():
    '''
    Setting up the input arguments.
    Takes the path to the pickle data and the save path for the
    test dataframe with predictions.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-d', type=str)
    parser.add_argument('--model_path', '-m', type=str)
    parser.add_argument('--save_path', '-s', type=str, default='./predictions.pickle')
    return parser.parse_args()

def truncate_Y(untruncated_y, nha, longest_molecule):
    truncated_ys = []
    for i in range(len(nha)):
        # What idx are we starting from?
        mol_start_counter = int(i * longest_molecule)

        # Get the atoms only in the molecule
        truncated_y_i = untruncated_y[mol_start_counter : mol_start_counter + int(nha[i])]
        truncated_ys.append(truncated_y_i)

    return truncated_ys

def scoring(true, pred):

    pred_simplified = np.zeros(len(pred))
    for i in range(len(pred)):
        if pred[i] > 0.5:
            pred_simplified[i] = 1
        else:
            pass

    true_p = np.mean(true) + 1e-10
    pred_p = np.mean(pred_simplified) + 1e-10

    tp = len(np.where(true + pred_simplified == 2)[0])
    tn = len(np.where(true + pred_simplified == 0)[0])
    fp = len(np.where(true - pred_simplified == -1)[0])
    fn = len(np.where(pred_simplified - true == -1)[0])

    fit = -tp * np.log(pred_p) - tn * np.log(1 - pred_p) + fp * np.log(true_p) + fn * np.log(1 - true_p)
    return fit

def main():

    args = init_args()
    data_path = args.data_path
    save_path = args.save_path
    model_path = args.model_path

    # The best pretrained model
    nmr_data_path = 'data/13C_nmrshiftdb.pickle'

    # Loading in the training data PURELY to generate the correct parameters for the model.
    # NO TRAINING IS OCCURING IN THIS MODULE.
    train_path = '/gpfs/workspace/users/kingse01/model_inputs/mpnn/old_gss_train_metallo.pickle'
    #train_path = 'data/retrospective_train.pickle'

    df = pd.read_pickle(data_path)
    train = pd.read_pickle(train_path)
    df = df.reset_index(drop=True)
    train = train.reset_index(drop=True)

    df = ds.num_atoms(df)

    longest_molecule = max(df['nha']) + 1 # adding plus one for the universal node.

    # Since we don't know what the true values are, we keep the Y's zero'd.
    Y = []
    for i in range(len(df)):
        label = np.zeros(longest_molecule)
        Y.append(label)
    df['Y'] = Y

    # Generating the atom symbols for each entry.
    atom_ids = []
    for i in range(len(df)):
        symbols = ds.atomic_symbols(df.iloc[i])
        atom_ids.append(symbols)
    df['atom_id'] = atom_ids

    nmr_df = pd.read_pickle(nmr_data_path)
    nmr_train = nmr_df['train_df']
    nmr_test = nmr_df['test_df']
    nmr_df = pd.concat((nmr_train, nmr_test)).reset_index(drop=True)

    atom_ids = []
    for i in range(len(nmr_df)):
        symbols = nmr.atomic_symbols(nmr_df.iloc[i])
        atom_ids.append(symbols)
    nmr_df['atom_id'] = atom_ids

    atom_list = nmr.total_atom_types(nmr_df)

    # Finding the complete list of unique reagents in the dataframe
    unique_reagents = train['reagent'].unique().tolist()

    # Finding the complete list of unique oxidants in the dataframe
    unique_oxidants = train['oxidant'].unique().tolist()

    # Finding the complete list of unique solvents in the dataframe
    unique_solvents = train['solvent'].unique().tolist()

    # Finding the complete list of unique acids in the dataframe
    unique_acids = train['acid'].unique().tolist()

    # Finding the complete list of unique additives in the dataframe
    unique_additives = train['additive'].unique().tolist()

    # Make sure all of the reagents/solvents/acids/etc. are in the training data.
    for reagent in df['reagent'].unique():
        if reagent not in unique_reagents:
            print(reagent, 'has never been seen by the model.')
            print('Please ensure that the reagent is spelled EXACTLY as it is spelled in the training data')
            print('Unique Reagents:')
            print(df['reagent'].unique())
            exit()

    for oxidant in df['oxidant'].unique():
        if oxidant not in unique_oxidants:
            print(oxidant, 'has never been seen by the model.')
            print('Please ensure that the oxidant is spelled EXACTLY as it is spelled in the training data')
            print('Unique Oxidants:')
            print(df['oxidant'].unique())
            exit()

    for acid in df['acid'].unique():
        if acid not in unique_acids:
            print(acid, 'has never been seen by the model.')
            print('Please ensure that the acid is spelled EXACTLY as it is spelled in the training data')
            print('Unique Acids:')
            print(df['acid'].unique())
            exit()

    for additive in df['additive'].unique():
        if additive not in unique_additives:
            print(additive, 'has never been seen by the model.')
            print('Please ensure that the additive is spelled EXACTLY as it is spelled in the training data')
            print('Unique Additives:')
            print(df['additive'].unique())
            exit()

    for solvent in df['solvent'].unique():
        if solvent not in unique_solvents:
            print(solvent, 'has never been seen by the model.')
            print('Please ensure that the solvent is spelled EXACTLY as it is spelled in the training data')
            print('Unique Solvents:')
            print(df['solvent'].unique())
            exit()

    df = df.reset_index(drop=True)

    preds = df.copy()

    # Setting up the Dataset.
    dataset_data = ms.Pfizer_Dataset(df, longest_molecule, atom_list, unique_reagents, unique_oxidants,
                                   unique_solvents, unique_acids, unique_additives)

    # Define the model.
    (g_0, h_t, rxn_vector_0, nha_0), not_important = dataset_data[0]

    message_size = 30
    message_passes = 3
    rxn_features_length = rxn_vector_0.size()[1]  # size of one hot encodings of reaction
    atom_num_list = nmr.total_atom_numbers(atom_list)

    model = LSF_MPNN(message_size, message_passes, atom_num_list, rxn_features_length, model_path)
    model.to(device)

    dataloader = ms.DataLoader(dataset_data, batch_size=len(preds), shuffle=False, collate_fn=ms.collate_y)

    model.eval()
    with torch.no_grad():
        for i, (g, h, rxn_vector, nha, Y) in enumerate(dataloader):
            # Note that shuffle is off for this dataloader so that results can be
            # appended to the preds dataframe in correct order.

            # Move the variables to device.
            g = g.to(device).float()
            h = h.to(device).float()
            rxn_vector = rxn_vector.to(device).float()
            nha = nha.to(device).float()

            # Run through the trained model.
            Y_pred = model(h, g, rxn_vector)

            Y_pred_no_dummy_atoms = truncate_Y(Y_pred.cpu().detach().numpy(), nha.cpu().detach().numpy(), longest_molecule)

    # Save the predictions and losses into dataframe for saving.
    preds['Y_pred'] = Y_pred_no_dummy_atoms
    preds.to_pickle(save_path)
    print('Predictions Complete. See', save_path, 'for results.')

if __name__ == '__main__':
    main()