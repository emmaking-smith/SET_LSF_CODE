'''
Late Stage Functionalization predictions after the NMR Pretraining.

NOTE: WE ARE USING THE ALL ATOM GRU 2 LINEAR LAYER SETUP FOR THIS COMMIT!
'''

import os
import argparse
import pandas as pd
import numpy as np
import torch
import logging
from collections import Counter

from neural_nets.LSF_Finetune_Net import LSF_MPNN

import utils.NMR_Pretrain_Setup as nmr
from torch.utils.tensorboard import SummaryWriter
import utils.df_setup as ds
import utils.MPNN_setup as ms

def init_args():
    '''
    Setting up the input arguments.
    Takes the path to the pickle data and the save path for the
    test dataframe with predictions.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', '-train', type=str)
    parser.add_argument('--test_data_path', '-test', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--pretrain_model_path', '-m', type=str, default="trained_models/nmr_model")
    parser.add_argument('--nmr_data_path', '-nmr', type=str, default="data/13C_nmrshiftdb.pickle")
    parser.add_argument('--weight_hyperparam', '-w', type=float)
    return parser.parse_args()

def overpred_loss_weights(true, pred):
    pred_p = torch.mean(pred) + 1e-10
    true_p = torch.mean(true) + 1e-10

    tp = true * pred * torch.log(true_p)
    tn = (1 - true) * (1 - pred) * torch.log(1 - true_p)
    fp = (1 - true) * pred * torch.log(1 - pred_p)
    fn = true * (1-pred) * torch.log(pred_p)

    out = tp + tn + fp + fn
    return -out

def underpred_loss_weights(true, pred):
    pred_p = torch.mean(pred) + 1e-10
    true_p = torch.mean(true) + 1e-10

    tp = true * pred * torch.log(pred_p)
    tn = (1-true) * (1-pred) * torch.log(1 - pred_p)
    fp = (1-true) * pred * torch.log(true_p)
    fn = true * (1-pred) * torch.log(1 - true_p)

    out = tp + tn + fp + fn
    return -out

def main():
    args = init_args()
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    save_path = args.save_path
    pretrain_model_path = args.pretrain_model_path
    nmr_data_path = args.nmr_data_path
    
    # Set up Tensorboard SummaryWriter.
    writer = SummaryWriter()

    # Set up logger.
    log_path = os.path.join(save_path, 'model_log.log')
    logging.basicConfig(filename=log_path, format='%(asctime)s %(message)s', filemode='w')

    # Creating an object
    logger = logging.getLogger()

    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)

    # Reading in the standardized pickle file.
    df = pd.read_pickle(train_data_path)
    df = df.reset_index(drop=True)

    test_data = pd.read_pickle(test_data_path)
    test_data = test_data.reset_index(drop=True)

    logger.info("Data Loaded")

    # Defining device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Our device is %s", device)

    # Finding the number of heavy atoms for each reactant and the
    # longest molecule in the dataframe for paddings.
    df = ds.num_atoms(df)
    test_data = ds.num_atoms(test_data)

    # longest_molecule = max(df['nha'])
    longest_molecule = max(max(df['nha']), max(test_data['nha']))
    longest_molecule = longest_molecule + 1 # +1 now to accomodate the universal node in each molecule.

    Y = []
    for i in range(len(df)):
        label = ds.make_labels(df.iloc[i], longest_molecule)
        Y.append(label)
    df['Y'] = Y

    Y = []
    for i in range(len(test_data)):
        label = ds.make_labels(test_data.iloc[i], longest_molecule)
        Y.append(label)
    test_data['Y'] = Y

    # Generating the one-hot encoding for the number of reactive sites.
    num_sites = []
    for i in range(len(df)):
        encoding = ds.one_hot_Y(df.iloc[i])
        num_sites.append(encoding)
    df['num_sites'] = num_sites

    # Generating the atom symbols for each entry.
    atom_ids = []
    for i in range(len(df)):
        symbols = ds.atomic_symbols(df.iloc[i])
        atom_ids.append(symbols)
    df['atom_id'] = atom_ids

    atom_ids = []
    for i in range(len(test_data)):
        symbols = ds.atomic_symbols(test_data.iloc[i])
        atom_ids.append(symbols)
    test_data['atom_id'] = atom_ids

    # Use the atom_list from the NMR pretraining.
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
    unique_reagents = df['reagent'].unique().tolist()

    # Finding the complete list of unique oxidants in the dataframe
    unique_oxidants = df['oxidant'].unique().tolist()

    # Finding the complete list of unique solvents in the dataframe
    unique_solvents = df['solvent'].unique().tolist()

    # Finding the complete list of unique acids in the dataframe
    unique_acids = df['acid'].unique().tolist()

    # Finding the complete list of unique additives in the dataframe
    unique_additives = df['additive'].unique().tolist()

    # Re-indexing the split datasets because python is dumb.
    train_data, val_data = ds.train_val_split(df)

    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
   
    logger.info("Datsets have been split into training, validation, and testing datasets")
    preds = test_data.copy()

    train_data = ms.Pfizer_Dataset(train_data, longest_molecule, atom_list, unique_reagents, unique_oxidants,
                 unique_solvents, unique_acids, unique_additives)
    val_data = ms.Pfizer_Dataset(val_data, longest_molecule, atom_list, unique_reagents, unique_oxidants,
                 unique_solvents, unique_acids, unique_additives)
    test_data = ms.Pfizer_Dataset(test_data, longest_molecule, atom_list, unique_reagents, unique_oxidants,
                 unique_solvents, unique_acids, unique_additives)

    # Define the model.
    (g_0, h_t, rxn_vector_0), not_important = train_data[0]

    message_size = 30
    message_passes = 3
    rxn_features_length = rxn_vector_0.size()[1] # size of one hot encodings of reaction
    atom_num_list = nmr.total_atom_numbers(atom_list)

    model = LSF_MPNN(message_size, message_passes, atom_num_list, rxn_features_length, pretrain_model_path)

    logger.debug("Model Defined")
    logger.debug("message_size = %d, message_passes = %d, atom_num_list = %s, rxn_features_length = %d, pretrain_model_path = %s",
                 message_size, message_passes, atom_num_list, rxn_features_length, pretrain_model_path)

    del message_size, message_passes, rxn_features_length, atom_num_list, pretrain_model_path

    # Set up the DataLoader()
    dataloader_train = ms.DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=ms.collate_y)
    dataloader_val = ms.DataLoader(val_data, batch_size=64, shuffle=True, collate_fn=ms.collate_y)
    dataloader_test = ms.DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=ms.collate_y)
    
    # Define the learning rate.
    lr = 1e-3

    # Define the optimzer.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Define number of epochs
    epochs = 100

    # Epoch for loop.
    model.to(device).float()

    weight_hyp = args.weight_hyperparam

    for epoch in range(epochs):
        batch_train_loss = []
        batch_val_loss = []
        # Training
        model.train()
        for i, (g, h, rxn_vector, Y) in enumerate(dataloader_train):
            g = g.to(device).float()
            h = h.to(device).float()
            rxn_vector = rxn_vector.to(device).float()
            Y = Y.to(device).float()

            optimizer.zero_grad()

            # Run through the MPNN
            Y_pred = model(h, g, rxn_vector)

            under_w = underpred_loss_weights(Y, Y_pred.clone().detach())
            over_w = overpred_loss_weights(Y, Y_pred.clone().detach())
            w = (weight_hyp * under_w) + over_w

            train_loss = torch.nn.BCELoss(weight=w, reduction='sum')(Y_pred, Y)

            # Calculate gradients and propagate backwards for each loss.
            train_loss.backward()
            optimizer.step()
            batch_train_loss.append(train_loss.cpu().detach().numpy())

        # Finding the average batch train loss for the epoch.
        batch_train_loss = np.mean(batch_train_loss)
        logger.debug('Mean Train Loss at epoch %d is %.4f', epoch, batch_train_loss)

        # Adding that average loss to the tensorboard.
        writer.add_scalar('Train Loss / Epoch', batch_train_loss, epoch)
        writer.close()
        
        # Evaluating on validation set. Turn off gradients.
        model.eval()
        with torch.no_grad():
            for i, (g, h, rxn_vector, Y) in enumerate(dataloader_val):
                # Move the variables to device.
                g = g.to(device).float()
                h = h.to(device).float()
                rxn_vector = rxn_vector.to(device).float()
                Y = Y.to(device).float()

                # Run through the MPNN
                Y_pred = model(h, g, rxn_vector)

                under_w = underpred_loss_weights(Y, Y_pred.clone().detach())
                over_w = overpred_loss_weights(Y, Y_pred.clone().detach())
                w = (weight_hyp * under_w) + over_w

                val_loss = torch.nn.BCELoss(weight=w, reduction='sum')(Y_pred, Y)

                batch_val_loss.append(val_loss.cpu().detach().numpy())

        # Finding the average batch validation loss for the epoch.
        batch_val_loss = np.mean(batch_val_loss)
        logger.debug('Mean Val Loss at epoch %d is %.4f', epoch, batch_val_loss)
        # Adding that average loss to the tensorboard.
        writer.add_scalar('Validation Loss / Epoch', batch_val_loss, epoch)
        writer.close()

    # Testing on the test data after 100 epochs of training.
    logger.debug('Model Training Complete')

    test_preds = []
    test_losses = []

    model.eval()
    with torch.no_grad():
        for i, (g, h, rxn_vector, Y) in enumerate(dataloader_test):
            # Note that shuffle is off for this dataloader so that results can be
            # appended to the preds dataframe in correct order.
            batch_size = g.size()[0]
            # Move the variables to device.
            g = g.to(device).float()
            h = h.to(device).float()
            rxn_vector = rxn_vector.to(device).float()
            Y = Y.to(device).float()

            # Run through the MPNN.
            Y_pred = model(h, g, rxn_vector)

            Y_pred = Y_pred.view([batch_size, -1])
            Y = Y.view([batch_size, -1])

            under_w = underpred_loss_weights(Y, Y_pred.clone().detach())
            over_w = overpred_loss_weights(Y, Y_pred.clone().detach())
            w = (weight_hyp * under_w) + over_w
            w = w.view([batch_size, -1])

            # Adding the test losses and predictions to the predictions dataframe.
            for i in range(batch_size):
                test_preds.append(Y_pred[i].cpu().detach().numpy())
                test_loss = torch.nn.BCELoss(weight=w[i], reduction='sum')(Y_pred[i], Y[i])
                test_losses.append(test_loss.cpu().detach().numpy())

    # Save the predictions and losses into dataframe for saving.
    preds['Y_pred'] = test_preds
    preds['loss'] = test_losses

    preds_save_path = os.path.join(save_path, 'pred.pickle')
    preds.to_pickle(preds_save_path)

    # Save the model to save path.
    model_save_path = os.path.join(save_path, 'model')
    torch.save(model.state_dict(), model_save_path)

if __name__ == '__main__':
    main()


