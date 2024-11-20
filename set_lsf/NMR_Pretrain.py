import numpy as np
import pandas as pd
import utils.NMR_Pretrain_Setup as nmr
from neural_nets.NMR_4ll_Net import NMR_MPNN
from torch.utils.data import DataLoader
import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import logging

# C : 273,322   0 : 52,511   N : 24,629   S : 4,886   Cl : 4,762   F : 3,048   P : 458

def init_args_nmr():
    '''
    Setting up the input arguments.
    Takes the path to the NMR pretrain data
    and the save path for the log file.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-d', type=str, default="data/13C_nmrshiftdb.pickle")
    parser.add_argument('--save_path', '-s', type=str)
    return parser.parse_args()

def main():
    args = init_args_nmr()
    data_path_nmr = args.data_path
    save_path_nmr = args.save_path

    # Set up Tensorboard SummaryWriter.
    writer = SummaryWriter()

    # Set up logger.
    log_path = os.path.join(save_path_nmr, 'nmr_pretrain_model_log.log')
    logging.basicConfig(filename=log_path, format='%(asctime)s %(message)s', filemode='w')

    # Creating an object
    logger = logging.getLogger()

    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)

    split_df = pd.read_pickle(data_path_nmr)
    train = split_df['train_df']
    test = split_df['test_df']
    df = pd.concat((train, test)).reset_index(drop=True)

    mol_ids = df['molecule_id'].unique()

    for i in range(len(mol_ids)):

        # Dealing with the duplicated entries.
        if len(df.loc[df['molecule_id'] == mol_ids[i]]) > 1:
            new_nmr_shifts = nmr.average_nmr_shifts(df, mol_ids[i])

            # Replacing all the duplicate molecule values with the average NMR shifts.
            df.loc[df['molecule_id'] == mol_ids[i], ['value']] = new_nmr_shifts

            # Removing the duplicate entries sans the first one.
            df = nmr.remove_duplicates(df, mol_ids[i])

        # Unlisting the non-duplicated entries for pleasing symmetry.
        else:
            index = df.loc[df['molecule_id'] == mol_ids[i]].index[0]
            list_dict = df.loc[index, ['value']]
            df.loc[df['molecule_id'] == mol_ids[i], ['value']] = np.tile(list_dict[0], 1)

    df = df.reset_index(drop=True)

    df, longest_molecule = nmr.get_num_heavy_atoms(df, universal_node=True)

    max_ppm = nmr.get_max_ppm(df)
    mean_ppm = nmr.get_mean_ppm(df)
    ppm_std = nmr.get_ppm_std(df)
    logger.debug('max ppm: %d, mean ppm: %d, ppm std: %d', max_ppm, mean_ppm, ppm_std)

    Y = []
    for i in range(len(df)):
        nmr_labels = nmr.get_y(df.iloc[i], longest_molecule, mean_ppm, ppm_std)
        Y.append(nmr_labels)
    df['Y'] = Y
    
    atom_id = []
    for i in range(len(df)):
        atoms = nmr.atomic_symbols(df.iloc[i])
        atom_id.append(atoms)
    df['atom_id'] = atom_id

    atom_list = nmr.total_atom_types(df)
    
    train_val_df, test_df = nmr.df_split(df, 0.1)
    train_df, val_df = nmr.df_split(train_val_df, 0.1)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    train_data = nmr.NMR_Dataset(train_df, longest_molecule, atom_list)
    val_data = nmr.NMR_Dataset(val_df, longest_molecule, atom_list)
    test_data = nmr.NMR_Dataset(test_df, longest_molecule, atom_list)
    preds = test_df.copy()

    train_data = nmr.NMR_Dataset(train_df, longest_molecule, atom_list)
    val_data = nmr.NMR_Dataset(val_df, longest_molecule, atom_list)
    test_data = nmr.NMR_Dataset(test_df, longest_molecule, atom_list)
    
    # Defining device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Our device is %s", device)

    # Define the Pretrain Model.
    message_size = 30
    message_passes = 3
    atom_num_list = nmr.total_atom_numbers(atom_list)

    pretrain_model = NMR_MPNN(message_size, message_passes, atom_num_list)

    logger.debug("Model Defined")
    logger.debug("message_size = %d, message_passes = %d, atom_num_list = %s",
                 message_size, message_passes, atom_num_list)

    del message_size, message_passes, atom_num_list

    # Set up the DataLoader()
    dataloader_train = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=nmr.collate_nmr)
    dataloader_val = DataLoader(val_data, batch_size=64, shuffle=True, collate_fn=nmr.collate_nmr)
    dataloader_test = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=nmr.collate_nmr)

    lr = 1e-3

    # Define the optimzer.
    optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=lr)

    # Define number of epochs - How do you know the right number of epochs?
    epochs = 100

    # Epoch for loop.
    pretrain_model.to(device).float()

    for epoch in range(epochs):
        batch_train_loss = []
        batch_val_loss = []
        # Training
        pretrain_model.train()
        for i, (g, h, Y) in enumerate(dataloader_train):
            # Move the variables to device.
            g = g.to(device).float()
            h = h.to(device).float()
            Y = Y.to(device).float()
            optimizer.zero_grad()

            nmr_pred = pretrain_model(g, h)
            train_loss = torch.nn.MSELoss()(nmr_pred, Y)
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
        pretrain_model.eval()
        with torch.no_grad():
            for i, (g, h, Y) in enumerate(dataloader_val):
                # Move the variables to device.
                g = g.to(device).float()
                h = h.to(device).float()
                Y = Y.to(device).float()

                nmr_pred = pretrain_model(g, h)

                val_loss = torch.nn.MSELoss()(nmr_pred, Y)

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

    pretrain_model.eval()
    with torch.no_grad():
        for i, (g, h, Y) in enumerate(dataloader_test):
            
            batch_size = g.size()[0]
            # Move the variables to device.
            g = g.to(device).float()
            h = h.to(device).float()
            Y = Y.to(device).float()

            # Find predicted Y for all molecules in test set.
            nmr_pred = pretrain_model(g, h)
            nmr_pred = nmr_pred.view([batch_size, -1])

            # Adding the test losses and predictions to the predictions dataframe.
            for i in range(batch_size):
                test_preds.append(nmr_pred[i].cpu().detach().numpy())
                test_loss = torch.nn.MSELoss()(nmr_pred[i], Y[i])
                test_losses.append(test_loss.cpu().detach().numpy())

    # Save the predictions and losses into dataframe for saving.
    preds['Y_pred'] = test_preds
    preds['loss'] = test_losses

    preds_save_path = os.path.join(save_path_nmr, 'nmr_pred.pickle')
    preds.to_pickle(preds_save_path)

    # Save the model to save path.
    model_save_path = os.path.join(save_path_nmr, 'nmr_model')
    torch.save(pretrain_model.state_dict(), model_save_path)


if __name__ == '__main__':
    main()
