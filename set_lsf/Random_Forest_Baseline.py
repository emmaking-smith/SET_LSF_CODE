import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import argparse

rf = RandomForestClassifier()

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', '-train', type=str, help='Path to pickled training dataframe.')
    parser.add_argument('--test_data_path', '-test', type=str, help='Path to pickled testing dataframe.')
    return parser.parse_args()

def find_longest_molecule(train, test):
    whole_df = pd.concat((train, test)).reset_index(drop=True)
    all_mols = [Chem.MolFromSmiles(x) for x in whole_df['reactant'].unique()]
    lengths = [x.GetNumHeavyAtoms() for x in all_mols]
    return max(lengths)

def make_atomwise_fps(mol, longest_molecule):
    fps = []
    for atom in mol.GetAtoms():
        atom_index = atom.GetIdx()
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, fromAtoms=[atom_index])
        fps.append(fp)
    fps_array = fps_to_array(fps, longest_molecule)
    return fps_array

def fps_to_array(fps_list, longest_molecule):
    fps_array = np.zeros((longest_molecule, 2048))
    for i, fp in enumerate(fps_list):
        for on_bit in fp.GetOnBits():
            fps_array[i, on_bit] = 1
    return fps_array

def one_hot_rxn(df_row, unique_reagents, unique_oxidants, unique_additives, unique_acids, unique_solvents, unique_p450s):
    one_hot_reagent = [int(df_row['reagent'] == x) for x in unique_reagents]
    one_hot_oxidant = [int(df_row['oxidant'] == x) for x in unique_oxidants]
    one_hot_additive = [int(df_row['additive'] == x) for x in unique_additives]
    one_hot_acid = [int(df_row['acid'] == x) for x in unique_acids]
    one_hot_solvent = [int(df_row['solvent'] == x) for x in unique_solvents]
    
    rxn_vector = one_hot_reagent + one_hot_oxidant + one_hot_additive + one_hot_acid + one_hot_solvent
    
    if unique_p450s is not None:
        one_hot_p450 = [int(df_row['P450'] == x) for x in unique_p450s]
        rxn_vector = rxn_vector + one_hot_p450
    
    return np.array(rxn_vector)

def make_labels(pickle_data, longest_molecule):
    reaction_sites = pickle_data['parent_centres']
    Y = np.zeros(longest_molecule)
   
    if len(reaction_sites) > 0:
        for site in reaction_sites:
            Y[int(site)] = 1
    return Y

def make_label_all_df(df, longest_molecule):
    Y = []
    for i in range(len(df)):
        label = make_labels(df.iloc[i], longest_molecule)
        Y.append(label)
    df['Y'] = Y
    return df

def make_nha(df):
    nhas = []
    for i in range(len(df)):
        smiles_i = df.loc[i, 'reactant']
        mol_i = Chem.MolFromSmiles(smiles_i)
        nha_i = mol_i.GetNumHeavyAtoms()
        nhas.append(nha_i)
    df['nha'] = nhas
    return df
    
def snip(pred, test, longest_molecule):
    all_preds = np.empty([0])
    
    for i in range(int(len(pred) / longest_molecule)):
        nha_i = test.loc[i, 'nha']
        p_i = np.delete(pred, np.arange(longest_molecule * i))
        p_i = p_i[0:nha_i]
        all_preds = np.concatenate((all_preds, p_i))
    return all_preds

def scores(true, pred):
    true_p = np.mean(true)
    pred_p = np.mean(pred)
    if pred_p == 0:
        pred_p = 1e-10
        
    tp = len(np.where(true + pred == 2)[0])
    tn = len(np.where(true + pred == 0)[0])
    fp = len(np.where(true - pred == -1)[0])
    fn = len(np.where(pred - true == -1)[0])
    print("tp", tp, "tn", tn, "fp", fp, "fn", fn)
    #fit = -tp * np.log(pred_p) - tn * np.log(1-pred_p) + fp * np.log(true_p) + fn * np.log(1-true_p)
    f1 = 2 * tp / (2 * tp + fp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    auroc = roc_auc_score(true, pred)
    return f1, accuracy, auroc

def canonicalize_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def make_fps_inputs(df, longest_molecule):
    inputs = []
    for i in range(len(df)):
        canon_smiles_i = canonicalize_smiles(df.loc[i, 'reactant'])
        mol_i = Chem.MolFromSmiles(canon_smiles_i)
        atomwise_fp_i = make_atomwise_fps(mol_i, longest_molecule)
        inputs.append(atomwise_fp_i)
    inputs = np.array(inputs).reshape([-1, 2048])
    return inputs

def make_rxn_inputs(df, longest_molecule, unique_reagents, unique_oxidants, unique_additives, unique_acids, 
                    unique_solvents, unique_p450s):
    inputs = []
    for i in range(len(df)):
        one_hot_i = one_hot_rxn(df.iloc[i], unique_reagents, unique_oxidants,
                                unique_additives, unique_acids, unique_solvents, unique_p450s)
        one_hot_i = list(one_hot_i) * longest_molecule
        inputs.append(one_hot_i)
    inputs = np.array(inputs).reshape([longest_molecule * len(df), -1])
    return inputs
    
def make_inputs(df, longest_molecule, unique_reagents, unique_oxidants, unique_additives, unique_acids, 
                    unique_solvents, unique_p450s):
    fps_inputs = make_fps_inputs(df, longest_molecule)
    rxn_inputs = make_rxn_inputs(df,longest_molecule, unique_reagents, unique_oxidants, unique_additives, unique_acids, 
                    unique_solvents, unique_p450s)
    inputs = np.concatenate((fps_inputs, rxn_inputs), axis=1)
    return inputs

def make_rf_labels(df):
    all_labels = []
    for i in range(len(df)):
        all_labels.append(df.loc[i, 'Y'])
    all_labels = np.array(all_labels).reshape([-1])
    return all_labels
        
def main():

    args = init_args()
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    
    print('Loading in data.')
    train_df = pd.read_pickle(train_data_path).reset_index(drop=True)
    test_df = pd.read_pickle(test_data_path).reset_index(drop=True)  
    
    unique_reagents = train_df['reagent'].unique().tolist()
    unique_oxidants = train_df['oxidant'].unique().tolist()
    unique_additives = train_df['additive'].unique().tolist()
    unique_acids = train_df['acid'].unique().tolist()
    unique_solvents = train_df['solvent'].unique().tolist()
    if 'P450' in list(train_df.keys()):
        unique_p450s = train_df['P450'].unique().tolist()
        print('Using P450s in reaction vector.')
    else:
        unique_p450s = None
    
    longest_molecule = find_longest_molecule(train_df, test_df)

    train_df = make_nha(train_df)
    train_df = make_label_all_df(train_df, longest_molecule)
    train_df = train_df.reset_index(drop=True)

    test_df = make_nha(test_df)
    test_df = make_label_all_df(test_df, longest_molecule)
    test_df = test_df.reset_index(drop=True)
        
    train_inputs = make_inputs(train_df, longest_molecule, unique_reagents, unique_oxidants, unique_additives, unique_acids, 
                    unique_solvents, unique_p450s=unique_p450s)                      
    train_labels = make_rf_labels(train_df)
    
    test_inputs = make_inputs(test_df, longest_molecule, unique_reagents, unique_oxidants, unique_additives, unique_acids, 
                    unique_solvents, unique_p450s=unique_p450s) 
    test_labels = make_rf_labels(test_df)
    
    print('Fitting the random forest')
    # Fitting the random forest
    rf.fit(train_inputs, train_labels)
    print('Predicting...')
    preds = rf.predict(test_inputs)
    
    # How well did the random forest do?
    preds = snip(preds, test_df, longest_molecule)
    true = snip(test_labels, test_df, longest_molecule)
    
    f1, accuracy, auroc = scores(true, preds)
    
    print(f'f1 score: {f1}')
    print(f'accuracy: {accuracy}')
    print(f'auroc: {auroc}')

    
if __name__ == '__main__':
    main()