'''
train / test df for Jensen et al.'s model.
'''

import pandas as pd
from reaction_numbering import *

def make_csv_friendly_labels(product_idxs):
    '''
    The csv file doesn't like a list of an array. So we're transforming them into strings.
    '''
    csv_friendly_label = ''
    for i in range(len(product_idxs)):
        csv_friendly_label = csv_friendly_label + str(int(product_idxs[i]))
        if i != (len(product_idxs) - 1):
            csv_friendly_label = csv_friendly_label + '--'
    return csv_friendly_label

def csv_labels_to_array(csv_friendly_labels):
    '''
    Going the other way.
    '''
    list_form = csv_friendly_labels.split('--')
    int_form = [int(x) for x in list_form]
    array_form = np.array(int_form)
    return array_form

def main():
    jensen_train_df = pd.DataFrame()
    train = pd.read_pickle('20_JAN_2023_updated_train.pickle')
 
    
    all_product_idxs = []
    all_true_rxns = []
    all_possible_rxns = []
    for i in range(len(train)):
        unique_canon_products, product_idxs = make_rxn(train.loc[i, 'reactant'],
                                                                train.loc[i, 'reagent'],
                                                                train.loc[i, 'product'])
        try:
            # May not be able to generate a complete atom mapping.
            true_rxn, possible_rxns = make_all_atom_idx_mapping(train.loc[i, 'reactant'], train.loc[i, 'reagent'],
                                                            unique_canon_products, product_idxs, train=False)
            product_idxs = make_csv_friendly_labels(product_idxs)

            all_true_rxns.append(true_rxn)
            all_possible_rxns.append(possible_rxns)
            all_product_idxs.append(product_idxs)
        except:
            pass


    jensen_train_df['rxn_smiles'] = all_true_rxns
    jensen_train_df['products_run'] = all_possible_rxns
    jensen_train_df['correct_rxn_labels'] = all_product_idxs
    jensen_train_df['reaction_id'] = np.arange(len(jensen_train_df))
    jensen_train_df.to_csv(
        '/gpfs/workspace/users/kingse01/jensen_test/reactivity_predictions_substitution/jensen_test_df.csv')

if __name__ == '__main__':
    main()