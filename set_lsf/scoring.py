'''
Finding the model's score.
'''

import numpy as np
import pandas as pd
import argparse

def snip(pred, test):
    '''
    Removing padding atoms from each prediction.
    '''
    all_preds = np.empty([0])
    for i in range(len(pred)):
        nha_i = test.loc[i, 'nha']
        p_i = pred[i][0:nha_i]
        all_preds = np.concatenate((all_preds, p_i))
    return all_preds

def simplify(pred):
    '''
    Changing each entry to a 0 or 1.
    '''
    sim_pred = np.zeros(len(pred))
    for i in range(len(pred)):
        if pred[i] > 0.5:
            sim_pred[i] = 1
        else:
            pass
    return sim_pred

def reward(true, pred):
    true_p = np.mean(true)
    pred_p = np.mean(pred)
    if pred_p == 0:
        pred_p = 1e-10
        
    print('true_p', true_p)
    print('pred_p', pred_p)
    tp = len(np.where(true + pred == 2)[0])
    tn = len(np.where(true + pred == 0)[0])
    fp = len(np.where(true - pred == -1)[0])
    fn = len(np.where(pred - true == -1)[0])
    print("tp", tp, "tn", tn, "fp", fp, "fn", fn)
    fit = -tp * np.log(pred_p) - tn * np.log(1-pred_p) + fp * np.log(true_p) + fn * np.log(1-true_p)

    return fit

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_pickle_path', '-preds', type=str)
    parser.add_argument('--evalutation', '-eval', type=str)
    return parser.parse_args()
def main():
  
    args = init_args()
    pred_pickle_path = args.pred_pickle_path
    evalutation = args.evalutation # should you use the P450 Y's or the retrospective Y's.
    
    all_y_p450 = np.load('data/all_y_p450.npy')
    all_y = np.load('data/all_y.npy')
    print('evalutation = ', evalutation)
    print(len(all_y))
    prediction_df = pd.read_pickle(pred_pickle_path)

    preds = snip(prediction_df['Y_pred'].tolist(), prediction_df)
    preds = simplify(preds)

    if evalutation == 'Retrospective':
        print(len(all_y))
        score = reward(all_y, preds)
    else:
        score = reward(all_y_p450, preds)
    print('score :', score)

if __name__ == '__main__':
    main()