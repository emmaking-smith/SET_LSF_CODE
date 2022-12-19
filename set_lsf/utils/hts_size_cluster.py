import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import PandasTools
from rdkit.ML.Cluster.Butina import ClusterData
from tqdm import tqdm

def clustering(prepped_data, n_div, dist):
    
    # turning it into a dataframe
    df = pd.DataFrame()
    df['mol'] = prepped_data
    df['fps'] = [AllChem.GetMorganFingerprintAsBitVect(
        m, 2, 2048) for m in df['mol']]
    fps = df['fps'].values

    # this generates the lower triangular part of the Tanimoto Distance matrix
    # between all the molecules (1 - Tanimoto Similarity)
    n = len(df)
    d_mat = np.empty(int(n*(n-1)/2))

    n = 0
    for i in tqdm(range(1, len(fps))):
        d_mat[n:n+i] = np.ones_like(fps[:i]) - \
            DataStructs.cDataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        n += i

    # this clusters the data and returns a tuple
    # where each element contains the centroid of the cluster
    # and then all the members of the cluster
    # the clusters are ordered by cluster size

    clusters = ClusterData(d_mat, nPts=len(
        df), isDistData=True, distThresh=dist, reordering=True)

    print('Number of clusters: {} (from {} mols)'.format(
            len(clusters), len(df)))

    # choose n_div diverse molecules by picking the centroid of the
    # n_div biggest clusters
    diverse_df = df.iloc[[clusters[n][0] for n in range(n_div)]]

    return(diverse_df['mol'])

def is_hetero(mol):
    answer = 0
    aro_n = Chem.MolFromSmarts('n')
    if len(mol.GetSubstructMatch(aro_n)) > 0:
        answer = 'hetero'
    else:
        answer = 'not hetero'
    return answer
    
def main():
    #hts = PandasTools.LoadSDF('/gpfs/workspace/users/kingse01/Enamine_hts.sdf')

    # 3 groupings 10 - 20, 21 - 30, 31 - 40.
    #small = []
    #medium = []
    #large = []
    #for i in hts['ROMol']:
    #    is_mol_hetero = is_hetero(i)
    #    if is_mol_hetero == 'hetero':
    #        atom_num = len(i.GetAtoms())
    #        if atom_num >= 10 and atom_num < 20:
    #            small.append(i)
    #        elif atom_num >= 20 and atom_num < 30:
    #            medium.append(i)
    #        elif atom_num >= 30 and atom_num < 40:
    #            large.append(i)
    #            
    #print(len(small), len(medium), len(large))

    #df_s = pd.DataFrame({'mol' : small})
    #df_m = pd.DataFrame({'mol' : medium})
    #df_l = pd.DataFrame({'mol' : large})
    print('all mediums')
    df_m1 = pd.read_pickle('/gpfs/workspace/users/kingse01/m1_clustered.pickle')
    df_m2 = pd.read_pickle('/gpfs/workspace/users/kingse01/m2_clustered.pickle')
    df_m3 = pd.read_pickle('/gpfs/workspace/users/kingse01/m3_clustered.pickle')
    df_m4 = pd.read_pickle('/gpfs/workspace/users/kingse01/m4_clustered.pickle')
    df_m5 = pd.read_pickle('/gpfs/workspace/users/kingse01/m5_clustered.pickle')
    df_m = pd.concat((df_m1, df_m2))
    df_m = pd.concat((df_m, df_m3))
    df_m = pd.concat((df_m, df_m4))
    df_m = pd.concat((df_m, df_m5))
    df_m = df_m.reset_index(drop=True)
    #df_s.to_pickle('/gpfs/workspace/users/kingse01/df_s.pickle')
    #df_m.to_pickle('/gpfs/workspace/users/kingse01/df_m.pickle')
    #df_l.to_pickle('/gpfs/workspace/users/kingse01/df_l.pickle')
    #df_s = pd.read_pickle('/gpfs/workspace/users/kingse01/df_s.pickle')
    #df_l = pd.read_pickle('/gpfs/workspace/users/kingse01/df_l.pickle')
    # df_m = pd.read_pickle('/gpfs/workspace/users/kingse01/df_m5.pickle')
    #s_clustered = clustering(df_s, 10, 0.7)
    #s_clustered.to_pickle('/gpfs/workspace/users/kingse01/s_clustered.pickle')
    m_clustered = clustering(df_m, 10, 0.7)
    #l_clustered = clustering(df_l, 10, 0.7)

    m_clustered.to_pickle('/gpfs/workspace/users/kingse01/m_all_clustered.pickle')
    #l_clustered.to_pickle('/gpfs/workspace/users/kingse01/l_clustered.pickle')

if __name__ == "__main__":
    main()
