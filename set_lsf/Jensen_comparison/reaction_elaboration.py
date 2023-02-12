
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from itertools import chain

reagent_smarts_dict = {'O2' : '[OH]', 'PF-07222283' : 'C#CCOc1ccc(C(c2ccc(OCCCC(F)F)cc2)=O)cc1', '(CF3SO2)2Zn' : 'C(F)(F)(F)',
                '(HCF2SO2)2Zn' : 'C(F)(F)', 'MeOCH2BF3K' : 'COC', 'N-Me formamide' : 'CNC=O', 'HOCH2SO2Na' : 'CO',
                'Selectfluor' : 'F', 'DAST' : 'F', 'H2O' : '[OH]', 'NFSI' : 'F', 'R-BF3K' : 'C',
                '1,2,3 triazole' : 'c1c[nH]nn1', 'cBuBF3K' : 'C1CCC1', 'CF3SO2Na' : 'C(F)(F)(F)', 'CH3COOH' : 'C',
                'tBuOOC(O)CH3 (tBPA)' : 'C', 'THFSO2Na' : 'O1CCCC1', 'Formamide' : 'O=CN', 'THPSO2Na' : 'C1CCCCO1',
                'Iodo oxetane' : 'C1COC1', '(CHF2CO)2O' : 'C(F)(F)', 'MeOH' : 'C', 'NBOCAzetidineSO2Na' :
                'CC(OC(N1CCC1)=O)(C)C', '1-CF3-cPrSO2Na' : 'FC(C1CC1)(F)F', 'iPrBF3K' : 'CCC', 'iPrCOOH' : 'CCC',
                'N-Chloro succinimide' : 'Cl', 'B(OMe)3' : 'C', 'CF3CH2CH2SO2Na' : 'CCC(F)(F)F', 'cPrBF3K' : 'C1CC1',
                'tBuBF3K' : 'CC(C)C', '4,4-diF cHexCH2BF3K' : 'FC1(F)CCC(C)CC1', 'cBuOCH2BF3K' : 'COC1CCC1',
                'cHexBF3K' : 'C1CCCCC1', 'THPBF3K' : 'C1CCCCO1', 'cPrCH2BF3K' : 'CC1CC1', 'pyrazole' : 'c1cn[nH]c1',
                '1,2,4, triazole' : 'c1nc[nH]n1', 'CBMG' : 'Cl', '(iPrSO2)2Zn' : 'CCC', 'cPrCH2OCH2BF3K' : 'COCC1CC1',
                'CH3CH2SO2Na' : 'C(C)', 'OxetaneBF3K' : 'C1COC1', 'Et3N.3HF' : 'F', 'HCF2SO2Na' : 'C(F)(F)',
                'Pivalic acid' : 'CC(C)C', 'N-Acetyl Gly' : 'CNC=O', 'NBOCAzetidineBF3K' : 'CC(OC(N1CCC1)=O)(C)C',
                'CF3BF3K' : 'C(F)(F)(F)', '4,4-diF cHexSO2Na' : 'FC1(F)CCCCC1', 'HCF2COOH' : 'C(F)(F)',
                '(CF3CH2SO2)2Zn' : 'C(F)(F)(F)C', '3,3-diF cBuBF3K' : 'FC1(F)CC1', 'NHBoc Gly' : 'CNC(OC(C)(C)C)=O',
                'NH(Boc)CH2BF3K' : 'CNC(OC(C)(C)C)=O', 'N-Me acetamide' : 'CNC=O', 'Glycine' : 'CN', 'NHMe Gly' :
                'CNC', 'N,N-diMe Gly' : 'CN(C)C', '1-pyrrolidine-Me BF3K' : 'CN1CCCC1', 'Iodo methylnitrile' : 'CC#N',
                'N,N-diMeCH2BF3K' : 'CN(C)C', 'cPrCOOH' : 'C1CC1', 'HOCH2COOH' : 'CO', '2-CH3 cPrSO2Na' : 'CC1CC1',
                '(4-(benzyloxy)butyl)B(OH)2' : 'CCCCOCC1=CC=CC=C1', 'HO(CH2)5CO2Na' : 'CCCCCO', '(nPrSO2)2Zn' : 'CCC',
                'AcNHCH2BF3K' : 'CNC=O', 'CF3CH2CH2BF3K' : 'CCC(F)(F)F', 'TBHP' : '[OH]', 'Iodo azetidineN-Boc' :
                'CC(OC(N1CCC1)=O)(C)C', '(FCH2SO2)2Zn' : 'C(F)', '(cHexSO2)2Zn' : 'C1CCCCC1', '(ClCH2SO2)2Zn' : 'C(Cl)',
                'PhIO' : '[OH]', 'H2O2' : '[OH]', 'NaOCl' : '[OH]', '2,6-Cl2PyNO' : '[OH]', 'PIDA' : '[OH]', 'None' : '[OH]',
                'mCPBA' : '[OH]', '[nBu4N]IO4' : '[OH]', 'CH3CF2SO2Na' : 'CC(F)(F)'}

reagent_smiles_dict = {'O2' : 'O=O', 'PF-07222283' : 'C#CCOc1ccc(C(c2ccc(OCCCC(F)F)cc2)=O)cc1', '(CF3SO2)2Zn' :
                'O=S(O[Zn]OS(=O)(C(F)(F)F)=O)(C(F)(F)F)=O', '(HCF2SO2)2Zn' : 'O=S(O[Zn]OS(=O)(C(F)F)=O)(C(F)F)=O',
                'MeOCH2BF3K' : 'COCB(F)F.[F-].[K+]', 'N-Me formamide' : 'O=CNC', 'HOCH2SO2Na' : 'OCS(O[Na])=O',
                'Selectfluor' : 'F[N+]12CC[N+](CC2)(CCl)CC1.F[B-](F)(F)F.F[B-](F)(F)F', 'DAST' : 'CCN(S(F)(F)F)CC',
                'H2O' : 'O', 'NFSI' : 'O=S(N(F)S(=O)(C1=CC=CC=C1)=O)(C2=CC=CC=C2)=O', 'R-BF3K' : 'CB(F)F.[F-].[K+]',
                '1,2,3 triazole' : 'c1c[nH]nn1', 'cBuBF3K' : 'FB(C1CCC1)F.[F-].[K+]', 'CF3SO2Na' : 'FC(F)(S(O[Na])=O)F',
                'CH3COOH' : 'CC(O)=O', 'tBuOOC(O)CH3 (tBPA)' : 'CC(OOC(C)(C)C)=O', 'THFSO2Na' : 'O=S(C1COCC1)O[Na]',
                'Formamide' : 'O=CN', 'THPSO2Na' : 'O=S(C1CCCCO1)O[Na]', 'Iodo oxetane' : 'IC1COC1', '(CHF2CO)2O' :
                'O=C(OC(C(F)F)=O)C(F)F', 'MeOH' : 'OC', 'NBOCAzetidineSO2Na' : 'O=S(C1CN(C1)C(OC(C)(C)C)=O)O[Na]',
                '1-CF3-cPrSO2Na' : 'O=S(C1(C(F)(F)F)CC1)O[Na]', 'iPrBF3K' : 'CC(B(F)F)C.[F-].[K+]', 'iPrCOOH' :
                'OC(C(C)C)=O', 'N-Chloro succinimide' : 'O=C(N1Cl)CCC1=O', 'B(OMe)3' : 'COB(OC)OC', 'CF3CH2CH2SO2Na' :
                'FC(CCS(O[Na])=O)(F)F', 'cPrBF3K' : 'FB(C1CC1)F.[F-].[K+]', 'tBuBF3K' : 'FB(C(C)(C)C)F.[F-].[K+]',
                '4,4-diF cHexCH2BF3K' : 'FC(CC1)(F)CCC1CB(F)F.[F-].[K+]', 'cBuOCH2BF3K' : 'FB(COC1CCC1)F.[F-].[K+]',
                'cHexBF3K' : 'FB(C1CCCCC1)F.[F-].[K+]', 'THPBF3K' : 'FB(C1CCCCO1)F.[F-].[K+]', 'cPrCH2BF3K' :
                'FB(CC1CC1)F.[F-].[K+]', 'pyrazole' : 'c1cn[nH]c1', '1,2,4, triazole' : 'c1nc[nH]n1', 'CBMG' :
                'COC(=O)NC(=NCl)NC(=O)OC', '(iPrSO2)2Zn' : 'O=S(O[Zn]OS(=O)(C(C)C)=O)(C(C)C)=O', 'cPrCH2OCH2BF3K' :
                'FB(OCC1CC1)F.[F-].[K+]', 'CH3CH2SO2Na' : 'CCS(O[Na])=O', 'OxetaneBF3K' : 'FB(C1COC1)F.[F-].[K+]',
                'Et3N.3HF' : 'CCN(CC)CC.[H]F.[H]F.[H]F', 'HCF2SO2Na' : 'FC(S(O[Na])=O)F', 'Pivalic acid' : 'CC(C)(C)C(O)=O',
                'N-Acetyl Gly' : 'O=C(O)CNC(C)=O', 'NBOCAzetidineBF3K' : 'FB(C1CN(C1)C(OC(C)(C)C)=O)F.[F-].[K+]',
                'CF3BF3K' : 'FB(C(F)(F)F)F.[F-].[K+]', '4,4-diF cHexSO2Na' : 'FC1(F)CCC(S(O[Na])=O)CC1', 'HCF2COOH' :
                'FC(C(O)=O)F', '(CF3CH2SO2)2Zn' : 'O=S(CC(F)(F)F)(O[Zn]OS(CC(F)(F)F)(=O)=O)=O', '3,3-diF cBuBF3K' :
                'FC1(F)CCC1B(F)F.[F-].[K+]', 'NHBoc Gly' : 'O=C(O)CNC(OC(C)(C)C)=O','NH(Boc)CH2BF3K' :
                'FB(CNC(OC(C)(C)C)=O)F.[F-].[K+]', 'N-Me acetamide' : 'CC(NC)=O', 'Glycine' : 'NCC(O)=O', 'NHMe Gly' :
                'O=C(O)CNC', 'N,N-diMe Gly' : 'O=C(O)CN(C)C', '1-pyrrolidine-Me BF3K' : 'FB(CN1CCCC1)F.[F-].[K+]',
                'Iodo methylnitrile' : 'IC#N', 'N,N-diMeCH2BF3K' : 'CN(C)CB(F)F.[F-].[K+]', 'cPrCOOH' : 'O=C(C1CC1)O',
                'HOCH2COOH' : 'OCC(O)=O', '2-CH3 cPrSO2Na' : 'CC1CC1S(O[Na])=O', '(4-(benzyloxy)butyl)B(OH)2' :
                'OB(CCCCOCC1=CC=CC=C1)O', 'HO(CH2)5CO2Na' : 'O=C(O[Na])CCCCCO', '(nPrSO2)2Zn' :
                'O=S(O[Zn]OS(=O)(CCC)=O)(CCC)=O', 'AcNHCH2BF3K' : 'CC(NCB(F)F)=O.[F-].[K+]', 'CF3CH2CH2BF3K' :
                'FC(CCB(F)F)(F)F.[F-].[K+]', 'TBHP' : 'OOC(C)(C)C', 'Iodo azetidineN-Boc' : 'IC1CN(C1)C(OC(C)(C)C)=O',
                '(FCH2SO2)2Zn' : 'O=S(O[Zn]OS(=O)(CF)=O)(CF)=O', '(cHexSO2)2Zn' : 'O=S(C1CCCCC1)(O[Zn]OS(C2CCCCC2)(=O)=O)=O',
                '(ClCH2SO2)2Zn' : 'O=S(CCl)(O[Zn]OS(CCl)(=O)=O)=O','PhIO' : 'O=IC1=CC=CC=C1', 'H2O2' : 'OO', 'NaOCl' :
                'Cl[O-].[Na+]', '2,6-Cl2PyNO' : 'ClC1=[N+](O)C(Cl)=CC=C1', 'PIDA' : 'O=C(OI(OC(C)=O)C1=CC=CC=C1)C',
                'None' : 'O', 'mCPBA' : 'O=C(OO)C1=CC=CC(Cl)=C1', '[nBu4N]IO4' : 'CCCC[N+](CCCC)(CCCC)CCCC.[O-][I+3]([O-])([O-])[O-]',
                'CH3CF2SO2Na' : 'CC(F)(F)S(O[Na])=O'}

def make_elaborations(sm_smiles, reagent):
    '''
    R-group is the smarts string.
    '''
    r_group_smarts = reagent_smarts_dict[reagent]
    reacts = (Chem.MolFromSmiles(sm_smiles), Chem.MolFromSmiles(r_group_smarts))

    if r_group_smarts == 'C':
        rxn = rdChemReactions.ReactionFromSmarts('[c,C:1].[C:2]>>[c,C:1]([C:2])')
    elif r_group_smarts == 'C#CCOc1ccc(C(c2ccc(OCCCC(F)F)cc2)=O)cc1':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2]([F:3])([F:4])>>[C:2]([c,C:1])([F:3])([F:4])')
    elif r_group_smarts == 'C(C)':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2]([C:3])>>[c,C:1]([C:2]([C:3]))')
    elif r_group_smarts == 'C(Cl)':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2]([Cl:3])>>[c,C:1]([C:2]([Cl:3]))')
    elif r_group_smarts == 'C(F)':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2]([F:3])>>[c,C:1]([C:2]([F:3]))')
    elif r_group_smarts == 'C(F)(F)':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2]([F:3])([F:4])>>[c,C:1]([C:2]([F:3])([F:4]))')
    elif r_group_smarts == 'C(F)(F)(F)':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2]([F:3])([F:4])([F:5])>>[c,C:1]([C:2]([F:3])([F:4])([F:5]))')
    elif r_group_smarts == 'C(F)(F)(F)C':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:6][C:2]([F:3])([F:4])([F:5])>>[c,C:1]([C:6][C:2]([F:3])([F:4])([F:5]))')
    elif r_group_smarts == 'C1CC1':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2]1[C:3][C:4]1>>[c,C:1]([C:2]1[C:3][C:4]1)')
    elif r_group_smarts == 'C1CCC1':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2]1[C:3][C:4][C:5]1>>[c,C:1]([C:2]1[C:3][C:4][C:5]1)')
    elif r_group_smarts == 'C1CCCCC1':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2]1[C:3][C:4][C:5][C:6][C:7]1>>[c,C:1]([C:2]1[C:3][C:4][C:5][C:6][C:7]1)')
    elif r_group_smarts == 'C1CCCCO1': # THP ortho
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2]1[C:3][C:4][C:5][C:6][0:7]1>>[c,C:1]([C:2]1[C:3][C:4][C:5][C:6][0:7]1)')
    elif r_group_smarts == 'C1COC1': # oxetane para
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2]1[C:3][O:4][C:5]1>>[c,C:1]([C:2]1[C:3][O:4][C:5]1)')
    elif r_group_smarts == 'CC#N':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2][C:3]#[N:4]>>[c,C:1]([C:2][C:3]#[N:4])')
    elif r_group_smarts == 'CC(C)C':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2][C:3]([C:4])[C:5]>>[C:2][C:3]([c,C:1])([C:4])([C:5])')
    elif r_group_smarts == 'CC(OC(N1CCC1)=O)(C)C':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2][C:3]([O:4][C:5]([N:6]1[C:7][C:8][C:9]1)=[O:10])([C:11])[C:12]>>[C:2][C:3]([O:4][C:5]([N:6]1[C:7][C:8]([c,C:1])[C:9]1)=[O:10])([C:11])[C:12]')
    elif r_group_smarts == 'CC1CC1' and reagent == 'cPrCH2BF3K':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2][C:3]1[C:4][C:5]1>>[c,C:1]([C:2][C:3]1[C:4][C:5]1)')
    elif r_group_smarts == 'CC1CC1' and reagent == '2-CH3 cPrSO2Na':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2][C:3]1[C:4][C:5]1>>[C:2][C:3]1[C:4][C:5]1([c,C:1])')
    elif r_group_smarts == 'CCC' and 'iPr' in reagent:
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2][C:3][C:4]>>[C:2][C:3]([c,C:1])[C:4]')
    elif r_group_smarts == 'CCC' and 'nPr' in reagent:
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2][C:3][C:4]>>[c,C:1]([C:2][C:3][C:4])')
    elif r_group_smarts == 'CCC(F)(F)F':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2][C:3][C:4]([F:5])([F:6])[F:7]>>[c,C:1]([C:2][C:3][C:4]([F:5])([F:6])[F:7])')
    elif r_group_smarts == 'CCCCCO':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2][C:3][C:4][C:5][C:6][OH:7]>>[c,C:1]([C:2][C:3][C:4][C:5][C:6][OH:7])')
    elif r_group_smarts == 'CCCCOCC1=CC=CC=C1':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2][C:3][C:4][C:5][O:6][C:7][c:8]1[c:9][c:10][c:11][c:12][c:13]1>>[c,C:1]([C:2][C:3][C:4][C:5][O:6][C:7][c:8]1[c:9][c:10][c:11][c:12][c:13]1)')
    elif r_group_smarts == 'CN':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2][N:3]>>[c,C:1]([C:2][N:3])')
    elif r_group_smarts == 'CN(C)C':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2][N:3]([C:4])[C:5]>>[c,C:1]([C:2][N:3]([C:4])[C:5])')
    elif r_group_smarts == 'CNC':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2][N:3][C:4]>>[c,C:1]([C:2][N:3][C:4])')
    elif r_group_smarts == 'CNC=O':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2][N:3][C:4]=[O:6]>>[C:2][N:3][C:4]([c,C:1])=[O:6]')
    elif r_group_smarts == 'CNC(OC(C)(C)C)=O':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2][N:3][C:4]([O:5][C:6]([C:7])([C:8])[C:9])=[O:10]>>[c,C:1]([C:2][N:3][C:4]([O:5][C:6]([C:7])([C:8])[C:9])=[O:10])')
    elif r_group_smarts == 'CO':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2][O:3]>>[c,C:1]([C:2][O:3])')
    elif r_group_smarts == 'COC':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2][O:3][C:4]>>[c,C:1]([C:2][O:3][C:4])')
    elif r_group_smarts == 'COC1CCC1':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2][O:3][C:4]1[C:5][C:6][C:7]1>>[c,C:1]([C:2][O:3][C:4]1[C:5][C:6][C:7]1)')
    elif r_group_smarts == 'COCC1CC1':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2][O:3][C:4][C:5]1[C:6][C:7]1>>[c,C:1]([C:2][O:3][C:4][C:5]1[C:6][C:7]1)')
    elif r_group_smarts == 'Cl':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[Cl:2]>>[c,C:1]([Cl:2])')
    elif r_group_smarts == 'F':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[F:2]>>[c,C:1]([F:2])')
    elif r_group_smarts == 'FC(C1CC1)(F)F':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[F:2][C:3]([C:4]1[C:5][C:6]1)([F:7])[F:8]>>[F:2][C:3]([C:4]1([c,C:1])[C:5][C:6]1)([F:7])[F:8]')
    elif r_group_smarts == 'FC1(F)CCC(C)CC1':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[F:2][C:3]1([F:4])[C:5][C:6][C:7]([C:8])[C:9][C:10]1>>[F:2][C:3]1([F:4])[C:5][C:6][C:7]([C:8]([c,C:1]))[C:9][C:10]1')
    elif r_group_smarts == 'FC1(F)CCCCC1':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[F:2][C:3]1([F:4])[C:5][C:6][C:7][C:8][C:9]1>>[F:2][C:3]1([F:4])[C:5][C:6][C:7]([c,C:1])[C:8][C:9]1')
    elif r_group_smarts == 'FC1(F)CC1':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[F:2][C:3]1([F:4])[C:5][C:6]1>>[F:2][C:3]1([F:4])[C:5]([c,C:1])[C:6]1')
    elif r_group_smarts == 'CN1CCCC1': # pyrroldine
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2][N:3]1[C:4][C:5][C:6][C:7]1>>[c,C:1]([C:2][N:3]1[C:4][C:5][C:6][C:7]1)')
    elif r_group_smarts == 'O1CCCC1': # THF
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[O:2]1[C:3][C:4][C:5][C:6]1>>[O:2]1[C:3][C:4]([c,C:1])[C:5][C:6]1')
    elif r_group_smarts == 'O=CN':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[O:2]=[C:3][N:4]>>[O:2]=[C:3]([c,C:1])[N:4]')
    elif r_group_smarts == '[OH]':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[OH:2]>>[c,C:1]([OH:2])')
    elif r_group_smarts == 'c1cn[nH]c1': # pyrazole
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[c:2]1[c:3][n:4][nH:5][c:6]1>>[c:2]1[c:3][n:4][n:5]([c,C:1])[c:6]1')
    elif r_group_smarts == 'c1c[nH]nn1': # 1,2,3-triazole
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[c:2]1[c:3][nH:4][n:5][n:6]1>>[c:2]1[c:3][n:4]([c,C:1])[n:5][n:6]1')
    elif r_group_smarts == 'c1nc[nH]n1': # 1,2,4-triazole
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[c:2]1[n:3][c:4][nH:5][n:6]1>>[c:2]1[n:3][c:4][n:5]([c,C:1])[n:6]1')
    elif r_group_smarts == 'CC(F)(F)':
        rxn = rdChemReactions.ReactionFromSmarts(
            '[c,C:1].[C:2][C:3]([F:4])([F:5])>>[C:2][C:3]([c,C:1])([F:4])([F:5])')

    products = rxn.RunReactants(reacts)
    products = [Chem.MolToSmiles(products[x][0]) for x in range(len(products))]
    return products
