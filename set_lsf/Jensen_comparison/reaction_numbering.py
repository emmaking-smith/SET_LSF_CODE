'''
Generating the atom mappings for input into Jensen et al.'s model.

'''

import numpy as np
from rxnmapper import RXNMapper
from rdkit import Chem
from reaction_elaboration import make_elaborations, reagent_smarts_dict, reagent_smiles_dict
import re

rxn_mapper = RXNMapper()

def canonicalize_smiles(smiles):
    s = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    return s

def make_all_products(sm_smiles, reagent, product_smiles, train=True):
    elaborations = make_elaborations(sm_smiles, reagent)
    products = elaborations + [sm_smiles] + product_smiles
    canon_products = []
    for p in products:
        try:
            canon_products.append(canonicalize_smiles(p))
        except:
            pass
    unique_canon_products = np.unique(canon_products)
    return unique_canon_products

def make_shuffled_products(unique_canon_products):
    shuffled = np.random.permutation(unique_canon_products)
    return shuffled

def find_product(unique_canon_products, canon_product_smiles):
    idxs = [np.where(unique_canon_products == x)[0] for x in canon_product_smiles]
    return idxs

def make_canon_product_smiles(sm_smiles, product_smiles_list):
    canon_products = []
    for smiles in product_smiles_list:
        try:
            canon_products.append(canonicalize_smiles(smiles))
        except:
            pass
    if len(canon_products) == 0:
        canon_products.append(canonicalize_smiles(sm_smiles))
    return canon_products

def make_rxn(sm_smiles, reagent, product_smiles):
    '''
    Getting the possile products and ids of correct products.
    '''
    # Canonicalize everything.
    sm_canon_smiles = canonicalize_smiles(sm_smiles)
    # Note that products can still be [nan] or [nan, nan, ..., p1, ...]
    product_canon_smiles = make_canon_product_smiles(sm_smiles, product_smiles)

    # Generating all possible carbon functionalizations AND unreacted SM to proxy
    # for 'no rxn' rxns and removing structures that are not chemically feasible.
    unique_canon_products = make_all_products(sm_canon_smiles, reagent, product_canon_smiles)
    # shuffled_unique_canon_products = make_shuffled_products(unique_canon_products)

    # Find the idxs that correspond to all products.
    product_idxs = find_product(unique_canon_products, product_canon_smiles)
    return unique_canon_products, product_idxs

def make_rxn_string(sm, reagent, product):
    # Adding in a clause that removes the reagent from
    # the reaction for accurate mapping.
    if product != sm:
        rxn_string = sm + '.' + reagent + '>>' + product
    else:
        rxn_string = sm + '>>' + product
    return rxn_string

def untangle_mapping_results(mapped_results):
    '''
    Getting the results out from the rxnmapper() dictionary.
    '''
    untangled_results = [mapped_results[x]['mapped_rxn'] for x in range(len(mapped_results))]
    return untangled_results

def find_mapped_products(untangled_results):
    '''
    Extracting the mapped products.
    '''
    mapped_products_string = ''
    for i in range(len(untangled_results)):
        mapped_products_string = mapped_products_string + untangled_results[i].split('>>')[1]
        if i != (len(untangled_results) - 1):
            mapped_products_string = mapped_products_string + '.'
    return mapped_products_string

def find_the_atom_numberings(mapped_smiles):
    '''
    I take no credit for this lovely lovely piece of code.
    '''
    atom_numbers = list(map(int, re.findall('(?<=\:)[0-9]+(?=\])', mapped_smiles)))
    return atom_numbers

def setting_the_molAtomMapNumber(mapped_smiles):
    '''
    For a single smiles string mind you.
    '''
    atom_numbers = find_the_atom_numberings(mapped_smiles)
    mol = Chem.MolFromSmiles(mapped_smiles)
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetIntProp('molAtomMapNumber', atom_numbers[i])
    set_smiles = Chem.MolToSmiles(mol)
    return set_smiles

def strip_mols_out_of_rxn_string(mapped_rxn):
    '''
    Taking a mapped rxn to smiles strings.
    '''
    reactants = mapped_rxn.split('>>')[0]
    product = mapped_rxn.split('>>')[1]
    if reactants == product:
        sm = reactants
        reagent = None
    else:
        sm = reactants.split('.')[0]
        reagent = reactants.split('.')[1]
    return sm, reagent, product

def make_the_GetIntProp(mapped_rxn):
    sm, reagent, product = strip_mols_out_of_rxn_string(mapped_rxn)
    sm = setting_the_molAtomMapNumber(sm)
    product = setting_the_molAtomMapNumber(product)
    if reagent is not None:
        reagent = setting_the_molAtomMapNumber(reagent)
        set_mapped_rxn = sm + '.' + reagent + '>>' + product
    else:
        set_mapped_rxn = sm + '>>' + product
    return set_mapped_rxn

def make_all_atom_idx_mapping(sm_smiles, reagent, unique_canon_products, product_idx, train=True):
    '''
    The whole enchilada.
    '''
    # Canonicalization because what the heck at this point. Can't hurt.
    canon_sm_smiles = canonicalize_smiles(sm_smiles)
    # Getting the canonical reagent smiles for rxn string.
    # NOTE!!! THE ATOM MAPPING DOES NOT WORK PROPERLY FOR UNBALANCED RXNS! YOU
    # MUST USE THE REAGENT SYNTHON RATHER THAN THE TRUE REAGENT.
    canon_reagent_smiles = canonicalize_smiles(reagent_smarts_dict[reagent])
    # Generating all rxn mappings for all possible rxn products.
    possible_rxns = [make_rxn_string(canon_sm_smiles, canon_reagent_smiles, x) for x in unique_canon_products]
    mapped_possible_rxns = rxn_mapper.get_attention_guided_atom_maps(possible_rxns)
    mapped_possible_rxns = untangle_mapping_results(mapped_possible_rxns)
    # Setting the molAtomMapNumber property.
    mapped_possible_rxns = [make_the_GetIntProp(x) for x in mapped_possible_rxns]
    # Selecting the mapping that will be used as a SM. The first entry should never be
    # a 'no rxn' rxn.
    true_rxn = mapped_possible_rxns[0]
    # If training, put the a  correct product first.
    if train == True:
        mapped_possible_rxns.insert(0, mapped_possible_rxns.pop(int(product_idx[0])))
    else:
        mapped_possible_rxns = make_shuffled_products(mapped_possible_rxns)
    # Taking the list of products to a single string : p1.p2.p3 ...
    mapped_products = find_mapped_products(mapped_possible_rxns)
    return true_rxn, mapped_products

def check_numbering(sm_numbered, p_numbered):
    '''
    Make sure that the two have equal number of atoms in them.
    '''
    check = 0
    for i in p_numbered:
        if i not in sm_numbered:
            check = check + 1
    return check