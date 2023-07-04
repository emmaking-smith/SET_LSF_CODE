# Predictive Minisci and P450 Late Stage Functionalization with Transfer Learning
Code for the <sup>13</sup>C NMR pretraining and LSF finetuning as described in King-Smith *et al.*

## Getting Your Bearings
### set_lsf:
* The modules used to train the best model(s).
* The scoring function noted in the Supporting Information.
* The module to predict LSF regioselectivity on new molecules.
* Generating the Random Forest Baseline.
* The pickle file containing the predictions of the prospective molecules from the best model, MPNN<sub>LSF</sub>.

### data:
* The molecules used in the prospective validation, pre- and post-Glasgow Subgraph Solver processing.
* The open-source <sup>13</sup>C NMR data used in the pretraining.
* Non-proprietary compounds data.
* The full dataset has been registered with Pfizer under the name pfizer_LSF_NatureCommunications_PublicationDate.

### neural_nets & trained_models:
* The Message Passing Neural Networks (MPNNs) used for pretraining, finetuning, and for running new molecules.
* The best trained models for each of the neural networks.

### P450:
* The P450-specific setup utils and finetuning train/test module.

### utils:
* The modules for preparing the data for running in the MPNNs.
* The modules for automatically extracting the reaction sites from input excel data.
* The module used to find and cluster maximally different molecules used in prospective validation.

### Fukui_Additions:
* The module to set up the MPNN with Fukui indices as part of the atom featurization.

### Jensen_comparison:
* The modules used to setup the training / testing data for comparison to Jensen *et al.*'s ml-QM-GNN.

## How to Use
### Training
To train the data on the non-proprietary compounds, run the following command:

`python LSF_Finetune.py -s {PATH TO SAVE FILE}`

Similarly, re-training the model can be done with the following command:

`python LSF_Finetune.py -train {PATH TO TRAINING DATA PICKLE FILE} -test {PATH TO TESTING DATA PICKLE FILE} -s {PATH TO SAVE FILE}`
### Predictions on New Molecules
To prediction the regiochemical outcomes for new molecules:
1. Prepare an excel file in the format of ```prospective_with_product_smiles.xlsx```.
2. Run ```excel_to_gss.py```.
3. Follow the ```### BASH COMMANDS ###``` listed at the end of ```excel_to_gss.py```.
4. Run ```gss_to_react_centers.py```.
5. Run ```Run_New_Molecules.py -d {PATH TO gss_to_react_centers.py OUTPUT} -m neural_nets/trained_models/best_retrospective_model -s {PATH TO SAVE FILE}```.

## Dependencies
Run on python 3.7.
* numpy==1.21.2
* pandas==1.3.4
* rdkit==2020.09.1
* torch==1.8.1+cu111
* logging==0.5.1.2
* argparse==1.1
* matplotlib==3.3.4
* networkx==1.11
* tdqm==4.62.3
* sklearn==1.0.1
* [Glasgow Subgraph Solver](https://github.com/ciaranm/glasgow-subgraph-solver)
* os
* pathlib
* collections
