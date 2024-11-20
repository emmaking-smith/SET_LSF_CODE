'''
A MPNN for finetuning on the LSF dataset.
'''


import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)
from neural_nets.NMR_4ll_Net import NMR_MPNN
device = "cuda" if torch.cuda.is_available() else "cpu"

class LSF_MPNN(nn.Module):
    '''
    The LSF selectivity prediction layers to be used in conjunction with NMR_Pretrain_Net module's
    NMR_MPNN
    ----
    Args:
        message_size (int):
            The size of the final dimension from the output from NMR_MPNN.

        rxn_features_length (int):
            The length of a single atom's / molecule's reaction features.

    Forward Args:
        embedded_molecules (tensor):
            The output from the NMR_MPNN network of size batch size x longest molecule.

        rxn_vector (tensor):
            The one hot encoded representations of the reaction conditions.

    Returns:
        output (tensor):
            The predictions of site selectivity for each atom.
            Output has size batch size x longest molecule x 1.
    '''
    def __init__(self, message_size, message_passes, all_unique_atoms, rxn_features_length, pretrain_path):
        super(LSF_MPNN, self).__init__()

        self.message_size = message_size
        self.message_passes = message_passes
        self.all_unique_atoms = all_unique_atoms
        self.rxn_features_length = rxn_features_length
        self.pretrain_path = pretrain_path
        self.top_4_unique_atoms = self.all_unique_atoms[0:4] + [0]

        # Expanding the output of the NMR MPNN before concatenation with rxn vector.
        self.mpnn = NMR_MPNN(self.message_size, self.message_passes, self.all_unique_atoms)
        self.mpnn.load_state_dict(torch.load(self.pretrain_path))

        # Removing the last set of linear layers that predict 13C NMR.
        self.cutoff_mpnn = list(self.mpnn.children())[:-1]
        self.cutoff_mpnn = torch.nn.Sequential(*self.cutoff_mpnn)

        # Freezing the gradients on the cutoff_mpnn.
        for param in self.cutoff_mpnn.parameters():
            param.requires_grad = False

        # Separating out the MPNN modules.
        self.message_func_single = self.cutoff_mpnn[0]
        self.message_func_double = self.cutoff_mpnn[1]
        self.message_func_triple = self.cutoff_mpnn[2]
        self.message_func_aromatic = self.cutoff_mpnn[3]
        self.message_func_universal = self.cutoff_mpnn[4]
        
        self.update_func_catchall = self.cutoff_mpnn[5]
        self.update_func = self.cutoff_mpnn[6]

        self.update_func_catchall_universal = self.cutoff_mpnn[7]
        self.update_func_universal = self.cutoff_mpnn[8]

        self.selectivity = nn.Sequential(
            nn.Linear(self.rxn_features_length + self.message_size, self.rxn_features_length + self.message_size),
            nn.ReLU(),
            nn.Linear(self.rxn_features_length + self.message_size, 1),
        )

    def forward(self, h, g, rxn_vector):

        batch_size = g.size()[0]

        # Padding the atomic representations to some higher dimension, d = message size.
        h_t = torch.cat([h, torch.zeros(h.size()[0], h.size()[1], self.message_size - h.size()[2]).type_as(h.data)], 2)

        # Finding the order of atoms for the input.
        atom_numbers = h[:, :, len(self.all_unique_atoms)].view([-1])

        # Message Passing Loop
        for i in range(self.message_passes):
            # Running the padded atomic information through the linear layer.
            h_t_single = self.message_func_single(h_t)
            h_t_double = self.message_func_double(h_t)
            h_t_triple = self.message_func_triple(h_t)
            h_t_aromatic = self.message_func_aromatic(h_t)
            h_t_universal = self.message_func_aromatic(h_t)

            # Matrix multiply adjacency matrix of each bond type with embedded hidden atomic reps.
            m_single = torch.bmm(g[:, 0], h_t_single)
            m_double = torch.bmm(g[:, 1], h_t_double)
            m_triple = torch.bmm(g[:, 2], h_t_triple)
            m_aromatic = torch.bmm(g[:, 3], h_t_aromatic)
            m_universal = torch.bmm(g[:, 4], h_t_universal)

            # Add the messages from each bond type together for each atom. For the full message.
            m = m_single + m_double + m_triple + m_aromatic

            # Running the GRUs more efficiently.
            h_no_batches = h_t.view([-1, h_t.size()[2]])
            m_no_batches = m.view([-1, m.size()[2]])
            m_uni_no_batches = m_universal.view([-1, m.size()[2]])

            gru_output = torch.empty_like(h_no_batches)
            gru_uni_output = torch.empty_like(h_no_batches)

for atom_type in torch.unique(atom_numbers):
                # Find the rows that correspond to that atom type.
                h_atom_type_subset = h_no_batches.index_select(0, torch.where(atom_numbers == atom_type)[0])
                m_atom_type_subset = m_no_batches.index_select(0, torch.where(atom_numbers == atom_type)[0])
                m_uni_atom_type_subset = m_uni_no_batches.index_select(0, torch.where(atom_numbers == atom_type)[0])

                # If that atom type is in the top 4 atoms, run through the specific GRUs.
                if atom_type in self.top_4_unique_atoms:
                    idx = str(int(atom_type.detach().cpu().numpy()))

                    gru_atom_type_subset = self.update_func[idx](h_atom_type_subset, m_atom_type_subset)
                    gru_uni_atom_type_subset = self.update_func_universal[idx](h_atom_type_subset,
                                                                               m_uni_atom_type_subset)

                    gru_output[torch.where(atom_numbers == atom_type)[0]] = gru_atom_type_subset
                    gru_uni_output[torch.where(atom_numbers == atom_type)[0]] = gru_uni_atom_type_subset

                # Else run that atom through the catchall GRU.
                else:
                    gru_atom_type_subset = self.update_func_catchall(h_atom_type_subset, m_atom_type_subset)
                    gru_uni_atom_type_subset = self.update_func_catchall_universal(h_atom_type_subset,
                                                                                   m_uni_atom_type_subset)

                    gru_output[torch.where(atom_numbers == atom_type)[0]] = gru_atom_type_subset
                    gru_uni_output[torch.where(atom_numbers == atom_type)[0]] = gru_uni_atom_type_subset

            # Adding the universal bond hidden states to the hidden states from the
            # single/double/triple/aromatic bond states.
            gru_total = gru_output + gru_uni_output

            # Putting the batches back in for rxn vector concatenation.
            h_t = gru_total.view([batch_size, h_t.size()[1], h_t.size()[2]])

        # Concatenating with rxn vector.
        embedded_rxns = torch.cat([h_t, rxn_vector], 2)

        # Run through fine tuning layers.
        output = self.selectivity(embedded_rxns)
        output = nn.Sigmoid()(output)
        output = output.view([-1])

        return output
