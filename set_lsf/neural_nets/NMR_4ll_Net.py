'''
A top 4 atom GRU, 4 linear layer Pretraining network for NMR shifts.
'''

import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

class NMR_MPNN(nn.Module):
    '''
    Edge based message passing neural network. Messages are passed via an neural network
    based on the edge type (single, double, triple, aromatic) and updated via a GRU.
    ----
    Args:
        message_size (int):
            The size of the message (zeros padding added as needed). This must be greater than or
            equal to the number of atomic features.

        message_passes (int):
            The number of times to run the message passing.

        all_unique_atoms (list):
            The list of unique atoms' atomic numbers.

    Forward Args:
        g (tensor):
            The 5D array of the adjacency matrices based on edge type.
            G has size batch size x number of bond types x longest molecule x longest molecule.

        h (tensor):
            The initial atomic features for each atom.
            H has size batch size x longest molecule x atomic features length

    '''
    def __init__(self, message_size, message_passes, all_unique_atoms):
        super(NMR_MPNN, self).__init__()

        self.message_size = message_size
        self.message_passes = message_passes
        self.all_unique_atoms = all_unique_atoms
        self.top_4_unique_atoms = self.all_unique_atoms[0:4] + [0]

        self.message_func_single = nn.Sequential(
            nn.Linear(self.message_size, self.message_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.message_size, self.message_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.message_size, self.message_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.message_size, self.message_size, bias=False),
        )
        self.message_func_double = nn.Sequential(
            nn.Linear(self.message_size, self.message_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.message_size, self.message_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.message_size, self.message_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.message_size, self.message_size, bias=False),
        )
        self.message_func_triple = nn.Sequential(
            nn.Linear(self.message_size, self.message_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.message_size, self.message_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.message_size, self.message_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.message_size, self.message_size, bias=False),
        )
        self.message_func_aromatic = nn.Sequential(
            nn.Linear(self.message_size, self.message_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.message_size, self.message_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.message_size, self.message_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.message_size, self.message_size, bias=False),
        )
        self.message_func_universal = nn.Sequential(
            nn.Linear(self.message_size, self.message_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.message_size, self.message_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.message_size, self.message_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.message_size, self.message_size, bias=False),
        )

        # Define an update function for the top 4 unique atom types. Note that 0 refers to
        # "dummy atoms" - atoms that are there for padding purposes.

        # Using separate GRUs for atoms and uni nodes.

        self.update_func_catchall = nn.GRUCell(self.message_size, self.message_size)
        self.update_func = nn.ModuleDict({'0' : nn.GRUCell(self.message_size, self.message_size)})

        self.update_func_catchall_universal = nn.GRUCell(self.message_size, self.message_size)
        self.update_func_universal = nn.ModuleDict({'0': nn.GRUCell(self.message_size, self.message_size)})

        for i in range(len(self.top_4_unique_atoms)):
            self.update_func[str(self.top_4_unique_atoms[i])] = nn.GRUCell(self.message_size, self.message_size)
            self.update_func_universal[str(self.top_4_unique_atoms[i])] = nn.GRUCell(self.message_size, self.message_size)

        # Run through linear layers to predict site selectivity for each atom.
        self.nmr_shifts = nn.Sequential(
            nn.Linear(self.message_size, self.message_size),
            nn.ReLU(),
            nn.Linear(self.message_size, self.message_size),
            nn.ReLU(),
            nn.Linear(self.message_size, self.message_size),
            nn.ReLU(),
            nn.Linear(self.message_size, 1),
        )

    def forward(self, g, h):
        batch_size = g.size()[0]

        # Padding the atomic representations to some higher dimension, d = message size.
        h_t = torch.cat([h, torch.zeros(h.size()[0], h.size()[1], self.message_size - h.size()[2]).type_as(h.data)], 2)

        # Finding the order of atoms for the input.
        atom_numbers = h[:,:,len(self.all_unique_atoms)].view([-1])

        # Message Passing Loop
        for i in range(self.message_passes):
            # Running the padded atomic information through the linear layer.
            h_t_single = self.message_func_single(h_t)
            h_t_double = self.message_func_double(h_t)
            h_t_triple = self.message_func_triple(h_t)
            h_t_aromatic = self.message_func_aromatic(h_t)
            h_t_universal = self.message_func_aromatic(h_t)

            # Matrix multiply adjacency matrix of each bond type with embedded hidden atomic reps.
            m_single = torch.bmm(g[:,0], h_t_single)
            m_double = torch.bmm(g[:,1], h_t_double)
            m_triple = torch.bmm(g[:,2], h_t_triple)
            m_aromatic = torch.bmm(g[:,3], h_t_aromatic)
            m_universal = torch.bmm(g[:,4], h_t_universal)

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
                    gru_uni_atom_type_subset = self.update_func_universal[idx](h_atom_type_subset, m_uni_atom_type_subset)

                    gru_output[torch.where(atom_numbers == atom_type)[0]] = gru_atom_type_subset
                    gru_uni_output[torch.where(atom_numbers == atom_type)[0]] = gru_uni_atom_type_subset

                # Else run that atom through the catchall GRU.
                else:
                    gru_atom_type_subset = self.update_func_catchall(h_atom_type_subset, m_atom_type_subset)
                    gru_uni_atom_type_subset = self.update_func_catchall_universal(h_atom_type_subset, m_uni_atom_type_subset)

                    gru_output[torch.where(atom_numbers == atom_type)[0]] = gru_atom_type_subset
                    gru_uni_output[torch.where(atom_numbers == atom_type)[0]] = gru_uni_atom_type_subset

            # Adding the universal bond hidden states to the hidden states from the
            # single/double/triple/aromatic bond states.
            gru_total = gru_output + gru_uni_output

            # Putting the batches back in for rxn vector concatenation.
            h_t = gru_total.view([batch_size, h_t.size()[1], h_t.size()[2]])

        # Run the h_t's through the nmr prediction layers and a hyperbolic tangent function to output.

        pred_nmr_shifts = self.nmr_shifts(h_t)
        output = nn.Tanh()(pred_nmr_shifts)
        output = output.view([batch_size, -1])
        return output