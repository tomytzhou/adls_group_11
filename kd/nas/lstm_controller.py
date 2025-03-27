import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class Controller(nn.Module):
    """ LSTM Controller for NAS """
    def __init__(self, search_space, state_keys, embedding_dim=32, lstm_hidden=32):
        super(Controller, self).__init__()
        self.embeddings = nn.ModuleDict()
        self.param_to_idx = {}
        self.idx_to_param = {}
        self.state_keys = state_keys

        # Create integer mappings for categorical choices
        total_params = 0
        for i, (param, choices) in enumerate(search_space.items()):
            self.param_to_idx[param] = {choice: j for j, choice in enumerate(choices)}
            self.idx_to_param[param] = {j: choice for j, choice in enumerate(choices)}
            self.embeddings[param] = nn.Embedding(len(choices), embedding_dim)
            total_params += len(choices)

        input_size = embedding_dim * len(search_space)
        self.lstm = nn.LSTM(input_size, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, 1) # Predicts reward

    def config_to_indices(self, config):
        """ Converts a config dictionary to a list of indices """
        indices = []
        for key in self.state_keys:
            value = config[key]
            indices.append(self.param_to_idx[key][value])
        return indices

    def indices_to_config(self, indices):
        """ Converts a list of indices back to a config dictionary """
        config = {}
        for i, key in enumerate(self.state_keys):
            index = indices[i]
            config[key] = self.idx_to_param[key][index]
        return config

    def forward(self, prev_best_indices, global_best_indices, current_indices, device='cuda'):
        """ Predict reward for current_indices given prev_best and global_best indices. """
        # Embed indices
        def get_embedding(indices_list):
            embs = []
            for i, key in enumerate(self.state_keys):
                idx_tensor = torch.tensor([indices_list[i]], device=device)
                embs.append(self.embeddings[key](idx_tensor))
            return torch.cat(embs, dim=-1)

        prev_best_emb = get_embedding(prev_best_indices)
        global_best_emb = get_embedding(global_best_indices)
        current_state_emb = get_embedding(current_indices)

        # LSTM input sequence: (batch=1, seq=3, features)
        sequence = torch.stack([prev_best_emb, global_best_emb, current_state_emb], dim=1)

        _, (h_n, _) = self.lstm(sequence)
        # Use the hidden state of the last LSTM step
        reward_pred = self.fc(h_n[-1])
        return reward_pred.squeeze()


def train_controller(controller, optimizer, training_data_indices, prev_best_indices, global_best_indices, device='cuda'):
    """Train the controller to predict rewards using MSE loss."""
    controller.train()
    total_loss = 0.0
    for state_indices, reward in training_data_indices:
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)

        optimizer.zero_grad()
        reward_pred = controller(prev_best_indices, global_best_indices, state_indices, device)
        loss = nn.functional.mse_loss(reward_pred.unsqueeze(0), reward_tensor) # Match shapes
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(training_data_indices)
