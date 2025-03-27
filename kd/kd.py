# === PyTorch & Standard Imports ===
import torch
import torch.nn as nn
import numpy as np
import time
import optuna
import copy

from torch.utils.data import DataLoader

# === Hugging Face Transformers Imports ===
from transformers import (
    AutoTokenizer,
    AutoConfig,
    DataCollatorForLanguageModeling,
    AutoModelForMaskedLM,
    AlbertConfig,
    AlbertForMaskedLM,
    BertConfig,
    BertForMaskedLM,
    RobertaConfig,
    RobertaForMaskedLM
    # AdamW  # Uncomment if using manually specified optimizer
)

# === Hugging Face Datasets Import ===
from datasets import load_dataset

# === Optuna Sampler for Hyperparameter Optimization ===
from optuna.samplers import TPESampler


# === Model and Dataset Configuration ===

# Available pre-trained model checkpoints (Uncomment the one you want to use)
# checkpoint = "albert/albert-base-v2"
# checkpoint = "bert-base-uncased"
# checkpoint = "roberta-base"
# checkpoint = "prajjwal1/bert-medium"
checkpoint = "prajjwal1/bert-small"  # Currently selected checkpoint

# Hugging Face dataset for training/testing language models
dataset_name = "xu-song/cc100-samples"


# === LSTM-Based Controller for Neural Architecture Search (NAS) ===
class Controller(nn.Module):
        """
        LSTM-based controller that predicts reward values for sampled architectures.
        It takes the current candidate configuration and compares it to the previous and global best configurations,
        then uses an LSTM to predict how promising the current configuration is.
        
        Inputs:
        - search_space: Dictionary where keys are architecture parameters (e.g., "hidden_dim")
                        and values are lists of discrete choices (e.g., [128, 256, 512])
        - embedding_dim: Dimension of embedding vectors for each parameter choice
        - lstm_hidden: Number of hidden units in the LSTM
        """

        def __init__(self, search_space, embedding_dim=32, lstm_hidden=32):
                super(Controller, self).__init__()

                # Create an embedding layer for each search space parameter
                self.embeddings = nn.ModuleDict()
                for param, choices in search_space.items():
                        # Each parameter choice gets its own embedding
                        self.embeddings[param] = nn.Embedding(len(choices), embedding_dim)

                # Input to LSTM is a sequence of 3 embeddings (prev_best, global_best, current)
                # Each embedding is a concatenation of all parameter embeddings
                input_size = embedding_dim * len(search_space)

                # Define LSTM
                self.lstm = nn.LSTM(input_size, lstm_hidden, batch_first=True)

                # Output layer: maps LSTM hidden state to a scalar reward
                self.fc = nn.Linear(lstm_hidden, 1)

                # Maintain consistent parameter ordering
                self.state_keys = list(search_space.keys())

        def forward(self, prev_best, global_best, current_state):
                """
                Forward pass of the controller.

                Inputs:
                - prev_best: List of indices representing the best configuration in the previous iteration
                - global_best: List of indices for the best configuration found so far
                - current_state: List of indices for the current sampled configuration

                Returns:
                - reward_pred: Predicted scalar reward for the current configuration
                """

                # === Embed the previous best configuration ===
                prev_best_emb = torch.cat([
                self.embeddings[param](torch.tensor([idx], device='cuda'))
                for param, idx in zip(self.state_keys, prev_best)
                ], dim=-1)  # Shape: (1, total_embedding_dim)

                # === Embed the global best configuration ===
                global_best_emb = torch.cat([
                self.embeddings[param](torch.tensor([idx], device='cuda'))
                for param, idx in zip(self.state_keys, global_best)
                ], dim=-1)

                # === Embed the current configuration ===
                current_state_emb = torch.cat([
                self.embeddings[param](torch.tensor([idx], device='cuda'))
                for param, idx in zip(self.state_keys, current_state)
                ], dim=-1)

                # === Stack the 3 embeddings to form an input sequence for LSTM ===
                # Shape: (3, total_embedding_dim)
                sequence = torch.stack([prev_best_emb, global_best_emb, current_state_emb], dim=0)

                # LSTM expects input of shape (batch_size, seq_len, input_size)
                # sequence.permute(1, 0, 2) => (1, 3, input_size)
                _, (h_n, _) = self.lstm(sequence.permute(1, 0, 2))

                # Take the final hidden state from the LSTM
                # h_n shape: (1, batch_size, lstm_hidden), we take h_n[-1] to remove num_layers dimension
                reward_pred = self.fc(h_n[-1])  # Shape: (batch_size, 1)

                return reward_pred.squeeze()  # Return scalar reward prediction


class KD:
        def __init__(self, model_type, checkpoint, dataset_name):
                # Initialize key attributes
                self.model_type = model_type

                # Choose appropriate model and configuration class based on the model type
                if self.model_type == "albert":
                        self.config_func = AlbertConfig
                        self.model_func = AlbertForMaskedLM
                elif self.model_type == "bert":
                        self.config_func = BertConfig
                        self.model_func = BertForMaskedLM
                elif self.model_type == "roberta":
                        self.config_func = RobertaConfig
                        self.model_func = RobertaForMaskedLM

                self.checkpoint = checkpoint
                self.dataset_name = dataset_name

                # Load and preprocess dataset
                dataset = load_dataset(self.dataset_name, "en", split="train[:100%]")
                tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

                def tokenize_function(example):
                        # Tokenize the text with truncation and padding
                        return tokenizer(
                                example["text"],
                                truncation=True,
                                padding="max_length",
                                max_length=50
                        )

                # Tokenize dataset and split into train/test
                dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
                dataset = dataset.train_test_split(test_size=0.2)
                self.train_data = dataset["train"]

                # Create data collator for masked language modeling
                data_collator = DataCollatorForLanguageModeling(
                        tokenizer=tokenizer,
                        mlm=True,
                        mlm_probability=0.15
                )
                self.data_collator = data_collator

                # Load teacher model and freeze its weights
                teacher_model = AutoModelForMaskedLM.from_pretrained(self.checkpoint)
                teacher_model.config.output_hidden_states = True
                teacher_model.eval()
                for param in teacher_model.parameters():
                        param.requires_grad = False
                self.teacher_model = teacher_model

                # Define model search space for NAS manually
                def get_search_space():
                        return {
                                "num_hidden_layers": [3, 4, 6, 10, 12],
                                "num_attention_heads": [2, 3, 4, 6, 12],
                                "hidden_size": [384, 768],
                                "intermediate_size": [384, 512, 576, 768, 1024, 1536, 2048, 3072],
                                "hidden_act": ['gelu', 'relu', 'silu']
                        }

                self.search_space = get_search_space()
                self.state_keys = list(self.search_space.keys())

                # DataLoader for training
                self.train_dataloader = DataLoader(self.train_data, batch_size=8, shuffle=True, collate_fn=self.data_collator)

        # --- Utility Methods ---
        def construct_student_model_from_config(self, config):
                # Construct a model from a config dictionary
                new_config = self.config_func(**config)
                model = self.model_func(new_config)
                return model

        def get_latency(self, config_or_model_config):
                """
                Estimate model inference latency on CPU.
                Uses dummy inputs and measures average latency over multiple runs.
                """
                if isinstance(config_or_model_config, dict):
                        model = self.construct_student_model_from_config(config_or_model_config)
                else:
                        model = self.model_func(config_or_model_config)

                model.eval().to('cpu')

                batch_size = 1
                seq_length = 50
                dummy_input = torch.randint(0, 100, (batch_size, seq_length))
                attention_mask = torch.ones_like(dummy_input)

                # Warmup
                with torch.no_grad():
                        for _ in range(5):
                                _ = model(dummy_input, attention_mask=attention_mask)

                # Timed runs
                num_runs = 20
                start = time.time()
                with torch.no_grad():
                        for _ in range(num_runs):
                                _ = model(dummy_input, attention_mask=attention_mask)
                end = time.time()

                avg_latency = (end - start) / num_runs
                return avg_latency

        def calculate_reward(self, loss, latency, teacher_latency, alpha=-0.06, beta=0.6**6):
                """
                Reward function used to score candidate architectures.
                Based on distillation loss and relative latency.
                """
                normalized_latency = latency / (beta * teacher_latency)
                reward = (1 - loss) * (normalized_latency ** alpha)
                return reward

        def mini_kd_trainer(self, proxy_data, teacher_model, student_model, epochs=5):
                """
                Lightweight knowledge distillation using a subset of training data (proxy_data).
                Returns average MSE loss between hidden states.
                """
                proxy_dataloader = DataLoader(proxy_data, batch_size=32, shuffle=True, collate_fn=self.data_collator)
                projection = nn.Linear(student_model.config.hidden_size, 2 * teacher_model.config.hidden_size).to('cuda')
                optimizer = torch.optim.AdamW(list(student_model.parameters()) + list(projection.parameters()), lr=1e-4)

                student_model.to('cuda')
                teacher_model.to('cuda')

                for epoch in range(epochs):
                        student_model.train()
                        running_loss = 0.0

                        for batch in proxy_dataloader:
                                input_ids = batch["input_ids"].to('cuda')
                                attention_mask = batch["attention_mask"].to('cuda')

                                with torch.no_grad():
                                        teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

                                student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

                                teacher_hidden = teacher_outputs.hidden_states[1:]
                                student_hidden = student_outputs.hidden_states[1:]
                                Hs = torch.stack(student_hidden)
                                Hs_proj = projection(Hs)

                                num_student_layers = len(student_hidden)
                                num_teacher_layers = len(teacher_hidden)

                                # Uniform + Last mapping for teacher layers
                                Ht = []
                                for i in range(num_student_layers):
                                        idx_uniform = int(i * num_teacher_layers / num_student_layers)
                                        idx_last = i + num_teacher_layers - num_student_layers
                                        H_teacher_concat = torch.cat([teacher_hidden[idx_uniform], teacher_hidden[idx_last]], dim=-1)
                                        Ht.append(H_teacher_concat)
                                Ht = torch.stack(Ht)

                                loss = nn.functional.mse_loss(Hs_proj, Ht)
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                running_loss += loss.item()

                        avg_loss = running_loss / len(proxy_dataloader)
                        print(f"Epoch {epoch+1}/{epochs} | Avg Distillation Loss: {avg_loss:.4f}")
                return avg_loss

        def train_controller(self, controller, optimizer, training_data, prev_best, global_best, epochs=10):
                """
                Train RL controller to predict rewards using MSE loss.
                """
                controller.train()
                for epoch in range(epochs):
                        running_loss = 0.0
                        for state, reward in training_data:
                                prev_best_tensor = torch.tensor(prev_best, device='cuda')
                                global_best_tensor = torch.tensor(global_best, device='cuda')
                                state_tensor = torch.tensor(state, device='cuda')
                                reward_tensor = torch.tensor([reward], dtype=torch.float32, device='cuda')

                                optimizer.zero_grad()
                                reward_pred = controller(prev_best_tensor, global_best_tensor, state_tensor)
                                loss = nn.functional.mse_loss(reward_pred.unsqueeze(0), reward_tensor)
                                loss.backward()
                                optimizer.step()
                                running_loss += loss.item()
                        avg_loss = running_loss / len(training_data)
                print(f"Controller Epoch {epochs} | Avg Loss: {avg_loss:.4f}")
                return avg_loss

        def trainer(self, teacher_model, student_model, epochs=10):
                """
                Full knowledge distillation training over multiple epochs.
                """
                projection = torch.nn.Linear(student_model.config.hidden_size, 2 * teacher_model.config.hidden_size).to('cuda')
                optimizer = torch.optim.AdamW(list(student_model.parameters()) + list(projection.parameters()), lr=5e-5)
                student_model.to('cuda')
                teacher_model.to('cuda')

                for epoch in range(epochs):
                        student_model.train()
                        running_loss = 0.0
                        for batch in self.train_dataloader:
                                input_ids = batch["input_ids"].to('cuda')
                                attention_mask = batch["attention_mask"].to('cuda')

                                with torch.no_grad():
                                        teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

                                student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

                                teacher_hidden = teacher_outputs.hidden_states[1:]
                                student_hidden = student_outputs.hidden_states[1:]

                                Hs = torch.stack(student_hidden)
                                Hs_proj = projection(Hs)

                                # Uniform+Last mapping
                                num_student_layers = len(student_hidden)
                                num_teacher_layers = len(teacher_hidden)
                                Ht = []
                                for i in range(num_student_layers):
                                        idx_uniform = int(i * num_teacher_layers / num_student_layers)
                                        idx_last = i + num_teacher_layers - num_student_layers
                                        H_teacher_concat = torch.cat([teacher_hidden[idx_uniform], teacher_hidden[idx_last]], dim=-1)
                                        Ht.append(H_teacher_concat)
                                Ht = torch.stack(Ht)

                                loss = torch.nn.functional.mse_loss(Hs_proj, Ht)

                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                running_loss += loss.item()

                        avg_loss = running_loss / len(self.train_dataloader)
                        print(f"Epoch {epoch+1} | Avg Distillation Loss: {avg_loss:.4f}")

                return avg_loss
        
        def optuna_construct_student_model(self, trial):
                config = copy.deepcopy(self.search_space)
                
                # Assign the candidate states to the config for the student models
                for param in self.search_space.keys():
                        param_idx = trial.suggest_int(param, 0, len(self.search_space[param]) - 1)
                        config[param] = self.search_space[param][param_idx]

                # Create the student model using the sampled configuration
                new_config = self.config_func(**config)
                trial_model = self.model_func(new_config)  # Initialize an untrained model using the config

                return trial_model

        def objective(self, trial):
                # Construct a student model using Optuna's trial
                student_model = self.optuna_construct_student_model(trial)

                # Train for 1 epoch using full KD and measure time and loss
                start = time.time()
                loss = self.trainer(self.teacher_model, student_model, epochs=1)
                training_time = time.time() - start

                print(f"Average loss: {loss}")
                print(f"Training  time: {training_time}")

                # Store model in trial for reference
                trial.set_user_attr("student_model", student_model)

                # Define reward based on loss and training time
                reward = 1 / (loss + training_time)
                print(reward)

                return reward

        def run_optuna_kd(self):
                # Set up and run Optuna-based NAS for KD
                sampler = TPESampler()
                study = optuna.create_study(direction="maximize", sampler=sampler)
                study.optimize(self.objective, n_trials=10)

        def run_rl_kd(self, M=1, N=5, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.05, pool_size=150, pdt=False):
                # Initialise controller model and optimizer
                controller = Controller(self.search_space).to('cuda')
                controller_optimizer = torch.optim.RMSprop(controller.parameters(), lr=1e-4)

                # Initialize random best states for controller guidance
                random_state = [np.random.randint(len(self.search_space[key])) for key in self.state_keys]
                global_best = {'state': random_state, 'reward': -float('inf')}
                previous_best = {'state': random_state, 'reward': -float('inf')}

                # Use 30% of training data for fast mini-KD training
                proxy_size = int(0.3 * len(self.train_data))
                proxy_data = self.train_data.shuffle().select(range(proxy_size))

                # Measure latency of the full teacher model
                teacher_latency = self.get_latency(self.teacher_model.config)

                evaluated_states = []

                # Initialize tracking variables
                best_model = None
                mini_kd_loss = []
                latencys = []
                rewards = []

                # Define search space memory dimensions and initialize cache
                memory_dim = [len(self.search_space[key]) for key in self.search_space.keys()]
                print(f'search space dimension: {memory_dim}')
                memory = np.zeros(memory_dim)

                # Begin NAS episodes
                for episode in range(M):
                        print(f"\nEpisode {episode+1}/{M}, Exploration Ratio: {epsilon:.2f}")
                        num_random = int(epsilon * N)
                        num_controller = N - num_random
                        candidate_states = []
                        candidate_models = []

                        # Exploitation: controller chooses promising candidates
                        if num_controller > 0:
                                pool_states = [[np.random.randint(len(self.search_space[key])) for key in self.state_keys]
                                        for _ in range(pool_size)]
                                predicted_rewards = []
                                for state in pool_states:
                                        with torch.no_grad():
                                                reward_pred = controller(previous_best['state'], global_best['state'], state)
                                        predicted_rewards.append(reward_pred.item())
                                indices = np.argsort(predicted_rewards)[-num_controller:]
                                for idx in indices:
                                        candidate_states.append(pool_states[idx])

                        # Exploration: generate random candidates
                        for _ in range(num_random):
                                state = [np.random.randint(len(self.search_space[key])) for key in self.state_keys]
                                candidate_states.append(state)

                        # Evaluate all candidates
                        episode_rewards = []
                        for state_idx, state in enumerate(candidate_states):
                                if memory[tuple(state)] != 0:
                                        print(f"\n Candidate {state_idx+1}/{N} is duplicate candidate. Training is skipped.")
                                        config = {key: self.search_space[key][idx] for key, idx in zip(self.state_keys, state)}
                                        print(f"Candidate {state_idx+1}/{N} Configuration:")
                                        for key, value in config.items():
                                                print(f"    {key}: {value}")
                                        student_model = self.construct_student_model_from_config(config)
                                        reward = memory[tuple(state)]
                                        episode_rewards.append(reward)
                                        evaluated_states.append({'state': state, 'config': config, 'reward': reward})
                                        print(f"    Reward: {reward:.4f}")
                                        rewards.append(reward)
                                else:
                                        config = {key: self.search_space[key][idx] for key, idx in zip(self.state_keys, state)}
                                        print(f"\n  Candidate {state_idx+1}/{N} Configuration:")
                                        for key, value in config.items():
                                                print(f"    {key}: {value}")
                                        student_model = self.construct_student_model_from_config(config)
                                        loss = self.mini_kd_trainer(proxy_data, self.teacher_model, student_model)
                                        latency = self.get_latency(config)
                                        reward = self.calculate_reward(loss, latency, teacher_latency)
                                        episode_rewards.append(reward)
                                        evaluated_states.append({'state': state, 'config': config, 'reward': reward})
                                        print(f"    Mini-KD Loss: {loss:.4f}, Latency: {latency:.4f}, Reward: {reward:.4f}")
                                        mini_kd_loss.append(loss)
                                        latencys.append(latency)
                                        rewards.append(reward)
                                        memory[tuple(state)] = reward
                                candidate_models.append(student_model)

                        # Update best models and states
                        best_idx = np.argmax(episode_rewards)
                        episode_best_state = candidate_states[best_idx]
                        episode_best_reward = episode_rewards[best_idx]
                        episode_best_model = candidate_models[best_idx]
                        previous_best = {'state': episode_best_state, 'reward': episode_best_reward}
                        print([episode_best_reward, global_best['reward']])
                        if episode_best_reward > global_best['reward']:
                                global_best = {'state': episode_best_state, 'reward': episode_best_reward}
                                best_model = episode_best_model

                        # Train controller on episode data + current bests
                        training_data = list(zip(candidate_states, episode_rewards))
                        training_data.append((global_best['state'], global_best['reward']))
                        training_data.append((previous_best['state'], previous_best['reward']))
                        self.train_controller(controller, controller_optimizer, training_data,
                                previous_best['state'], global_best['state'])

                        # Decay epsilon (exploration rate)
                        epsilon = max(epsilon_min, epsilon - epsilon_decay)

                # Optional: full KD training on best model found
                if pdt:
                        self.trainer(self.teacher_model, best_model)

                return best_model
