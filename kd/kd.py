import torch
import torch.nn as nn
import numpy as np
import time
import optuna
import copy

from torch.utils.data import DataLoader
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
    # AdamW
)
from datasets import load_dataset
from optuna.samplers import TPESampler


# checkpoint = "albert/albert-base-v2"
# tokenizer_checkpoint = "albert/albert-base-v2"
# checkpoint = "bert-base-uncased"
# tokenizer_checkpoint = "bert-base-uncased"
# checkpoint = "roberta-base"
# tokenizer_checkpoint = "roberta-base"
# checkpoint = "prajjwal1/bert-medium"
# tokenizer_checkpoint = "prajjwal1/bert-medium"
checkpoint = "prajjwal1/bert-small"
dataset_name = "xu-song/cc100-samples"


# LSTM Controller
class Controller(nn.Module):
        def __init__(self, search_space, embedding_dim=32, lstm_hidden=32):
                super(Controller, self).__init__()
                self.embeddings = nn.ModuleDict()
                for param, choices in search_space.items():
                        self.embeddings[param] = nn.Embedding(len(choices), embedding_dim)
                input_size = embedding_dim * len(search_space)
                self.lstm = nn.LSTM(input_size, lstm_hidden, batch_first=True)
                self.fc = nn.Linear(lstm_hidden, 1)
                self.state_keys = list(search_space.keys())

        def forward(self, prev_best, global_best, current_state):
                """Predict reward for current_state given prev_best and global_best."""
                prev_best_emb = torch.cat([self.embeddings[param](torch.tensor([idx], device='cuda'))
                                        for param, idx in zip(self.state_keys, prev_best)], dim=-1)
                global_best_emb = torch.cat([self.embeddings[param](torch.tensor([idx], device='cuda'))
                                        for param, idx in zip(self.state_keys, global_best)], dim=-1)
                current_state_emb = torch.cat([self.embeddings[param](torch.tensor([idx], device='cuda'))
                                        for param, idx in zip(self.state_keys, current_state)], dim=-1)
                sequence = torch.stack([prev_best_emb, global_best_emb, current_state_emb], dim=0)
                _, (h_n, _) = self.lstm(sequence.permute(1, 0, 2))
                reward_pred = self.fc(h_n[-1])
                return reward_pred.squeeze()

class KD:
        def __init__(self, model_type, checkpoint, dataset_name):
                self.model_type = model_type
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
     
                dataset = load_dataset(self.dataset_name, "en", split="train[:100%]")
                tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

                def tokenize_function(example):
                        return tokenizer(
                                example["text"],
                                truncation=True,
                                padding="max_length",
                                max_length=50
                        )

                # Tokenize and split dataset
                dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
                dataset = dataset.train_test_split(test_size=0.2)
                self.train_data = dataset["train"]

                data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=0.15
                )

                self.data_collator = data_collator

                # Load teacher model and ensure it outputs hidden states.
                teacher_model = AutoModelForMaskedLM.from_pretrained(self.checkpoint)
                teacher_model.config.output_hidden_states = True
                teacher_model.eval()  # Set teacher in evaluation mode
                for param in teacher_model.parameters():
                        param.requires_grad = False

                self.teacher_model = teacher_model

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

                self.train_dataloader = DataLoader(self.train_data, batch_size=8, shuffle=True, collate_fn=self.data_collator)

        # Utils
        def construct_student_model_from_config(self, config):
                new_config = self.config_func(**config)
                model = self.model_func(new_config)
                return model

        def get_latency(self, config_or_model_config):
                """
                Measure average latency (in seconds) for a forward pass.
                Use CPU as expected deployment on CPU and will probs get more stable results
                """
                # construct model
                if isinstance(config_or_model_config, dict):
                        model = self.construct_student_model_from_config(config_or_model_config)
                else:
                        model = self.model_func(config_or_model_config)
                model.eval()
                model.to('cpu')

                # dummy input
                batch_size = 1
                seq_length = 50
                dummy_input = torch.randint(0, 100, (batch_size, seq_length))
                attention_mask = torch.ones_like(dummy_input)

                # warmup
                with torch.no_grad():
                        for _ in range(5):
                                _ = model(dummy_input, attention_mask=attention_mask)

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
                reward function from paper:
                reward = (1 - L_HS) * (lat(S)/(beta * lat(T)))^alpha
                """
                normalized_latency = latency / (beta * teacher_latency)
                reward = (1 - loss) * (normalized_latency ** alpha)
                return reward
        
        def mini_kd_trainer(self, proxy_data, teacher_model, student_model, epochs=5):
                """Perform Mini-KD on proxy data and return average distillation loss."""
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
                                Hs = torch.stack([h for h in student_hidden])
                                Hs_proj = projection(Hs)
                                num_student_layers = len(student_hidden)
                                num_teacher_layers = len(teacher_hidden)
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
                """Train the controller to predict rewards using MSE loss."""
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
                                # loss = nn.functional.mse_loss(reward_pred, reward_tensor)
                                loss = nn.functional.mse_loss(reward_pred.unsqueeze(0), reward_tensor)
                                loss.backward()
                                optimizer.step()
                                running_loss += loss.item()
                        avg_loss = running_loss / len(training_data)
                print(f"Controller Epoch {epochs} | Avg Loss: {avg_loss:.4f}")
                return avg_loss

        def trainer(self, teacher_model, student_model, epochs=10):
                projection = torch.nn.Linear(student_model.config.hidden_size,
                                                2 * teacher_model.config.hidden_size).to('cuda')

                # Jointly optimise student model and projection - simplifcation compared to paper
                optimizer = torch.optim.AdamW(list(student_model.parameters()) + list(projection.parameters()), lr=5e-5)

                student_model.to('cuda')
                teacher_model.to('cuda')

                for epoch in range(epochs):
                        student_model.train()
                        running_loss = 0.0

                        for batch in self.train_dataloader:
                                input_ids = batch["input_ids"].to('cuda')
                                attention_mask = batch["attention_mask"].to('cuda')

                                # Forward pass through teacher (with no gradients).
                                with torch.no_grad():
                                        teacher_outputs = teacher_model(input_ids=input_ids,
                                                                        attention_mask=attention_mask,
                                                                        output_hidden_states=True)
                                student_outputs = student_model(input_ids=input_ids,
                                                                attention_mask=attention_mask,
                                                                output_hidden_states=True)

                                # Extract hidden states - skip embedding
                                teacher_hidden = teacher_outputs.hidden_states[1:]
                                student_hidden = student_outputs.hidden_states[1:]

                                # For each student layer, project the full hidden state
                                Hs = torch.stack([h for h in student_hidden])  # shape: (num_student_layers, batch, seq_len, hidden_size)
                                Hs_proj = projection(Hs)  # shape: (num_student_layers, batch, seq_len, 2 * teacher_hidden_size)

                                # Map teacher hidden states to student layers using a Uniform+Last strategy
                                num_student_layers = len(student_hidden)
                                teacher_layers = teacher_hidden
                                num_teacher_layers = len(teacher_layers)
                                Ht = []
                                for i in range(num_student_layers):
                                        # Uniform mapping index (adjust indices if using 0-indexing):
                                        idx_uniform = int(i * num_teacher_layers / num_student_layers)
                                        # Last mapping index:
                                        idx_last = i + num_teacher_layers - num_student_layers

                                        H0 = teacher_layers[idx_uniform]
                                        H1 = teacher_layers[idx_last]
                                        # Concatenate along the hidden dimension.
                                        H_teacher_concat = torch.cat([H0, H1], dim=-1)  # shape: (batch, seq_len, 2*teacher_hidden_size)
                                        Ht.append(H_teacher_concat)

                                # Stack teacher states to match student projection
                                Ht = torch.stack(Ht)  # shape: (num_student_layers, batch, seq_len, 2 * teacher_hidden_size)

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
                        param_idx = trial.suggest_int(param, 0, len(self.search_space[param])-1)
                        config[param] = config[param][param_idx]

                new_config = self.config_func(**config)
                trial_model =self.model_func(new_config) # Initialize an untrained bert model using the config

                return trial_model
        
        def objective(self, trial):
                student_model = self.optuna_construct_student_model(trial)

                start = time.time()
                loss = self.trainer(self.teacher_model, student_model, epochs=1)
                training_time = time.time() - start

                print(f"Average loss: {loss}")
                print(f"Training  time: {training_time}")

                trial.set_user_attr("student_model", student_model)
                reward = 1 / (loss + training_time)
                print(reward)

                return reward

        def run_optuna_kd(self):
                sampler = TPESampler()
                study = optuna.create_study(direction="maximize", sampler=sampler)  # Minimize loss
                study.optimize(self.objective, n_trials=10)

        def run_rl_kd(self, M=20, N=5, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.05, pool_size=150):
                # Initialise controller
                controller = Controller(self.search_space).to('cuda')
                controller_optimizer = torch.optim.RMSprop(controller.parameters(), lr=1e-4)

                # Initialise best states
                random_state = [np.random.randint(len(self.search_space[key])) for key in self.state_keys]
                global_best = {'state': random_state, 'reward': -float('inf')}
                previous_best = {'state': random_state, 'reward': -float('inf')}

                # Proxy dataset - 30% of original for shorter KD
                proxy_size = int(0.3 * len(self.train_data))
                proxy_data = self.train_data.shuffle().select(range(proxy_size))

                teacher_latency = self.get_latency(self.teacher_model.config)

                evaluated_states = []

                # NAS Episodes
                best_model = None
                mini_kd_loss = []
                latencys = []
                rewards = []
                avg_dist_loss = []
                memory = np.zeros([5, 5, 2, 8, 3]) # Cached mamoery to store the location of the rewards

                for episode in range(M):
                        print(f"\nEpisode {episode+1}/{M}, Exploration Ratio: {epsilon:.2f}")
                        num_random = int(epsilon * N)
                        num_controller = N - num_random
                        candidate_states = []
                        candidate_models = []

                        # Exploitation: Controller predicts high reward states
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

                        # Exploration: Random states
                        for _ in range(num_random):
                                state = [np.random.randint(len(self.search_space[key])) for key in self.state_keys]
                                candidate_states.append(state)

                        # Evaluate candidates
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

                # Update best states
                best_idx = np.argmax(episode_rewards)
                episode_best_state = candidate_states[best_idx]
                episode_best_reward = episode_rewards[best_idx]
                episode_best_model = candidate_models[best_idx]
                # print(episode_best_model)
                previous_best = {'state': episode_best_state, 'reward': episode_best_reward}
                print([episode_best_reward, global_best['reward']])
                if episode_best_reward > global_best['reward']:
                        global_best = {'state': episode_best_state, 'reward': episode_best_reward}
                        best_model = episode_best_model
                # print(best_model)
                # Train controller
                training_data = list(zip(candidate_states, episode_rewards))
                training_data.append((global_best['state'], global_best['reward']))
                training_data.append((previous_best['state'], previous_best['reward']))
                self.train_controller(controller, controller_optimizer, training_data,
                                previous_best['state'], global_best['state'])

                # Decay exploration ratio
                epsilon = max(epsilon_min, epsilon - epsilon_decay)

                return best_model