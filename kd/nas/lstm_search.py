import torch
import numpy as np
import logging

from .lstm_controller import Controller, train_controller
from utils import construct_student_model, get_latency, calculate_reward
from trainer import hidden_state_distillation_trainer

logger = logging.getLogger(__name__)

def run_lstm_nas(args, device, teacher_model, teacher_latency, search_space,
                 proxy_dataloader, latency_measure_dataloader):
    """ Runs the NAS process using the LSTM controller. """
    
    logger.info("--- Starting NAS with LSTM Controller ---")
    state_keys = list(search_space.keys())
    controller = Controller(search_space, state_keys).to(device)
    controller_optimizer = torch.optim.RMSprop(controller.parameters(), lr=1e-4) # As per paper

    # Initialise best states (use indices for controller)
    initial_random_config = {key: np.random.choice(search_space[key]) for key in state_keys}

    initial_indices = controller.config_to_indices(initial_random_config)
    global_best = {'indices': initial_indices, 'reward': -float('inf')}
    previous_best = {'indices': initial_indices, 'reward': -float('inf')}

    evaluated_configs = {} # Store results: key=tuple(config), value=dict(reward, loss, latency, size)

    epsilon = 1.0
    epsilon_min = 0.05

    for episode in range(args.lstm_episodes):
        logger.info(f"\nLSTM Episode {episode+1}/{args.lstm_episodes}, Exploration Ratio: {epsilon:.2f}")
        num_random = int(epsilon * args.lstm_samples)
        num_controller = args.lstm_samples - num_random
        candidate_indices = [] # Store lists of indices

        # Exploitation: Controller predicts high reward states
        if num_controller > 0:
            # Sample a pool of random candidate indices
            pool_indices = []
            for _ in range(args.lstm_samples * 5): # Larger pool for better selection
                config = {key: np.random.choice(search_space[key]) for key in state_keys}
                # Ensure validity
                while config["hidden_size"] % config["num_attention_heads"] != 0:
                    config = {key: np.random.choice(search_space[key]) for key in state_keys}
                pool_indices.append(controller.config_to_indices(config))

            predicted_rewards = []
            controller.eval() # Set controller to evaluation mode for prediction
            with torch.no_grad():
                for indices in pool_indices:
                    reward_pred = controller(previous_best['indices'], global_best['indices'], indices, device)
                    predicted_rewards.append(reward_pred.item())

            # Select top N indices based on predicted reward
            sorted_pool_indices = [p_idx for _, p_idx in sorted(zip(predicted_rewards, pool_indices), key=lambda pair: pair[0], reverse=True)]
            candidate_indices.extend(sorted_pool_indices[:num_controller])

        # Exploration: Random states (indices)
        for _ in range(num_random):
            config = {key: np.random.choice(search_space[key]) for key in state_keys}
            # Ensure validity
            while config["hidden_size"] % config["num_attention_heads"] != 0:
                config = {key: np.random.choice(search_space[key]) for key in state_keys}
            candidate_indices.append(controller.config_to_indices(config))

        # Evaluate candidates
        episode_rewards_indices = [] # Store (reward, indices) tuples
        current_episode_training_data = [] # For training controller: (state_indices, reward)

        for i, indices in enumerate(candidate_indices):
            config = controller.indices_to_config(indices)
            config_key = tuple(sorted(config.items())) # Use sorted tuple as key

            if config_key in evaluated_configs:
                result = evaluated_configs[config_key]
                logger.info(f"  Candidate {i+1}/{args.lstm_samples} (cached): Reward={result['reward']:.4f}")
            else:
                logger.info(f"  Candidate {i+1}/{args.lstm_samples}: Evaluating {config}")
                student_model = construct_student_model(config, args.teacher)
                model_size = sum(p.numel() for p in student_model.parameters() if p.requires_grad)

                # Mini-KD
                loss = hidden_state_distillation_trainer(
                    proxy_dataloader, teacher_model, student_model, epochs=args.mini_kd_epochs, lr=args.learning_rate
                )

                # Latency
                latency = get_latency(student_model, latency_measure_dataloader)

                # Reward
                reward = calculate_reward(loss, latency, teacher_latency, args.reward_alpha, args.reward_beta)

                result = {'config': config, 'reward': reward, 'loss': loss, 'latency': latency, 'size': model_size}
                evaluated_configs[config_key] = result
                logger.info(f"    Result: Reward={reward:.4f}, Loss={loss:.4f}, Latency={latency:.6f}s, Size={model_size/1e6:.2f}M")

            episode_rewards_indices.append((result['reward'], indices))
            current_episode_training_data.append((indices, result['reward']))

        # Update best states for this episode and globally (use indices)
        episode_rewards_indices.sort(key=lambda x: x[0], reverse=True)
        best_episode_reward, best_episode_indices = episode_rewards_indices[0]
        previous_best = {'indices': best_episode_indices, 'reward': best_episode_reward}
        if best_episode_reward > global_best['reward']:
            global_best = {'indices': best_episode_indices, 'reward': best_episode_reward}
            logger.info(f"  New global best reward: {global_best['reward']:.4f}")

        # Train controller
        current_episode_training_data.append((global_best['indices'], global_best['reward']))
        current_episode_training_data.append((previous_best['indices'], previous_best['reward']))
        controller_loss = train_controller(controller, controller_optimizer, current_episode_training_data,
                                            previous_best['indices'], global_best['indices'], device)
        logger.info(f"  Controller training loss: {controller_loss:.4f}")

        # Decay exploration ratio
        epsilon = max(epsilon_min, epsilon - args.epsilon_decay)

    # Return list of evaluated config dictionaries with their results
    return list(evaluated_configs.values())
