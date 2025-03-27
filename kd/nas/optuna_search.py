import json
import logging


import optuna
from optuna.samplers import TPESampler

from utils import construct_student_model, get_latency, calculate_reward
from trainer import hidden_state_distillation_trainer

logger = logging.getLogger(__name__)

def objective(trial, args, teacher_model, teacher_latency, search_space,
              proxy_dataloader, latency_measure_dataloader):
    """ Optuna objective function. """
    logger.info(f"\n--- Optuna Trial {trial.number} ---")

    config_dict = {}

    config_dict["num_hidden_layers"] = trial.suggest_categorical("num_hidden_layers", search_space["num_hidden_layers"])
    config_dict["num_attention_heads"] = trial.suggest_categorical("num_attention_heads", search_space["num_attention_heads"])
    config_dict["hidden_size"] = trial.suggest_categorical("hidden_size", search_space["hidden_size"])

    # Ensure hidden_size is divisible by num_attention_heads - Resample heads if not
    retries = 0
    max_retries = 5
    while config_dict["hidden_size"] % config_dict["num_attention_heads"] != 0 and retries < max_retries:
        logger.debug(f"Hidden size {config_dict['hidden_size']} not divisible by heads {config_dict['num_attention_heads']}. Resampling heads (Trial {trial.number}).")
        config_dict["num_attention_heads"] = trial.suggest_categorical("num_attention_heads", search_space["num_attention_heads"])
        retries += 1

    config_dict["intermediate_size"] = trial.suggest_categorical("intermediate_size", search_space["intermediate_size"])
    config_dict["hidden_act"] = trial.suggest_categorical("hidden_act", search_space["hidden_act"])
    logger.info(f"  Suggested Config: {config_dict}")

    # Construct Student Model
    student_model = construct_student_model(config_dict, args.teacher)
    model_size = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    logger.info(f"  Model Size: {model_size / 1e6:.2f}M parameters")


    loss = hidden_state_distillation_trainer(
        proxy_dataloader, teacher_model, student_model, epochs=args.mini_kd_epochs, lr=args.learning_rate
    )
    latency = get_latency(student_model, latency_measure_dataloader)
    reward = calculate_reward(loss, latency, teacher_latency, args.reward_alpha, args.reward_beta)

    # Store Metadata
    trial.set_user_attr("config_str", json.dumps(config_dict)) # Store config as string
    trial.set_user_attr("loss", loss)
    trial.set_user_attr("latency", latency)
    trial.set_user_attr("size", model_size)

    return reward

def run_optuna_nas(args, device, teacher_model, teacher_latency, search_space,
                   proxy_dataloader, latency_measure_dataloader):
    """ Runs the NAS process using Optuna. """

    logger.info("--- Starting NAS with Optuna ---")
    sampler = TPESampler(seed=19) 
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(
        lambda trial: objective(trial, args, teacher_model, teacher_latency, search_space,
                                proxy_dataloader, latency_measure_dataloader),
        n_trials=args.optuna_trials
    )

    logger.info("--- Optuna Study Complete ---")
    logger.info(f"Number of finished trials: {len(study.trials)}")

    results = []
    for trial in study.trials:
        config_str = trial.user_attrs.get("config_str", "{}")
        config = json.loads(config_str)
        results.append({
            'config': config,
            'reward': trial.value,
            'loss': trial.user_attrs.get("loss", float('inf')),
            'latency': trial.user_attrs.get("latency", float('inf')),
            'size': trial.user_attrs.get("size", 0)
        })

    return results
