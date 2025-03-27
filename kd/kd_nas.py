import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    AutoModelForMaskedLM,
)
from datasets import load_dataset
import logging
import argparse
import os
import json

from utils import tokenize_function, get_latency, construct_student_model
from search_space import get_search_space_auto
from trainer import hidden_state_distillation_trainer
from nas.lstm_search import run_lstm_nas
from nas.optuna_search import run_optuna_nas


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main(args):
    """ Main function to orchestrate the KD-NAS process. """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        logger.error("Need CUDA")

    logger.info(f"Loading dataset '{args.dataset}' and tokenizer '{args.teacher}'...")
    try:
        if args.dataset.lower() == "xu-song/cc100-samples":
            raw_dataset = load_dataset(args.dataset, "en", split=f"train[:{args.dataset_subset_size}]")
        else:
            raw_dataset = load_dataset(args.dataset, split=f"train[:{args.dataset_subset_size}]")
        tokenizer = AutoTokenizer.from_pretrained(args.teacher)
        text_column = "text" # Default

    except Exception as e:
        logger.error(f"Failed to load dataset or tokenizer: {e}", exc_info=True)
        exit(1)

    tokenized_dataset = raw_dataset.map(
        lambda ex: tokenize_function(ex, tokenizer, text_column),
        batched=True,
        remove_columns=raw_dataset.column_names # Remove original columns
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    full_dataloader = DataLoader(
        tokenized_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator
    )
    proxy_size = int(args.proxy_fraction * len(tokenized_dataset))
    proxy_data = tokenized_dataset.select(range(proxy_size))
    proxy_dataloader = DataLoader(
        proxy_data, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator
    )
    # Dataloader for consistent latency measurement (small, fixed subset, batch size 1)
    latency_subset_size = min(100, len(proxy_data))
    latency_measure_dataloader = DataLoader(
        proxy_data.select(range(latency_subset_size)),
        batch_size=1,
        collate_fn=data_collator
    )
    logger.info(f"Full dataset size: {len(tokenized_dataset)}, Proxy dataset size: {proxy_size}, Latency measurement batches: {latency_subset_size}")

    # Load Teacher Model and Calculate Latency
    teacher_model = AutoModelForMaskedLM.from_pretrained(args.teacher)
    teacher_model.eval()
    teacher_model.to(device)
    for param in teacher_model.parameters():
        param.requires_grad = False # Freeze teacher

    teacher_latency = get_latency(teacher_model, latency_measure_dataloader)
    logger.info(f"Teacher latency: {teacher_latency:.6f}s")

    search_space = get_search_space_auto(teacher_model)

    # Run NAS
    if args.search.lower() == 'lstm':
        nas_results = run_lstm_nas(args, device, teacher_model, teacher_latency, search_space,
                                   proxy_dataloader, latency_measure_dataloader)
    elif args.search.lower() == 'optuna':
        nas_results = run_optuna_nas(args, device, teacher_model, teacher_latency, search_space,
                                     proxy_dataloader, latency_measure_dataloader)
    else:
        raise ValueError(f"Unknown search strategy: {args.search}")

    if not nas_results:
        logger.error("NAS process finished with no valid results. Exiting.")
        exit(1)

    # Sort results by reward
    nas_results.sort(key=lambda x: x.get('reward', -float('inf')), reverse=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # Output Top 10 if --nas-only
    if args.nas_only:
        top_10 = nas_results[:10]
        output_file = os.path.join(args.output_dir, "top_nas_architectures.txt")
        logger.info(f"NAS Only mode: Saving top {len(top_10)} architectures to {output_file}")
        with open(output_file, 'w') as f:
            f.write(f"Top {len(top_10)} Architectures from {args.search.upper()} NAS\n")
            f.write(f"Teacher: {args.teacher}, Dataset: {args.dataset}\n")
            f.write("="*30 + "\n")
            for i, result in enumerate(top_10):
                f.write(f"Rank {i+1}:\n")
                f.write(f"  Reward: {result.get('reward', 'N/A'):.4f}\n")
                f.write(f"  Loss (Mini-KD): {result.get('loss', 'N/A'):.4f}\n")
                f.write(f"  Latency: {result.get('latency', 'N/A'):.6f}s\n")
                f.write(f"  Size (Params): {result.get('size', 0)/1e6:.2f}M\n")
                # Pretty print config
                config_str = json.dumps(result.get('config', {}), indent=4)
                f.write(f"  Config:\n{config_str}\n")
                f.write("-"*30 + "\n")
        logger.info("Top architectures saved.")

    # Train final model if not --nas-only
    else:
        best_result = nas_results[0]
        best_config = best_result.get('config')

        logger.info("\n--- Training Final Best Architecture ---")
        logger.info(f"Best Config Found (Reward: {best_result.get('reward', 'N/A'):.4f}):")
        logger.info(json.dumps(best_config, indent=2))

        final_student_model = construct_student_model(best_config, args.teacher)
        logger.info(f"Final student model param count: {sum(p.numel() for p in final_student_model.parameters() if p.requires_grad) / 1e6:.2f}M")

        final_avg_loss = hidden_state_distillation_trainer(
            full_dataloader, teacher_model, final_student_model, epochs=args.full_kd_epochs, lr=args.learning_rate
        )
        logger.info(f"Full KD Training Complete. Final Avg Loss: {final_avg_loss:.4f}")

        # Save the final model
        model_save_path = os.path.join(args.output_dir, "final_student_model")
        final_student_model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        # Save the config used as well
        with open(os.path.join(model_save_path, "best_nas_config.json"), 'w') as f:
            json.dump(best_config, f, indent=4)

    logger.info("\nKD-NAS process finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Architecture Search with Knowledge Distillation")

    # Required Arguments
    parser.add_argument("--teacher", type=str, default="google-bert/bert-base-uncased")
    parser.add_argument("--search", type=str, required=True, choices=['LSTM', 'Optuna'])

    # Optional Arguments - Data
    parser.add_argument("--dataset", type=str, default="imdb")
    parser.add_argument("--dataset_subset_size", type=int, default=50000)
    parser.add_argument("--output_dir", type=str, default="./kd_nas_output")

    # Optional Arguments - NAS Hyperparameters
    parser.add_argument("--lstm_episodes", type=int, default=15)
    parser.add_argument("--lstm_samples", type=int, default=20)
    parser.add_argument("--epsilon_decay", type=float, default=0.05)
    parser.add_argument("--optuna_trials", type=int, default=100)
    parser.add_argument("--mini_kd_epochs", type=int, default=4)
    parser.add_argument("--proxy_fraction", type=float, default=0.2)

    # Optional Arguments - Training Hyperparameters
    parser.add_argument("--full_kd_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)

    # Optional Arguments - Reward Calculation
    parser.add_argument("--reward_alpha", type=float, default=-0.07)
    parser.add_argument("--reward_beta", type=float, default=0.6)

    # Optional Arguments - Control Flow
    parser.add_argument("--nas_only", action='store_true',
                        help="If set, only run NAS and save top architectures, skip final training.")

    args = parser.parse_args()

    main(args)
