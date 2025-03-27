
import torch
import time
import logging
from transformers import AutoConfig, AutoModelForMaskedLM

logger = logging.getLogger(__name__)

def tokenize_function(example, tokenizer, text_column="text"):
    """Tokenizes a single example."""
    # Adjust text_column based on dataset structure if needed
    return tokenizer(
        example[text_column],
        truncation=True,
        padding="max_length",
        max_length=128 # Keep sequence length manageable
    )

def get_latency(model, dataloader, num_inference_runs=5):
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    # Warmup runs
    warmup_batch = next(iter(dataloader))
    for _ in range(5):
        input_ids = warmup_batch["input_ids"].to(device)
        attention_mask = warmup_batch["attention_mask"].to(device)
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attention_mask)

    # Measurement runs
    total_time = 0.0
    
    for i, batch in enumerate(dataloader):
        if i >= num_inference_runs:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start = time.time()
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attention_mask)
        total_time += time.time() - start

    avg_latency = total_time / num_inference_runs
    return avg_latency


def calculate_reward(loss, latency, teacher_latency, alpha=-0.06, beta=0.06):
    
    normalized_latency = latency / (beta * teacher_latency)

    latency_term = normalized_latency ** alpha
    # handle if loss is > 1
    loss_term = max(0.01, 1.0 - loss)

    reward = loss_term * latency_term
    return reward

def construct_student_model(config_dict, base_checkpoint):
    """Constructs student model from a configuration dictionary."""
    student_config = AutoConfig.from_pretrained(base_checkpoint)
    student_config.update(config_dict)
    model = AutoModelForMaskedLM.from_config(student_config)
    model.config.output_hidden_states = True
    return model
