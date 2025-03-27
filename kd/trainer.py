import torch
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)

def hidden_state_distillation_trainer(dataloader, teacher_model, student_model, device='cuda', epochs=10, lr=5e-5):
    """ Performs hidden state distillation using Uniform+Last mapping. """
    teacher_model.to(device)
    student_model.to(device)

    # Ensure models have hidden state output enabled
    teacher_model.config.output_hidden_states = True
    student_model.config.output_hidden_states = True

    projection = torch.nn.Linear(student_model.config.hidden_size,
                                 2 * teacher_model.config.hidden_size).to(device)
    optimizer = torch.optim.AdamW(list(student_model.parameters()) + list(projection.parameters()), lr=lr)

    teacher_model.eval() # Teacher is frozen

    total_epoch_loss = 0.0

    for epoch in range(epochs):
        student_model.train() # Student and projection are trained
        projection.train()
        running_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                output_hidden_states=True)
            student_outputs = student_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            output_hidden_states=True)

            # Extract hidden states (skip embedding layer output at index 0)
            teacher_hidden = teacher_outputs.hidden_states[1:]
            student_hidden = student_outputs.hidden_states[1:]

            num_student_layers = len(student_hidden)
            num_teacher_layers = len(teacher_hidden)

            all_student_Hs = torch.stack(student_hidden)
            projected_Hs = projection(all_student_Hs)

            target_teacher_list = []
            uniform_interval = num_teacher_layers / num_student_layers

            for i in range(num_student_layers):
                idx_uniform = min(math.floor(i * uniform_interval), num_teacher_layers - 1)
                # Ensure idx_last is non-negative
                idx_last = min(max(0, (num_teacher_layers - num_student_layers) + i), num_teacher_layers - 1)

                H_uniform = teacher_hidden[idx_uniform]
                H_last = teacher_hidden[idx_last]

                H_teacher_concat = torch.cat([H_uniform, H_last], dim=-1)
                target_teacher_list.append(H_teacher_concat)

            target_teacher = torch.stack(target_teacher_list)
            loss = F.mse_loss(projected_Hs, target_teacher)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        epoch_avg_loss = running_loss / num_batches if num_batches > 0 else 0
        logger.info(f"Epoch {epoch+1}/{epochs} | Avg Distillation Loss: {epoch_avg_loss:.4f}")
        total_epoch_loss += epoch_avg_loss

    # Return the average loss over all epochs
    return total_epoch_loss / epochs if epochs > 0 else 0
