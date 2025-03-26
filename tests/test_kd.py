import sys
import os

# Add the ADLS_project directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
import torch.optim as optim
from ADLS_project.kd.kd import Controller, KD

kd = KD(model_type='bert', checkpoint="bert-base-uncased", dataset_name="xu-song/cc100-samples")  # or your actual args

def test_reward_function():
    reward = kd.calculate_reward(loss=0.2, latency=0.03, teacher_latency=0.01)
    assert isinstance(reward, float)

@pytest.fixture
def sample_search_space():
    return {
        "num_hidden_layers": [3, 6, 12],
        "num_attention_heads": [2, 4, 8],
        "hidden_size": [256, 512],
    }

@pytest.fixture
def controller(sample_search_space):
    model = Controller(search_space=sample_search_space)
    return model.cuda()  # Assuming your model expects CUDA

def test_controller_forward_output(controller):
    prev_best = [0, 1, 0]  # Indexes from the search space choices
    global_best = [2, 2, 1]
    current_state = [1, 0, 0]

    # Call the model's forward method
    output = controller(prev_best, global_best, current_state)

    # Check the output is a single scalar tensor
    assert isinstance(output, torch.Tensor)
    assert output.dim() == 0  # scalar

def test_student_model_construction():
    model_config = {
        "num_hidden_layers": 3,
        "num_attention_heads": 3,
        "hidden_size": 384,
        "intermediate_size": 384,
        "hidden_act": 'relu'
    }
    model = kd.construct_student_model_from_config(config=model_config)
    assert model is not None
    assert hasattr(model, 'forward')

def test_mini_distillation_returns_loss():
    print("Testing mini_kd_trainer() ...")
    student_config = {
        "num_layers": 1,
        "hidden_size": 1,
        "intermediate_size": 1,
        "num_attention_heads": 1,
    }
    student_model = kd.construct_student_model_from_config(config=student_config)
    proxy_size = int(0.1 * len(kd.train_data))
    proxy_data = kd.train_data.shuffle().select(range(proxy_size))
    loss = kd.mini_kd_trainer(proxy_data, kd.teacher_model, student_model, epochs=1)
    
    assert isinstance(loss, float)

def test_trainer():
    print("Testing mini_kd_trainer() ...")
    student_config = {
        "num_layers": 1,
        "hidden_size": 1,
        "intermediate_size": 1,
        "num_attention_heads": 1,
    }
    student_model = kd.construct_student_model_from_config(config=student_config)
    loss = kd.trainer(kd.teacher_model, student_model, epochs=1)
    
    assert isinstance(loss, float)

@pytest.fixture
def setup_controller_and_data():
    # Mock search space with small vocab size for simplicity
    search_space = {
        "num_layers": [1, 2],
        "hidden_size": [16, 32],
        "intermediate_size": [32, 64],
        "num_attention_heads": [1, 2],
    }

    controller = Controller(search_space).to('cuda')
    optimizer = optim.Adam(controller.parameters(), lr=0.01)

    # Create dummy training data
    training_data = []
    for _ in range(5):
        state = [0, 1, 0, 1]  # index into each parameter's embedding
        reward = torch.rand(1).item()  # random float between 0 and 1
        training_data.append((state, reward))

    prev_best = [1, 0, 1, 0]
    global_best = [1, 1, 1, 1]

    kd = KD(model_type='bert', checkpoint="bert-base-uncased", dataset_name="xu-song/cc100-samples")

    return kd, controller, optimizer, training_data, prev_best, global_best


def test_train_controller(setup_controller_and_data):
    kd, controller, optimizer, training_data, prev_best, global_best = setup_controller_and_data
    avg_loss = kd.train_controller(controller, optimizer, training_data, prev_best, global_best, epochs=1)

    assert isinstance(avg_loss, float)
    assert avg_loss >= 0.0


def test_get_latency():
    config = {
        "num_layers": 1,
        "hidden_size": 2,
        "intermediate_size": 20,
        "num_attention_heads": 2,
    }
    avg_latency = kd.get_latency(config)
    assert isinstance(avg_latency, float)
    assert avg_latency > 0