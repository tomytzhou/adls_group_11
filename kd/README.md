This script performs Knowledge Distillation guided Neural Architecture Search.

**Basic Usage:**

```bash
python kd_nas.py --search <strategy> [options]
```

**Required Arguments:**

*   `--search <strategy>`: NAS search strategy. Choices: `LSTM`, `Optuna`.

**Key Optional Arguments:**

*   `--teacher <model_id>`: Hugging Face identifier for the teacher model (default: `google-bert/bert-base-uncased`).
*   `--dataset <dataset_id>`: Hugging Face dataset identifier (default: `imdb`).
*   `--output_dir <path>`: Directory to save results (default: `./kd_nas_output`).
*   `--nas_only`: If set, only run NAS and save top architectures list; skips final student training.

**Strategy-Specific Arguments:**

*   `--lstm_episodes <int>`: Number of episodes for LSTM search (default: 15).
*   `--lstm_samples <int>`: Architectures per LSTM episode (default: 20).
*   `--optuna_trials <int>`: Number of trials for Optuna search (default: 100).

**Other Options:**

*   `--dataset_subset_size <int>`: Initial dataset samples for NAS (default: 50000).
*   `--mini_kd_epochs <int>`: Epochs for NAS evaluation KD (default: 4).
*   `--proxy_fraction <float>`: Dataset fraction for NAS evaluation KD (default: 0.2).
*   `--full_kd_epochs <int>`: Epochs for final student training (default: 20).
*   `--batch_size <int>`: Training batch size (default: 32).
*   `--learning_rate <float>`: KD training learning rate (default: 5e-5).
*   `--reward_alpha <float>`: Reward latency exponent (default: -0.07).
*   `--reward_beta <float>`: Reward latency scaling factor (default: 0.6).

**Examples:**

1.  **Run LSTM NAS only (quick test):**
    ```bash
    python kd_nas.py --search LSTM --nas_only --lstm_episodes 5 --lstm_samples 5
    ```

2.  **Run Optuna NAS and train the best student model:**
    ```bash
    python kd_nas.py --search Optuna --teacher FacebookAI/roberta-base --dataset xu-song/cc100-samples --optuna_trials 50 --output_dir ./roberta_nas_results
    ```
    *(Note: You might need to add dataset-specific args like `--dataset_name wikitext-2-raw-v1`)*

3.  **Run LSTM NAS and train the best student model with custom teacher:**
    ```bash
    python kd_nas.py --search LSTM --teacher distilbert/distilbert-base-uncased --dataset imdb --lstm_episodes 10 --output_dir ./distilbert_nas_lstm
    ```
```
