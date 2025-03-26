# adls_group_11: Knowledge distillation with neural architacture search for language models
## Overview

The project is centered around designing an efficient optimization pipeline for knowledge distillation of language models. Unlike traditional KD, which focuses on a one-to-one teacher-to-student model compression, our pipeline uses neural architecture search (NAS) to perform KD on a pool of student models. This allows for discovery of more optimal network structures while also enable us to expand the usability of the KD pipeline to a range of different models.

Our pipeline offers the following improvements compare to traditional KD optimizations:
- Being able to perform KD on multiple student models that all belongs to the same or different categpries. For instance, KD can be performed with a random mix of bert model variants.
- Automatically select the best performing models for a certain language tasks. This means models no longer need to be pre-selected by the programmer in advance. Hence our pipeline can be used when the nature of the language task is unknown and the pipeline will automatically select the best models with minimal human intervention.
- Faster KD with a multi-stage KD process. This will be covered in more details in the functionality description section.
- More flexible training process. Parameters of the model can be easily changed to optimize for accuracy, speed or generalization.

## Functionality description
### Pipeline overview
Our pipeline can use either the built-in optimizer (optuna) or the reinforcement learning optimizer. The main steps of the KD process is summarized below:
1. Select a teacher model and a task dataset that can be used to train it.
2. Extract the architecture structure from the teacher model. Then use them to construct a search space for KD training on the student models.
3. Select a variety of student model choices.
4. For the first episode, using the search space to first randomly selects N candidate configs for student models.
5. Perform mini-KD on each candidate model. During the mini-KD training, each candidate model learns a mapping from the teacher model's layers to its own layers. The mini-KD training uses only 30% of the input dataset. Each candidate is trained for a few epochs (typically 4 or 5) and the loss at the last epoch is stored. The latency for each candidate is computed at the end of the training.
6. Using a reward function to calculate a performance score for each candidate models.
7. If using the optuna optimizer, the pipeline will automatically choose the more promising candidates from the search space. If using the RL optimizer, the candidate historys, including the global best, episode best and previous global best candidate configs are stored and passed in to a LSTM. The LSTM will be trained to predict the most promising candidate config.
8. At the end of each episode, decrease the exploration ratio so that in the end of the training, all candidate config will be chosen using the LSTM. This step is ignored for the optuna version.
9. At the end of the last episode, the global best model will be selected for post NAS training. This step can be skipped if one wish to prevent overfitting or if the model has a fast converging nature.
10. The trained best model will be tested using the glue score function.

![Screenshot1](https://github.com/tomytzhou/adls_group_11/blob/main/WhatsApp%20Image%202025-03-26%20at%2015.25.43_56d0a755.jpg)

### Data loader
This function uses a checkpoint of a pretrained model from Hugging face to load the dataset and the tokenizer.

### Construct teacher model
This function uses the previously loaded checkpoints to construct a teacher model to be used for the distillation process.

### Construct search space
Based on the teacher model, constrcut a dictionary of feasible architecture parameters for the student models. The config of the teacher model is used to create a range of config parameters to be used. This step is currently done manually.

### Mini-KD
For each hidden layers in the teacher model, we extract the hidden states of the layers. It does the same by extracting the hidden states from the student candidate model. And since the teacher layer hidden states will have equal or higher data width than the hidden states of the candidate models, we use a multi-layer mapping function and a projection layer to encode and align the hidden states of the teacher model to that of the student model. Then we conpute the mean square error between the 2 sets of hidden states. And finally we train the candidate model for a few epoches along with the projection layer. The candidate model will then be assigned a score, called reward, using its last epoch loss and its latency.

### Model ranking
After training all the selected candidate models during a single episode, we need to rank them based on their rewards. We store the best performing model and its config into the global best tuple. As well as the prevously best performing model in the last episode and its config. We also need to store the config of all the candidate models that we have selected in that episode. These cached models are then used to train the RL controller.

### LSTM unit
Our training loop includes a controller that uses past knowledge about model distillation to make prediction that selects the most promising candidates for the next episode. We used a LSTM unit as our controller. It takes as input the data of the global best model, the previous episode best model and all the candidates from the previous episode.

### Full KD trainer
This function is used to perform post NAS finetuning to the best performing model obtained from the NAS session. It works in the same way as the mini-KD function but uses the entire dataset.

### GLUE score calculator
The function that evaluates the gleu score for each best performing model category on the target language tasks. Within the function a selection of language tasks are selected and their corresponding datasets are loaded using the datasets library. The function first uses the trainer module from the transformer library to finetune the model for a few epoches. There also an option to disable finetuning. It then uses the trainer module's built-in evaluation function to compute a score for the datasets previously loaded.
