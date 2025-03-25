# adls_group_11: Knowledge distillation with neural architacture search for language models
## Overview

The project is centered around designing an efficient optimization pipeline for knowledge distillation of language models. Unlike traditional KD, which focuses on a one-to-one teacher-to-student model compression, our pipeline uses neural architecture search (NAS) to perform KD on a pool of student models. This allows for discovery of more optimal network structures while also enable us to expand the usability of the KD pipeline to a range of different models.

Our pipeline offers the following improvements compare to traditional KD optimizations:
- Being able to perform KD on multiple student models that all belongs to the same or different categpries. For instance, KD can be performed with a random mix of bert model variants.
- Automatically select the best performing models for a certain language tasks. This means models no longer need to be pre-selected by the programmer in advance. Hence our pipeline can be used when the nature of the language task is unknown and the pipeline will automatically select the best models with minimal human intervention.
- Faster KD with a multi-stage KD process. This will be covered in more details in the functionality description section.
- More flexible training process. Parameters of the model can be easily changed to optimize for accuracy, speed or generalization.

## Functionality description
The main steps of the KD process is summarized below:
1. d
