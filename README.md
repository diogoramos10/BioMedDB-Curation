# PubMedBERT Model Branch

This branch contains the code for the **PubMedBERT** model, designed for training and testing. The data required for this model is available in the main repository. 

## Docker Environment

A `Dockerfile` is provided in this branch to simplify the setup and execution of the environment. You can easily simulate the environment by building and running the Docker image created with this `Dockerfile`.

## Running the Code

All files in this repository can be executed using:

```bash
python <name_of_file>

However, for proper usage, only the following files should be directly run:

PubMedBERTTrain.py: This file handles the training of the PubMedBERT model.
PubMedBERT.py: This file is for testing the model and evaluating its performance.
The AuxClass.py file serves as an auxiliary class to support the above scripts and should not be run independently.

## Output Information

PubMedBERTTrain.py:

Running this script will produce a folder named results, which contains a checkpoint for each training epoch. These checkpoints can be used to restore or continue training later.

PubMedBERT.py:

Running this file will generate three outputs:
A confusion matrix with an overview of performance metrics.
A file containing each predicted class and score, along with the actual class and score.
A metrics file that contains the detailed evaluation results of the model.
