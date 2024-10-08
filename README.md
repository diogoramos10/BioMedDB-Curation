# Biomedical Document Retrieval for Database Curation - Exposome-Explorer

This repository provides a deep learning approach using BERT-based models to reduce a list of documents to the most relevant ones for the **Exposome-Explorer** database. The primary goal is to automate and enhance the document retrieval process by focusing on relevancy within the Exposome domain.

Different models will be used, with the code for each model located in its respective branch of this repository. The dataset required for this task is available in the **BM25 branch**. The dataset in combination with this deep learning model aims to improve document filtering efficiency by identifying relevant documents more accurately.

## How to Run

To execute the model, you simply need to run the `MonoBERT.py` file. This script handles the end-to-end process of tokenizing, processing documents, and generating predictions for document relevance.

### Command to run:

`python MonoBERT.py`

This command will begin the document evaluation process, using the BERT-based model to rank the relevance of documents based on queries.

## Auxiliary Class Information

The **AuxClass.py** file serves as a utility class that supports the main script by handling common tasks such as file reading and metric calculations. This file should not be run directly, as it only provides helper functions for the main scripts.

## Output Information

### MonoBERT.py

Running this file will generate three outputs:

1. A confusion matrix with an overview of performance metrics.
2. A file containing each predicted class and score, along with the actual class and score.
3. A metrics file that contains the detailed evaluation results of the model.
