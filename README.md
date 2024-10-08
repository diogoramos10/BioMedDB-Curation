# BM25 Model Branch

This branch is related to the **BM25** stage of the dissertation. It contains all the necessary components for testing and running the BM25 model.

## Dataset

The dataset required for this stage is present in this repository, enabling full testing of the BM25 model without needing to retrieve external data.

## Running the Code

The only runnable file in this branch is:

BM25.py

To run the BM25 model, use the following command:

python BM25.py

### Required Packages

Make sure the following Python packages are installed before running the script:

- rank_bm25: This package is used for the BM25 ranking implementation.
- tabulate: This package is used to format output data into tables.

You can install these packages using pip:

- pip install rank_bm25 tabulate

### Auxiliary Files

All other files in this branch are auxiliary classes that support the main BM25.py file and should not be run independently.
