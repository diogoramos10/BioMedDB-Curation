FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /home/dl.ramos

# Install any needed dependencies specified in requirements.txt
RUN pip install torch transformers scikit-learn evaluate accelerate matplotlib seaborn 