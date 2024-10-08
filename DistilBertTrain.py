import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import evaluate
from AuxClass import read_entries_from_file, read_true_labels
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, fbeta_score
import torch
import numpy as np
import random

# Set script directory and seed for reproducibility across different modules
script_directory = os.path.dirname(os.path.abspath(__file__))
seed_variable = 2
torch.manual_seed(seed_variable)  # Set seed for PyTorch operations
np.random.seed(seed_variable)  # Set seed for NumPy operations
random.seed(seed_variable)  # Set seed for Python's random operations
torch.cuda.manual_seed(seed_variable)  # Set seed for CUDA operations 
# """DIET
# Below are paths related to the DIET dataset, which are commented out for now.

# Diet_path = os.path.join(script_directory, 'Diet')  # Path to the Diet dataset folder

# train_path = os.path.join(Diet_path, 'Train')  # Path to the Train folder

# Path for training abstracts
# train_abstracts_path = os.path.join(train_path, 'diet_train_abstracts.txt')

# Path for training titles
# train_titles_path = os.path.join(train_path, 'diet_train_titles.txt')

# Path to the training classes (this path is active)
train_class_path = os.path.join(script_directory, 'diet_train_class.txt')

# Path to the abstracts and titles combined (this path is active)
train_all_path = os.path.join(script_directory, 'abstracts+titles.txt')

# Path to the queries file (this path is active)
queries_path = os.path.join(script_directory, 'Diet_queries.txt')

# Path for storing training results (commented out)
# results_path = os.path.join(script_directory, 'Training_results')

# Path to save graphical results (commented out)
# graphics_path = os.path.join(train_path, 'graphics.txt')

"""test_path = os.path.join(Diet_path, 'Test')
These are paths related to the test set, commented out for now.

# Path for testing abstracts
test_abstracts_path = os.path.join(test_path, 'diet_test_abstracts.txt')

# Path for testing titles
test_titles_path = os.path.join(test_path, 'diet_test_titles.txt')

# Path for combined test abstracts and titles
test_all_path = os.path.join(test_path, 'abstracts+titles.txt')

# Path for graphical results from test set
graphics1_path = os.path.join(test_path, 'graphics.txt')

# Path for test classes
test_class_path = os.path.join(test_path, 'diet_test_class.txt')

# Path for queries file for the test set
queries_path = os.path.join(Diet_path, 'Diet_queries.txt')

# Paths for storing results from the train and test sets
results_diet_train_path = os.path.join(train_path, 'results.txt')
results_diet_test_path = os.path.join(test_path, 'results.txt')
"""

# """

# Load the pre-trained DistilBERT model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)  # Load tokenizer from pre-trained DistilBERT
model = DistilBertForSequenceClassification.from_pretrained(model_name)  # Load classification model

# Use GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # Move model to the specified device 

# Read training data (abstracts + titles) and convert it to a NumPy array
article_list = read_entries_from_file(train_all_path)
article_list = np.array(article_list)  # Convert list to NumPy array
print(len(article_list))  # Print the number of articles (for debugging purposes)

# Read true labels for the articles and convert them to a NumPy array
true_labels = read_true_labels(train_class_path, 0, len(article_list))
print(len(true_labels))  # Print the number of true labels (for debugging purposes)
true_labels = np.array(true_labels)  # Convert list of labels to NumPy array

# Define a custom dataset class to handle tokenization and encoding of texts and labels
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts  # List of texts (abstracts + titles)
        self.labels = labels  # Corresponding true labels for the texts
        self.tokenizer = tokenizer  # Pre-trained tokenizer (DistilBERT)
        self.max_len = max_len  # Maximum length for tokenized text

    def __len__(self):
        return len(self.texts)  # Return the number of texts

    def __getitem__(self, idx):
        text = self.texts[idx]  # Get the text at the specified index
        try:
            label = self.labels[idx]  # Get the label at the specified index
        except IndexError as e:
            print(f"IndexError occurred! Length of labels: {len(self.labels)}, Index: {idx}")  # Handle IndexError
            raise e
        # Tokenize and encode the text using the specified tokenizer
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,  # Add special tokens like [CLS] and [SEP]
            max_length=self.max_len,  # Truncate or pad to the maximum length
            return_token_type_ids=False,  # Token type IDs are not needed for DistilBERT
            padding='max_length',  # Pad to the maximum length
            truncation=True,  # Truncate if the text exceeds max_len
            return_attention_mask=True,  # Return attention mask
            return_tensors='pt',  # Return PyTorch tensors
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten().to(device),  # Flatten and move input IDs to device (GPU/CPU)
            'attention_mask': encoding['attention_mask'].flatten().to(device),  # Flatten and move attention mask
            'labels': torch.tensor(label, dtype=torch.long).to(device)  # Convert label to tensor and move to device
        }

# Create a dataset object from the article list and true labels, with tokenization
dataset = CustomDataset(
    texts=article_list,
    labels=true_labels,
    tokenizer=tokenizer,
    max_len=512  # Set maximum length for tokenized sequences
)

# Split the dataset into training and validation sets (80% training, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create a directory to store the results if it doesn't already exist
# Personalizar o nome para cada experiencia
output_dir = './results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create results directory

# Training arguments for the Trainer class, including additional parameters
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=20,  # Number of epochs increased to 20
    per_device_train_batch_size=8,  # Batch size of 8 for training
    per_device_eval_batch_size=8,  # Batch size of 8 for evaluation
    warmup_steps=500,  # Number of warmup steps
    weight_decay=0.01,  # Weight decay for regularization
    logging_dir='./logs',  # Directory to store logs
    logging_steps=10,  # Log every 10 steps
    eval_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",  # Save model at the end of each epoch
    learning_rate=1e-5,  # Learning rate
    save_steps=500,  # Save checkpoint every 500 steps
    eval_steps=100,  # Evaluate every 100 steps
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="combined_metric",  # Use combined metric to determine the best model
    greater_is_better=True,  # A higher value for combined metric is better
    seed=seed_variable,  # Set seed for reproducibility
)

# Load the accuracy metric from the Hugging Face evaluate library
metric = evaluate.load("accuracy", trust_remote_code=True)

# Function to compute metrics for evaluation
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)  # Get predicted class labels
    labels = p.label_ids  # Get true labels
    
    # Calculate precision, recall, f1-score, and f2-score
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    f2 = fbeta_score(labels, preds, beta=2, average='binary')
    acc = accuracy_score(labels, preds)  # Calculate accuracy
    
    # Calculate combined metric as the average of precision, recall, f1, and f2
    combined_metric = (precision + recall + f1 + f2) / 4  
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        'combined_metric': combined_metric,  # Return combined metric
    }

# Initialize the Trainer class for training and evaluation
trainer = Trainer(
    model=model,  # The model to be trained
    args=training_args,  # Training arguments
    train_dataset=train_dataset,  # Training dataset
    eval_dataset=val_dataset,  # Validation dataset
    tokenizer=tokenizer,  # Tokenizer for encoding texts
    compute_metrics=compute_metrics,  # Metrics to compute during evaluation
)

# Train the model
train_results = trainer.train()

# Evaluate the model on the validation set
trainer.evaluate()

"""# Save training results (commented out)
with open(os.path.join(results_path, 'train_results.txt'), 'w') as train_results_file:
    train_results_file.write(str(train_results))

# Save evaluation results (commented out)
eval_results = trainer.evaluate()
with open(os.path.join(results_path, 'eval_results.txt'), 'w') as eval_results_file:
    eval_results_file.write(str(eval_results))
"""

# Save the trained model and tokenizer to a specified directory
save_directory = './trained_model'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)  # Create directory if it doesn't exist
model.save_pretrained(save_directory)  # Save the model
tokenizer.save_pretrained(save_directory)  # Save the tokenizer

# Print confirmation that the model and tokenizer have been saved
print(f"Model and tokenizer saved to {save_directory}")
