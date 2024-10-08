import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, fbeta_score
import evaluate
from AuxClass import read_entries_from_file, read_true_labels
import torch
from torch import nn
import numpy as np
import random

# Set seed for reproducibility across different libraries (PyTorch, NumPy, etc.)
seed_variable = 1
torch.manual_seed(seed_variable)  # Seed for PyTorch
np.random.seed(seed_variable)  # Seed for NumPy
random.seed(seed_variable)  # Seed for Python random
torch.cuda.manual_seed(seed_variable)  # Seed for CUDA (GPU)

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

# """DIET (Path setup for the DIET dataset)"""
# Commented out path definitions for alternative datasets
#Diet_path = os.path.join(script_directory, 'Diet')

#train_path = os.path.join(Diet_path, 'Train')

#train_abstracts_path = os.path.join(train_path, 'diet_train_abstracts.txt')
#train_titles_path = os.path.join(train_path, 'diet_train_titles.txt')
#train_class_path = os.path.join(script_directory,'diet_train_class.txt')
#train_all_path = os.path.join(script_directory, 'abstracts+titles.txt')
#queries_path = os.path.join(script_directory,'Diet_queries.txt')
#results_path = os.path.join(script_directory,'Training_results')

# Paths for training data
train_class_path = os.path.join(script_directory, 'diet_train_class.txt')
train_all_path = os.path.join(script_directory, 'abstracts+titles.txt')

# Commented out paths for graphics (optional)
#graphics_path = os.path.join(train_path, 'graphics.txt')

""" Paths for test data (alternative dataset) """
#test_path = os.path.join(Diet_path, 'Test')

#test_abstracts_path = os.path.join(test_path, 'diet_test_abstracts.txt')
#test_titles_path = os.path.join(test_path, 'diet_test_titles.txt')

#test_all_path = os.path.join(test_path, 'abstracts+titles.txt')
#graphics1_path = os.path.join(test_path, 'graphics.txt')
#test_class_path = os.path.join(test_path,'diet_test_class.txt')
#queries_path = os.path.join(Diet_path,'Diet_queries.txt')
#results_diet_train_path = os.path.join(train_path,'results.txt')
#results_diet_test_path = os.path.join(test_path,'results.txt')

# Initialize model with a pre-trained BiomedBERT for classification
model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Load tokenizer
config = AutoConfig.from_pretrained(model_name)  # Load model config
config.hidden_dropout_prob = 0.3  # Set dropout probability in hidden layers
config.attention_probs_dropout_prob = 0.3  # Set dropout probability in attention layers
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)  # Load model with custom config

# Update the classifier layer for binary classification
num_labels = 2  
model.classifier = nn.Linear(config.hidden_size, num_labels)

# Move model to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # Move model to the selected device

# Load training data
article_list = read_entries_from_file(train_all_path)  # Load articles (texts)
article_list = np.array(article_list)  # Convert to numpy array
print(len(article_list))  # Print number of articles
true_labels = read_true_labels(train_class_path, 0, len(article_list))  # Load true labels
print(len(true_labels))  # Print number of labels
true_labels = np.array(true_labels)  # Convert labels to numpy array

# Custom dataset class for handling tokenized inputs and labels
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]  # Get the text by index
        try:
            label = self.labels[idx]  # Get the label by index
        except IndexError as e:
            print(f"IndexError occurred! Length of labels: {len(self.labels)}, Index: {idx}")
            raise e
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten().to(device),  # Move input tensors to the selected device
            'attention_mask': encoding['attention_mask'].flatten().to(device),  # Move attention mask to device
            'labels': torch.tensor(label, dtype=torch.long).to(device)  # Move label tensor to device
        }

# Create dataset instance
dataset = CustomDataset(
    texts=article_list,
    labels=true_labels,
    tokenizer=tokenizer,
    max_len=512  # Max token length set to 512
)

# Split the dataset into training (80%) and validation (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

"""
# Optional block for counting labels in the train and validation datasets
def count_ones_in_dataset(subset):
    labels = []
    loader = DataLoader(subset, batch_size=1, shuffle=False)
    for batch in loader:
        labels.append(batch['labels'].item())
    labels = np.array(labels)
    number_of_ones = np.sum(labels == 0)
    return number_of_ones

train_ones = count_ones_in_dataset(train_dataset)
val_ones = count_ones_in_dataset(val_dataset)

print(f"Number of zeros in train_dataset: {train_ones}")
print(f"Number of zeros in val_dataset: {val_ones}")

# Display dataset indices
dataset = train_dataset
name = "Train dataset"

indices = [i for i in dataset.indices]
print(f"{name} indices: {indices}")

dataset = val_dataset
name = "Validation dataset"

indices = [i for i in dataset.indices]
print(f"{name} indices: {indices}")
"""

# Create output directory if it doesn't exist
output_dir = './results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set up training arguments for the Trainer API
training_args = TrainingArguments(
    output_dir=output_dir,  # Directory for saving results
    num_train_epochs=50,  # Number of training epochs
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=16,  # Batch size for evaluation
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # Weight decay for optimization
    logging_dir='./logs',  # Directory for logging
    logging_steps=10,  # Log every 10 steps
    eval_strategy="epoch",  # Evaluate after each epoch
    save_strategy="epoch",  # Save model after each epoch
    learning_rate=1e-5,  # Learning rate
    save_steps=500,  # Save every 500 steps
    eval_steps=100,  # Evaluate every 100 steps
    seed=seed_variable,  # Set seed for reproducibility
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="combined_metric",  # Metric to determine the best model
    greater_is_better=True,  # Whether higher metric value is better
)

# Load evaluation metric
metric = evaluate.load("accuracy", trust_remote_code=True)

"""
# Optional block for computing various metrics including precision, recall, and F1-score
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    f2 = fbeta_score(labels, preds, beta=2, average='binary')
    acc = accuracy_score(labels, preds)
    
    additional_metrics = metric.compute(predictions=preds, references=labels)
    
    combined_metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
    }
    combined_metrics.update(additional_metrics)
    
    return combined_metrics
"""

# Compute metrics function that returns accuracy, precision, recall, F1, F2, and a combined metric
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)  # Get predictions
    labels = p.label_ids  # Get true labels
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')  # Compute precision, recall, F1
    f2 = fbeta_score(labels, preds, beta=2, average='binary')  # Compute F2 score
    acc = accuracy_score(labels, preds)  # Compute accuracy
    
    combined_metric = (precision + recall + f1 + f2) / 4  # Calculate a combined metric (average of precision, recall, F1, F2)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        'combined_metric': combined_metric,
    }

# Initialize the Trainer with the model, arguments, datasets, and metric computation function
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Ensure all model parameters are contiguous in memory before saving
def make_contiguous(model):
    for name, param in model.named_parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()  # Make the parameter data contiguous
        # Ensure gradients are also contiguous
        if param.grad is not None and not param.grad.is_contiguous():
            param.grad.data = param.grad.data.contiguous()  # Make gradient data contiguous

# Make the model's tensors contiguous before saving
make_contiguous(model)

# Start training the model
train_results = trainer.train()

# Evaluate the model after training
eval_results = trainer.evaluate()

"""
# Optional block for saving the training and evaluation results to files
# Save training results
with open(os.path.join(results_path, 'train_results.txt'), 'w') as train_results_file:
    train_results_file.write(str(train_results))

# Save evaluation results
eval_results = trainer.evaluate()
with open(os.path.join(results_path, 'eval_results.txt'), 'w') as eval_results_file:
    eval_results_file.write(str(eval_results))
"""

# Create directory for saving the trained model and tokenizer
save_directory = './trained_model'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Save the trained model and tokenizer to disk
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")  # Output message to confirm save
