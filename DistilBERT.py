from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from AuxClass import read_entries_from_file, text_to_string, calculate_metrics, get_value_from_line, read_true_labels
import torch
import numpy as np
import torch.nn.functional as F

# Set the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

# """DIET
# This section contains paths related to the DIET dataset (commented out for now)

"""Diet_path = os.path.join(script_directory, 'Diet')

train_path = os.path.join(Diet_path, 'Train')

train_abstracts_path = os.path.join(train_path, 'diet_train_abstracts.txt')
train_titles_path = os.path.join(train_path, 'diet_train_titles.txt')
train_class_path = os.path.join(train_path,'diet_train_class.txt')
train_all_path = os.path.join(train_path, 'abstracts+titles.txt')
graphics_path = os.path.join(train_path, 'graphics.txt')

test_path = os.path.join(Diet_path, 'Test')
"""
# Paths related to the test set
# test_abstracts_path = os.path.join(script_directory, 'diet_test_abstracts.txt')
# test_titles_path = os.path.join(script_directory, 'diet_test_titles.txt')

# Path to the test data (abstracts + titles combined)
test_all_path = os.path.join(script_directory, 'abstracts+titles1.txt')
# graphics1_path = os.path.join(script_directory, 'graphics.txt')

# Path to the test class labels
test_class_path = os.path.join(script_directory, 'diet_test_class.txt')

# Path to the queries
queries_path = os.path.join(script_directory, 'Diet_queries.txt')

# results_diet_train_path = os.path.join(script_directory,'results.txt')
# results_diet_test_path = os.path.join(script_directory,'results.txt')

# """

"""REPRO
# This section contains paths related to reproducibility (commented out for now)

Repro_Path = os.path.join(script_directory, 'Reproducibility')
all_path = os.path.join(Repro_Path, 'abstracts+titles.txt')
class_path = os.path.join(Repro_Path,'reproducibility_class.txt')

queries_path = os.path.join(Repro_Path,'Repro_queries.txt')
# results_path = os.path.join(Repro_Path,'results.txt')
# graphics_path = os.path.join(Repro_Path, 'graphics.txt')

# """

# Set the device (GPU if available, otherwise CPU)
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA device selected.")
else:
    device = torch.device('cpu')
    print("CPU device selected.")

# Path to the trained model
model_path = 'trained_model'

# Load the tokenizer and model from the specified model path
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)

# Read the test articles and queries
article_list = read_entries_from_file(test_all_path)
queries = text_to_string(queries_path).split(" ")

# Number of queries and articles
n_queries = len(queries)
n_articles = len(article_list)

# Initialize counters and document scores list
# counter = 0
counter1 = 0
document_scores = []

# Maximum chunk length for tokenization (DistilBERT has a max input length of 512 tokens)
max_chunk_length = 512

# Iterate over the articles in the test set
for idx, doc in enumerate(article_list):
    print(idx)  # Print the index for debugging purposes
    
    # Tokenize the document and send it to the device (GPU/CPU)
    tokenized_doc = tokenizer(doc, return_tensors="pt", add_special_tokens=True, truncation=True, padding=True)
    input_ids_doc = tokenized_doc["input_ids"].to(device)
    attention_mask_doc = tokenized_doc["attention_mask"].to(device)

    # Split the input into chunks if it exceeds the maximum length (512 tokens)
    num_chunks = -(-input_ids_doc.size(1) // max_chunk_length)  # Calculate the number of chunks
    chunks_input_ids = torch.split(input_ids_doc, max_chunk_length, dim=1)
    chunks_attention_mask = torch.split(attention_mask_doc, max_chunk_length, dim=1)

    # Initialize a list to store scores for each chunk
    chunk_scores = []
    
    # Process each chunk
    for chunk_idx, (chunk_input_ids, chunk_attention_mask) in enumerate(zip(chunks_input_ids, chunks_attention_mask)):
        with torch.no_grad():  # Disable gradient calculations to save memory
            # Get the model's output logits and calculate probabilities
            output = model(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask).logits
            probs = F.softmax(output, dim=-1)  # Apply softmax to get class probabilities
            doc_score = probs[:, 1].item()  # Get the score for class 1 (positive class)
            chunk_scores.append(doc_score)  # Append the score for this chunk

    # Take the average score from all chunks (in this case, using the first chunk score)
    avg_doc_score = chunk_scores[0]
    document_scores.append(avg_doc_score)  # Append the aggregated score for the document

# Set the threshold for binary classification
threshold = 0.5
# Predict classes based on the threshold (1 if score >= 0.5, else 0)
predicted_classes = [1 if score >= threshold else 0 for score in document_scores]

# Read true labels from the test class file
true_labels = read_true_labels(test_class_path, 0, len(article_list))
true_labels = np.array(true_labels)  # Convert the true labels to a NumPy array

# Write the predicted and true classes to a file
with open("Print_sem_metricas.txt", 'w') as file:
   for idx, (pred_class, true_class) in enumerate(zip(predicted_classes, true_labels)):
        output_string = f"Document {idx}: Predicted Class {pred_class} (Aggregated Score: {document_scores[idx]:.4f}), Real Class: {true_class}\n"
        file.write(output_string)

# Calculate precision, recall, F1, F2, and ROC AUC using the predicted and true labels
precision, recall, f1, f2, roc_auc, predicted_classes = calculate_metrics(document_scores, true_labels)
# Print the calculated metrics
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"F2-score: {f2:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("Predicted Classes:", predicted_classes)

# Write the calculated metrics to a file
with open("Print_metricas.txt", 'w') as file:
    output_string = (
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1-score: {f1:.4f}\n"
        f"F2-score: {f2:.4f}\n"
        f"ROC AUC: {roc_auc:.4f}\n"
    )
    file.write(output_string)

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_classes)

# Plot the confusion matrix using seaborn's heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)  # Plot confusion matrix
plt.title('Confusion Matrix')  # Title of the plot
plt.xlabel('Predicted Label')  # X-axis label
plt.ylabel('True Label')  # Y-axis label

# Add the calculated metrics to the confusion matrix plot
plt.text(2, 0, f'Precision: {precision:.4f}', fontsize=12)
plt.text(2, 1, f'Recall: {recall:.4f}', fontsize=12)
plt.text(2, 2, f'F1-score: {f1:.4f}', fontsize=12)
plt.text(2, 3, f'F2-score: {f2:.4f}', fontsize=12)
plt.text(2, 4, f'ROC AUC: {roc_auc:.4f}', fontsize=12)

# Save the confusion matrix with metrics as a PNG image
plt.savefig('confusion_matrix_with_metrics.png', bbox_inches='tight')
