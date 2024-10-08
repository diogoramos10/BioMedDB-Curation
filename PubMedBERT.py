import os
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from AuxClass import read_entries_from_file, text_to_string, calculate_metrics, read_true_labels

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Paths for test data (commented block for alternate dataset paths)
#"""DIET

"""Diet_path = os.path.join(script_directory, 'Diet')

train_path = os.path.join(Diet_path, 'Train')

train_abstracts_path = os.path.join(train_path, 'diet_train_abstracts.txt')
train_titles_path = os.path.join(train_path, 'diet_train_titles.txt')
train_class_path = os.path.join(train_path,'diet_train_class.txt')
train_all_path = os.path.join(train_path, 'abstracts+titles.txt')
graphics_path = os.path.join(train_path, 'graphics.txt')

test_path = os.path.join(Diet_path, 'Test')"""

# Alternate test data paths (abstracts, titles)
#test_abstracts_path = os.path.join(script_directory, 'diet_test_abstracts.txt')
#test_titles_path = os.path.join(script_directory, 'diet_test_titles.txt')

# Another test data path including both abstracts and titles
#test_all_path = os.path.join(script_directory, 'abstracts+titles1.txt')
#graphics1_path = os.path.join(script_directory, 'graphics.txt')
#test_class_path = os.path.join(script_directory,'diet_test_class.txt')
#queries_path = os.path.join(script_directory,'Diet_queries.txt')
#results_diet_train_path = os.path.join(script_directory,'results.txt')
#results_diet_test_path = os.path.join(script_directory,'results.txt')

# Current test data paths
test_all_path = os.path.join(script_directory, 'abstracts+titles1.txt')
test_class_path = os.path.join(script_directory,'diet_test_class.txt')
queries_path = os.path.join(script_directory,'Diet_queries.txt')

#"""

"""REPRO

Repro_Path = os.path.join(script_directory, 'Reproducibility')
all_path = os.path.join(Repro_Path, 'abstracts+titles.txt')
class_path = os.path.join(Repro_Path,'reproducibility_class.txt')

queries_path = os.path.join(Repro_Path,'Repro_queries.txt')
#results_path = os.path.join(Repro_Path,'results.txt')
#graphics_path = os.path.join(Repro_Path, 'graphics.txt')

#"""

# Check if a CUDA device is available; if not, default to CPU
if torch.cuda.is_available():
    device = torch.device('cuda')  
    print("CUDA device selected.")
else:
    device = torch.device('cpu')  
    print("CPU device selected.")

# Load the pre-trained model and tokenizer from a local directory 
model_path = 'trained_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)  # Load the tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)  # Load the model and move it to the selected device 

# Read test documents 
article_list = read_entries_from_file(test_all_path)
n_articles = len(article_list)  # Get the number of articles in the list

document_scores = []  # Initialize an empty list to store document scores
max_chunk_length = 512  # Set the maximum token length per chunk 

# Iterate over the list of articles
for idx, doc in enumerate(article_list):
    print(idx)  # Print current article index
    tokenized_doc = tokenizer(doc, return_tensors="pt", add_special_tokens=True, truncation=True, padding=True)
    input_ids_doc = tokenized_doc["input_ids"].to(device)  # Move input IDs to the selected device 
    attention_mask_doc = tokenized_doc["attention_mask"].to(device)  # Move attention masks to the selected device

    # Split long documents into chunks that fit the model's input limit
    num_chunks = -(-input_ids_doc.size(1) // max_chunk_length)  # Calculate the number of chunks needed
    chunks_input_ids = torch.split(input_ids_doc, max_chunk_length, dim=1)  # Split input IDs into chunks
    chunks_attention_mask = torch.split(attention_mask_doc, max_chunk_length, dim=1)  # Split attention masks

    chunk_scores = []  # List to hold the scores of individual chunks
    for chunk_idx, (chunk_input_ids, chunk_attention_mask) in enumerate(zip(chunks_input_ids, chunks_attention_mask)):
        with torch.no_grad():  # No gradient calculation is needed for inference
            output = model(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask).logits  # Get model logits
            probs = F.softmax(output, dim=-1)  # Apply softmax to get probability distribution
            doc_score = probs[:, 1].item()  # Use the probability of class 1 (positive class)
            chunk_scores.append(doc_score)  # Append chunk score to list

    avg_doc_score = chunk_scores[0]  # Picks the first chunk score 
    document_scores.append(avg_doc_score)  # Store the average document score

# Set a classification threshold
threshold = 0.5
predicted_classes = [1 if score >= threshold else 0 for score in document_scores]  # Classify based on the threshold

# Read true labels for the test set
true_labels = read_true_labels(test_class_path, 0, len(article_list))
true_labels = np.array(true_labels)  # Convert the list of true labels into a numpy array

# Save results to a text file without metrics
with open("Print_sem_metricas.txt", 'w') as file:
    for idx, (pred_class, true_class) in enumerate(zip(predicted_classes, true_labels)):
        output_string = f"Document {idx}: Predicted Class {pred_class} (Aggregated Score: {document_scores[idx]:.4f}), Real Class: {true_class}\n"
        file.write(output_string)  # Write predicted and true classes to the file

# Calculate precision, recall, F1-score, F2-score, ROC AUC, and NDCG@10 using custom function
precision, recall, f1, f2, roc_auc, ndcg10, predicted_classes = calculate_metrics(document_scores, true_labels)

# Print the calculated metrics
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"F2-score: {f2:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"NDCG@10: {ndcg10:.4f}")
print("Predicted Classes:", predicted_classes)

# Save the metrics to a text file
with open("Print_metricas.txt", 'w') as file:
    output_string = (
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1-score: {f1:.4f}\n"
        f"F2-score: {f2:.4f}\n"
        f"ROC AUC: {roc_auc:.4f}\n"
        f"NDCG@10: {ndcg10:.4f}"
    )
    file.write(output_string)  # Write the metrics to a file

# Create a confusion matrix based on true labels and predicted labels
conf_matrix = confusion_matrix(true_labels, predicted_classes)

# Plot the confusion matrix 
plt.figure(figsize=(10, 7))  # Set the figure size
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)  # Plot confusion matrix with annotations
plt.title('Confusion Matrix')  # Set plot title
plt.xlabel('Predicted Label')  # Set x-axis label
plt.ylabel('True Label')  # Set y-axis label

# Add the calculated metrics as text inside the plot
plt.text(2, 0, f'Precision: {precision:.4f}', fontsize=12)
plt.text(2, 1, f'Recall: {recall:.4f}', fontsize=12)
plt.text(2, 2, f'F1-score: {f1:.4f}', fontsize=12)
plt.text(2, 3, f'F2-score: {f2:.4f}', fontsize=12)
plt.text(2, 4, f'ROC AUC: {roc_auc:.4f}', fontsize=12)
plt.text(2, 5, f"NDCG@10: {ndcg10:.4f}", fontsize=12)

# Save the plot to a file
plt.savefig('confusion_matrix_with_metrics.png', bbox_inches='tight')
