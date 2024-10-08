from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from AuxClass import read_entries_from_file, text_to_string, calculate_metrics,read_true_labels
import torch
import numpy as np
import torch.nn.functional as F

# Set the current script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))

# """DIET
# Below are paths related to the DIET dataset, currently commented out

"""Diet_path = os.path.join(script_directory, 'Diet')

train_path = os.path.join(Diet_path, 'Train')

train_abstracts_path = os.path.join(train_path, 'diet_train_abstracts.txt')
train_titles_path = os.path.join(train_path, 'diet_train_titles.txt')
train_class_path = os.path.join(train_path,'diet_train_class.txt')
train_all_path = os.path.join(train_path, 'abstracts+titles.txt')
graphics_path = os.path.join(train_path, 'graphics.txt')

test_path = os.path.join(Diet_path, 'Test')

test_abstracts_path = os.path.join(test_path, 'diet_test_abstracts.txt')
test_titles_path = os.path.join(test_path, 'diet_test_titles.txt')

test_all_path = os.path.join(test_path, 'abstracts+titles.txt')
graphics1_path = os.path.join(test_path, 'graphics.txt')
test_class_path = os.path.join(test_path,'diet_test_class.txt')
queries_path = os.path.join(Diet_path,'Diet_queries.txt')
results_diet_train_path = os.path.join(train_path,'results.txt')
results_diet_test_path = os.path.join(test_path,'results.txt')
#"""

# """REPRO
# Paths related to reproducibility experiments
Repro_Path = os.path.join(script_directory, 'Reproducibility')
all_path = os.path.join(script_directory, 'abstracts+titles.txt')
class_path = os.path.join(script_directory,'reproducibility_class.txt')

queries_path = os.path.join(script_directory,'Repro_queries.txt')
results_path = os.path.join(script_directory,'Print_sem_metricas.txt')
# results_path = os.path.join(Repro_Path,'results.txt')
# graphics_path = os.path.join(Repro_Path, 'graphics.txt')
# """

# Check if CUDA is available for GPU acceleration, otherwise fall back to CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA device selected.")
else:
    device = torch.device('cpu')
    print("CPU device selected.")

# Load pre-trained model and tokenizer (MonoBERT) from Hugging Face
model_name = "castorini/monobert-large-msmarco-finetune-only"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)  # Move model to the selected device

# Read the articles and queries from the specified files
article_list = read_entries_from_file(all_path)
queries = text_to_string(queries_path).split(" ")  # Convert query file content into a list of words

n_queries = len(queries)  # Number of queries
n_articles = len(article_list)  # Number of articles

# Initialize counters for document and query processing
counter = 0
counter1 = 0

document_scores = []  # List to store aggregated document scores

max_chunk_length = 512  # Maximum length of token chunks (for long sequences)

# Iterate through each document
for doc in article_list:
    print(counter1)  # Print the document index for debugging/monitoring progress

    # Tokenize the document
    tokenized_doc = tokenizer(doc, return_tensors="pt", add_special_tokens=True, truncation=True, padding=True)

    # Tokenize each query individually
    tokenized_queries = [tokenizer(query, return_tensors="pt", add_special_tokens=True, truncation=True, padding=True) for query in queries]

    input_ids_doc = tokenized_doc["input_ids"].to(device)  # Move tokenized document IDs to device (GPU/CPU)
    attention_mask_doc = tokenized_doc["attention_mask"].to(device)  # Move document attention mask to device

    total_query_score = 0  # Initialize the total query score

    # Iterate through each tokenized query
    for query_tokenized in tokenized_queries:
        input_ids_query = query_tokenized["input_ids"].to(device)  # Move query input IDs to device
        attention_mask_query = query_tokenized["attention_mask"].to(device)  # Move query attention mask to device

        # Combine document and query input IDs and attention masks
        combined_input_ids = torch.cat([input_ids_query, input_ids_doc], dim=1)
        combined_attention_mask = torch.cat([attention_mask_query, attention_mask_doc], dim=1)

        # Split combined input into chunks if it exceeds max_chunk_length
        num_chunks = -(-combined_input_ids.size(1) // max_chunk_length)  # Calculate the number of chunks
        chunks_input_ids = torch.split(combined_input_ids, max_chunk_length, dim=1)  # Split input into chunks
        chunks_attention_mask = torch.split(combined_attention_mask, max_chunk_length, dim=1)  # Split attention mask into chunks

        chunk_scores = []  # List to store scores for each chunk
        # Process each chunk through the model
        for chunk_idx, (chunk_input_ids, chunk_attention_mask) in enumerate(zip(chunks_input_ids, chunks_attention_mask)):
            with torch.no_grad():  # Disable gradient calculation for inference
                output = model(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask).logits
                probs = F.softmax(output, dim=-1)  # Get softmax probabilities
                query_score = probs[:, 1].item()  # Extract the score for class 1 (relevant class)
                chunk_scores.append(query_score)  # Store the chunk score

        # If the document is split into chunks, increment the counter
        if num_chunks > 1:
            counter += 1

        # Calculate the average score for the query across chunks
        avg_query_score = sum(chunk_scores) / len(chunk_scores)
        total_query_score += avg_query_score  # Add the average query score to the total score
    
    # Calculate the average document score across all queries
    avg_document_score = total_query_score / len(tokenized_queries)
    document_scores.append(avg_document_score)  # Store the document score
    counter1 += 1  # Increment the document counter

# Apply a threshold to classify the document (1 if score >= 0.5, else 0)
threshold = 0.5
predicted_classes = [1 if score >= threshold else 0 for score in document_scores]

# Read the true labels from the file
true_labels = read_true_labels(class_path, 0, len(article_list))
true_labels = np.array(true_labels)  # Convert labels to a NumPy array

# Write the predicted classes and scores to the results file
with open(results_path, 'w') as file:
    for idx, pred_class in enumerate(predicted_classes):
        output_string = f"Document {idx}: Predicted Class {pred_class} (Aggregated Score: {document_scores[idx]:.4f})\n"
        file.write(output_string)


# Calculate the total number of documents split into multiple chunks
total_n = counter / n_queries
print(total_n)

# Calculate evaluation metrics: Precision, Recall, F1, F2, ROC AUC
precision, recall, f1, f2, roc_auc, predicted_classes = calculate_metrics(document_scores, true_labels)

# Print the evaluation metrics
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"F2-score: {f2:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("Predicted Classes:", predicted_classes)
