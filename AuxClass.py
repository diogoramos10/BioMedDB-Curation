from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, roc_auc_score
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


script_directory = os.path.dirname(os.path.abspath(__file__))

# Function to read entries from a file and return them as a list
def read_entries_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        entries = file.read().split('\n')
    return entries

# Function to read text from a file and process it by replacing ', ' with a space
def text_to_string(filename):
    with open(filename, 'r', encoding='utf-8-sig') as file:
        text = file.read()
    text = text.replace(', ', ' ')
    return text

# Function to compute Discounted Cumulative Gain (DCG) at rank k
def dcg_at_k(scores, k):
    scores = np.asfarray(scores)[:k]
    if scores.size:
        return np.sum(scores / np.log2(np.arange(2, scores.size + 2)))
    return 0.0

# Function to compute Normalized Discounted Cumulative Gain (NDCG) at rank k
def ndcg_at_k(true_labels, scores, k):
    order = np.argsort(scores)[::-1]  # Sort scores in descending order
    true_labels = np.take(true_labels, order[:k])  # Take top k labels
    best_labels = np.sort(true_labels)[::-1]  # Best possible order of labels
    dcg = dcg_at_k(true_labels, k)
    idcg = dcg_at_k(best_labels, k)
    return dcg / idcg if idcg > 0.0 else 0.0

# Function to calculate various metrics based on document scores and true labels
def calculate_metrics(document_scores, true_labels):
    threshold = 0.5
    # Convert scores to binary predictions based on the threshold
    predicted_classes = [1 if score >= threshold else 0 for score in document_scores]

    # Calculate precision, recall, F1-score, F2-score, ROC AUC, and NDCG@10
    precision = precision_score(true_labels, predicted_classes)
    recall = recall_score(true_labels, predicted_classes)
    f1 = f1_score(true_labels, predicted_classes)
    f2 = fbeta_score(true_labels, predicted_classes, beta=2)
    roc_auc = roc_auc_score(true_labels, document_scores)
    ndcg10 = ndcg_at_k(true_labels, document_scores, 10)

    return precision, recall, f1, f2, roc_auc, ndcg10, predicted_classes

# Function to get a numeric value from a specific line in a file
def get_value_from_line(file_path, line_number):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if 1 <= line_number <= len(lines):
                try:
                    num = int(lines[line_number - 1].strip())  # Convert line content to an integer
                    if num in (0, 1):
                        return num
                    else:
                        print(f"Warning: Line {line_number} contains an invalid number. Only 0 or 1 allowed.")
                except ValueError:
                    print(f"Warning: Line {line_number} is non-numeric.")
            else:
                print(f"Error: Line number {line_number} is out of range.")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")

    return None

# Function to read true labels from a specific range of lines in a file
def read_true_labels(file_path, start_line, end_line):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        true_labels = [int(line.strip()) for line in lines[start_line:end_line]]
    return true_labels

# Function to count the number of tokens (or elements) in a list
def count_tokens(input_ids):
    return len(input_ids)

# Function to calculate the average and standard deviation of various metrics from multiple files
def calculate_metrics_average_and_std(filenames):
    metrics = {
        'Precision': [],
        'Recall': [],
        'F1-score': [],
        'F2-score': [],
        'ROC AUC': [],
        'NDCG@10': []
    }

    pattern = re.compile(r'(.+?):\s*([0-9.]+)')  # Regular expression to extract metric name and value

    for filename in filenames:
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            continue

        with open(filename, 'r') as file:
            for line in file:
                match = pattern.match(line.strip())
                if match:
                    metric_name = match.group(1)
                    value = float(match.group(2))
                    if metric_name in metrics:
                        metrics[metric_name].append(value)

    # Calculate average and standard deviation for each metric
    metrics_avg_std = {}
    for metric, values in metrics.items():
        if values:
            avg = np.mean(values)
            std = np.std(values)
            metrics_avg_std[metric] = {'average': avg, 'std_dev': std}
        else:
            metrics_avg_std[metric] = {'average': None, 'std_dev': None}

    return metrics_avg_std

# Function to generate and save a table of metric averages and standard deviations as an image
def generate_metrics_table(metrics_avg_std, save_path='metrics_table.png'):
    col_labels = list(metrics_avg_std.keys())
    rows = ['Average']
    columns = []

    for metric in col_labels:
        stats = metrics_avg_std[metric]
        if stats['average'] is not None and stats['std_dev'] is not None:
            combined = f"{round(stats['average'], 4)} ± {round(stats['std_dev'], 4)}"
        else:
            combined = 'N/A'
        columns.append([combined])

    columns = list(map(list, zip(*columns)))

    fig, ax = plt.subplots(figsize=(12, 6))  # Increased figure size for better readability
    ax.axis('tight')
    ax.axis('off')

    # Create a table with the calculated metric averages and standard deviations
    table = ax.table(cellText=columns, rowLabels=rows, colLabels=col_labels, cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.5, 1.5)

    for key, cell in table.get_celld().items():
        cell.set_height(0.15)  # Adjust cell height
        cell.set_fontsize(14)
        cell.PAD = 0.2  # Adjust padding

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Table saved as {save_path}")
    plt.close()

filenames = []

items = calculate_metrics_average_and_std(filenames)

generate_metrics_table(items)

# Function to process a log file and plot training and evaluation losses, as well as evaluation metrics
def process_and_plot_file(file_name):
    training_losses = []
    eval_losses = []
    eval_precisions = []
    eval_recalls = []
    eval_f1s = []
    eval_f2s = []

    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("{'loss':"):
                data = eval(line.strip())
                training_losses.append((data['epoch'], data['loss']))
            elif line.startswith("{'eval_loss':"):
                data = eval(line.strip())
                eval_losses.append((data['epoch'], data['eval_loss']))
                eval_precisions.append((data['epoch'], data['eval_precision']))
                eval_recalls.append((data['epoch'], data['eval_recall']))
                eval_f1s.append((data['epoch'], data['eval_f1']))
                eval_f2s.append((data['epoch'], data['eval_f2']))

    # Sort data by epoch for proper plotting
    training_losses.sort(key=lambda x: x[0])
    eval_losses.sort(key=lambda x: x[0])
    eval_precisions.sort(key=lambda x: x[0])
    eval_recalls.sort(key=lambda x: x[0])
    eval_f1s.sort(key=lambda x: x[0])
    eval_f2s.sort(key=lambda x: x[0])

    # Unpack epochs and values
    epochs_train, losses_train = zip(*training_losses) if training_losses else ([], [])
    epochs_eval, losses_eval = zip(*eval_losses) if eval_losses else ([], [])
    _, precisions = zip(*eval_precisions) if eval_precisions else ([], [])
    _, recalls = zip(*eval_recalls) if eval_recalls else ([], [])
    _, f1s = zip(*eval_f1s) if eval_f1s else ([], [])
    _, f2s = zip(*eval_f2s) if eval_f2s else ([], [])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Plot training and evaluation losses
    if epochs_train and losses_train:
        ax1.plot(epochs_train, losses_train, label='Training Loss', marker='o', linewidth=2)
    if epochs_eval and losses_eval:
        ax1.plot(epochs_eval, losses_eval, label='Evaluation Loss', marker='o', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.set_title(f'Training and Evaluation Loss for {os.path.basename(file_name)}', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True)

    # Plot evaluation metrics
    if epochs_eval and precisions:
        ax2.plot(epochs_eval, precisions, label='Precision', marker='o', linewidth=2)
    if epochs_eval and recalls:
        ax2.plot(epochs_eval, recalls, label='Recall', marker='o', linewidth=2)
    if epochs_eval and f1s:
        ax2.plot(epochs_eval, f1s, label='F1 Score', marker='o', linewidth=2)
    if epochs_eval and f2s:
        ax2.plot(epochs_eval, f2s, label='F2 Score', marker='o', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Metric', fontsize=14)
    ax2.set_title(f'Evaluation Metrics for {os.path.basename(file_name)}', fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"{os.path.splitext(file_name)[0]}_combined.png", dpi=300)
    plt.close()

    print(f"Combined plot saved for {file_name}")

# file_names = []

# for file_name in file_names:
#     process_and_plot_file(file_name)

# Reproducibility data paths
Repro_Path = os.path.join(script_directory, 'Reproducibility')

Train_Path = os.path.join(Repro_Path, 'Train')
Test_Path = os.path.join(Repro_Path, 'Test')

all_path = os.path.join(Repro_Path, 'abstracts+titles.txt')
class_path = os.path.join(Repro_Path, 'reproducibility_class.txt')

# Function to split dataset into training and testing sets
def split_dataset(script_directory, train_rel_ratio=0.8):
    # Load data paths
    abstracts_titles_path = os.path.join(Repro_Path, 'abstracts+titles.txt')
    repro_class_path = os.path.join(Repro_Path, 'reproducibility_class.txt')

    # Read files with the correct encoding
    with open(abstracts_titles_path, 'r', encoding='utf-8') as f:
        abstracts_titles = f.readlines()
    
    with open(repro_class_path, 'r', encoding='utf-8') as f:
        repro_class = [int(line.strip()) for line in f.readlines()]
    
    # Create a DataFrame for easier manipulation
    data = pd.DataFrame({
        'abstracts_titles': abstracts_titles,
        'repro_class': repro_class
    })

    # Separate relevant and non-relevant documents
    relevant = data[data['repro_class'] == 1]
    non_relevant = data[data['repro_class'] == 0]

    # Print total number of relevant documents
    num_relevant = len(relevant)
    print(f"Total number of relevant documents: {num_relevant}")

    # Determine the number of relevant documents for training and testing sets
    num_rel_train = max(5, int(train_rel_ratio * num_relevant))
    num_rel_test = num_relevant - num_rel_train

    # Check if we have enough relevant documents
    if num_rel_train + num_rel_test > num_relevant:
        # Calculate the maximum feasible number of relevant documents for each set
        total_docs = num_relevant // 2
        num_rel_train = num_rel_test = total_docs

    # Split relevant documents
    rel_train, rel_test = train_test_split(relevant, test_size=num_rel_test, random_state=42)

    # Split non-relevant documents, keeping the rest of the ratio
    non_rel_train, non_rel_test = train_test_split(non_relevant, test_size=0.3, random_state=42)
    
    # Combine the relevant and non-relevant documents for the final training and testing sets
    train_data = pd.concat([non_rel_train, rel_train]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_data = pd.concat([non_rel_test, rel_test]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the splits
    train_dir = os.path.join(Repro_Path, 'Train')
    test_dir = os.path.join(Repro_Path, 'Test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    train_abstracts_titles_path = os.path.join(train_dir, 'abstracts+titles2.txt')
    train_repro_class_path = os.path.join(train_dir, 'Repro_class_train.txt')
    test_abstracts_titles_path = os.path.join(test_dir, 'abstracts+titles3.txt')
    test_repro_class_path = os.path.join(test_dir, 'Repro_class_test.txt')
    
    with open(train_abstracts_titles_path, 'w', encoding='utf-8') as f:
        f.writelines(train_data['abstracts_titles'])
    
    with open(train_repro_class_path, 'w', encoding='utf-8') as f:
        f.writelines([str(line) + '\n' for line in train_data['repro_class']])
    
    with open(test_abstracts_titles_path, 'w', encoding='utf-8') as f:
        f.writelines(test_data['abstracts_titles'])
    
    with open(test_repro_class_path, 'w', encoding='utf-8') as f:
        f.writelines([str(line) + '\n' for line in test_data['repro_class']])
    
    # Print statistics
    print("Training Set:")
    print(f"Total documents: {len(train_data)}")
    print(f"Relevant documents: {train_data['repro_class'].sum()}")
    print(f"Non-relevant documents: {len(train_data) - train_data['repro_class'].sum()}")
    
    print("\nTesting Set:")
    print(f"Total documents: {len(test_data)}")
    print(f"Relevant documents: {test_data['repro_class'].sum()}")
    print(f"Non-relevant documents: {len(test_data) - test_data['repro_class'].sum()}")

    print("Data split and saved successfully")


# split_dataset(script_directory)

# Function to print dataset statistics for training and testing sets
def print_dataset_statistics(script_directory):
    # Paths to the split datasets
    train_dir = os.path.join(Repro_Path, 'Train')
    test_dir = os.path.join(Repro_Path, 'Test')
    
    train_abstracts_titles_path = os.path.join(train_dir, 'abstracts+titles2.txt')
    train_repro_class_path = os.path.join(train_dir, 'Repro_class_train.txt')
    test_abstracts_titles_path = os.path.join(test_dir, 'abstracts+titles3.txt')
    test_repro_class_path = os.path.join(test_dir, 'Repro_class_test.txt')
    
    # Load data
    with open(train_abstracts_titles_path, 'r', encoding='utf-8') as f:
        train_abstracts_titles = f.readlines()
    
    with open(train_repro_class_path, 'r', encoding='utf-8') as f:
        train_repro_class = [int(line.strip()) for line in f.readlines()]
    
    with open(test_abstracts_titles_path, 'r', encoding='utf-8') as f:
        test_abstracts_titles = f.readlines()
    
    with open(test_repro_class_path, 'r', encoding='utf-8') as f:
        test_repro_class = [int(line.strip()) for line in f.readlines()]
    
    # Create DataFrames for easier manipulation
    train_data = pd.DataFrame({
        'abstracts_titles': train_abstracts_titles,
        'repro_class': train_repro_class
    })
    
    test_data = pd.DataFrame({
        'abstracts_titles': test_abstracts_titles,
        'repro_class': test_repro_class
    })
    
    # Print statistics
    print("Training Set:")
    print(f"Total documents: {len(train_data)}")
    print(f"Relevant documents: {train_data['repro_class'].sum()}")
    print(f"Non-relevant documents: {len(train_data) - train_data['repro_class'].sum()}")
    
    print("\nTesting Set:")
    print(f"Total documents: {len(test_data)}")
    print(f"Relevant documents: {test_data['repro_class'].sum()}")
    print(f"Non-relevant documents: {len(test_data) - test_data['repro_class'].sum()}")

# print_dataset_statistics(script_directory)

# Update Repro_Path to point to the 'Pollutants' directory
Repro_Path = os.path.join(script_directory, 'Pollutants')

Dpbs = os.path.join(Repro_Path, 'POLYCLORINATED')

# Function to print total number of relevant documents in a dataset
def statistics(script_directory):
    # Load data
    abstracts_titles_path = os.path.join(Dpbs, 'abstracts+titles.txt')
    repro_class_path = os.path.join(Dpbs, 'polychlorinated_class.txt')

    # Read files with the correct encoding
    with open(abstracts_titles_path, 'r', encoding='utf-8') as f:
        abstracts_titles = f.readlines()
    
    with open(repro_class_path, 'r', encoding='utf-8') as f:
        repro_class = [int(line.strip()) for line in f.readlines()]
    
    # Create a DataFrame for easier manipulation
    data = pd.DataFrame({
        'abstracts_titles': abstracts_titles,
        'repro_class': repro_class
    })

    # Separate relevant and non-relevant documents
    relevant = data[data['repro_class'] == 1]
    non_relevant = data[data['repro_class'] == 0]

    # Print total number of relevant documents
    num_relevant = len(relevant)
    print(f"Total number of relevant documents: {num_relevant}")

# Call the statistics function
statistics(Repro_Path)
