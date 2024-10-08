from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, roc_auc_score
import re
import os
import numpy as np
import matplotlib.pyplot as plt

# Function to read entries from a file and return them as a list of strings
def read_entries_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        entries = file.read().split('\n')  # Split file content by new lines
    return entries

# Function to convert the content of a file to a single string
def text_to_string(filename):
    with open(filename, 'r', encoding='utf-8-sig') as file:
        text = file.read()  # Read the entire file content

    text = text.replace(', ', ' ')  # Replace commas followed by spaces with a single space

    return text

# Function to calculate classification metrics (precision, recall, f1-score, f2-score, ROC-AUC)
def calculate_metrics(document_scores, true_labels):
    threshold = 0.5  # Define a threshold for binary classification
    predicted_classes = [1 if score >= threshold else 0 for score in document_scores]  # Predict class based on threshold
    
    # Calculate various classification metrics
    precision = precision_score(true_labels, predicted_classes)
    recall = recall_score(true_labels, predicted_classes)
    f1 = f1_score(true_labels, predicted_classes)
    f2 = fbeta_score(true_labels, predicted_classes, beta=2)
    roc_auc = roc_auc_score(true_labels, document_scores)
    
    # Return the calculated metrics and the predicted classes
    return precision, recall, f1, f2, roc_auc, predicted_classes

# Function to read and validate a value from a specific line number in a file
def get_value_from_line(file_path, line_number):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()  # Read all lines in the file
            if 1 <= line_number <= len(lines):
                try:
                    num = int(lines[line_number - 1].strip())  # Convert the line content to integer
                    if num in (0, 1):
                        return num  # Return the value if it's 0 or 1
                    else:
                        print(f"Warning: Line {line_number} contains an invalid number. Only 0 or 1 allowed.")
                except ValueError:
                    print(f"Warning: Line {line_number} is non-numeric.")  # Handle non-numeric values
            else:
                print(f"Error: Line number {line_number} is out of range.")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")  # Handle file not found error
    
    return None

# Function to read true labels from a file given a range of lines
def read_true_labels(file_path, start_line, end_line):
    with open(file_path, 'r') as file:
        lines = file.readlines()  # Read all lines in the file
        true_labels = [int(line.strip()) for line in lines[start_line:end_line]]  # Convert lines to integers
    return true_labels

# Function to count the number of tokens (IDs) in a list
def count_tokens(input_ids):
    return len(input_ids)  # Return the length of the input list

# Function to calculate average and standard deviation of metrics from multiple files
def calculate_metrics_average_and_std():
    metrics = {
        'Precision': [],  # Store Precision values
        'Recall': [],  # Store Recall values
        'F1-score': [],  # Store F1-score values
        'F2-score': [],  # Store F2-score values
        'ROC AUC': [],  # Store ROC AUC values
        'NDCG@10': []  # Store NDCG@10 values
    }

    pattern = re.compile(r'(.+?):\s*([0-9.]+)')  # Regex to match metric names and values

    current_directory = os.path.dirname(os.path.abspath(__file__))  # Get the current directory

    # Iterate over file indices to read multiple metric files
    for i in range(30, 39):
        filename = os.path.join(current_directory, f'Print_metricas_{i}.txt')

        if not os.path.exists(filename):
            print(f"File not found: {filename}")  # Handle missing files
            continue

        # Read each file line by line and extract metrics
        with open(filename, 'r') as file:
            for line in file:
                match = pattern.match(line.strip())
                if match:
                    metric_name = match.group(1).strip()  # Extract metric name
                    value = float(match.group(2))  # Extract metric value
                    if metric_name in metrics:
                        metrics[metric_name].append(value)

    # Calculate average and standard deviation for each metric
    metrics_avg_std = {}
    for metric, values in metrics.items():
        if values:
            avg = np.mean(values)  # Calculate average
            std = np.std(values)  # Calculate standard deviation
            metrics_avg_std[metric] = {'average': avg, 'std_dev': std}
        else:
            metrics_avg_std[metric] = {'average': None, 'std_dev': None}

    return metrics_avg_std

# Function to generate and save a table of metrics (average and standard deviation)
def generate_metrics_table(metrics_avg_std, save_path='metrics_table.png'):
    # Convert dictionary keys to a list for column labels
    col_labels = list(metrics_avg_std.keys())

    # Create a list for the table
    rows = ['Average']
    columns = []

    # Format metrics data for each column
    for metric in col_labels:
        stats = metrics_avg_std[metric]
        if stats['average'] is not None and stats['std_dev'] is not None:
            combined = f"{round(stats['average'], 4)} ± {round(stats['std_dev'], 4)}"
        else:
            combined = 'N/A'
        columns.append([combined])

    # Transpose the columns list to match the new orientation
    columns = list(map(list, zip(*columns)))

    # Plot the table
    fig, ax = plt.subplots(figsize=(12, 6))  # Increased figure size for better readability
    ax.axis('tight')
    ax.axis('off')

    # Create the table with increased padding
    table = ax.table(cellText=columns, rowLabels=rows,
                     colLabels=col_labels, cellLoc='center', loc='center')

    # Set the font size for the table
    table.auto_set_font_size(False)
    table.set_fontsize(14)

    # Adjust column widths and add more padding for readability
    table.scale(1.5, 1.5)

    # Increase the cell height and adjust padding
    for key, cell in table.get_celld().items():
        cell.set_height(0.15)  # Increase the height of each cell
        cell.set_fontsize(14)  # Ensure the font size is consistent
        cell.PAD = 0.2  # Adjust internal padding

    # Save the table as an image
    plt.savefig(save_path, bbox_inches='tight', dpi=300)  # Increase DPI for better resolution
    print(f"Table saved as {save_path}")
    plt.close()

# Call the functions to calculate metrics and generate the table
items = calculate_metrics_average_and_std()
generate_metrics_table(items)
