from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, roc_auc_score

# Function to read entries from a file and return them as a list of strings
def read_entries_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        entries = file.read().split('\n')  # Read the file and split it into a list by new lines
    return entries

# Function to read the content of a file into a single string and replace commas followed by spaces
def text_to_string(filename):
    with open(filename, 'r', encoding='utf-8-sig') as file:
        text = file.read()  # Read the entire file content
    
    # Replace occurrences of ', ' with a space
    text = text.replace(', ', ' ')

    return text

# Function to calculate various classification metrics based on document scores and true labels
def calculate_metrics(document_scores, true_labels):
    threshold = 0.5  # Set threshold for binary classification
    # Assign predicted class based on threshold: 1 if score >= threshold, else 0
    predicted_classes = [1 if score >= threshold else 0 for score in document_scores]
    
    # Calculate precision, recall, F1, F2, and ROC AUC using sklearn metrics
    precision = precision_score(true_labels, predicted_classes)
    recall = recall_score(true_labels, predicted_classes)
    f1 = f1_score(true_labels, predicted_classes)
    f2 = fbeta_score(true_labels, predicted_classes, beta=2)  # F-beta score with beta = 2
    roc_auc = roc_auc_score(true_labels, document_scores)
    
    # Return calculated metrics and the predicted classes
    return precision, recall, f1, f2, roc_auc, predicted_classes

# Function to read a specific line from a file and return its value, handling errors appropriately
def get_value_from_line(file_path, line_number):
    try:
        # Open the file and read all lines
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # Ensure the line number is within valid range
            if 1 <= line_number <= len(lines):
                try:
                    # Convert the selected line to an integer
                    num = int(lines[line_number - 1].strip())
                    # Return the value if it's valid (0 or 1), otherwise print a warning
                    if num in (0, 1):
                        return num
                    else:
                        print(f"Warning: Line {line_number} contains an invalid number. Only 0 or 1 allowed.")
                except ValueError:
                    # Handle the case where the line content is non-numeric
                    print(f"Warning: Line {line_number} is non-numeric.")
            else:
                # Error message if the requested line number is out of range
                print(f"Error: Line number {line_number} is out of range.")
    except FileNotFoundError:
        # Error message if the file is not found
        print(f"Error: File '{file_path}' not found.")
    
    return None

# Function to read true labels from a file given a range of lines (from start_line to end_line)
def read_true_labels(file_path, start_line, end_line):
    with open(file_path, 'r') as file:
        lines = file.readlines()  # Read all lines in the file
        # Convert each line to an integer, stripping whitespace
        true_labels = [int(line.strip()) for line in lines[start_line:end_line]]
    return true_labels

# Function to count the number of tokens (input IDs) in a list
def count_tokens(input_ids):
    return len(input_ids)  # Return the length of the input list (number of tokens)
