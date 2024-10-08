from tabulate import tabulate  # Import the 'tabulate' library for generating tables

# Function to get the maximum score from a file
def getMaxScore(filename):
    with open(filename, 'r') as file:
        scores = [float(line.strip()) for line in file.readlines()]
        max_score = max(scores)
        return max_score

# Function to get the minimum score from a file
def getMinScore(filename):
    with open(filename, 'r') as file:
        scores = [float(line.strip()) for line in file.readlines()]
        max_score = min(scores)
        return max_score

# Function to count the total number of lines in a file
def count_lines(file_path):
    with open(file_path, 'r') as file:
        line_count = sum(1 for line in file)
    return line_count

# Function to find lines that have scores above a certain threshold
def find_lines_above_threshold(filename, threshold):
    try:
        lines_above_threshold = 0
        line_numbers = []

        with open(filename, 'r') as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    score = float(line.strip())
                except ValueError:
                    print(f"Error parsing score on line {line_number}: {line.strip()}")
                    continue  

                if score > threshold:
                    lines_above_threshold += 1
                    line_numbers.append(line_number)

        return lines_above_threshold, line_numbers

    except FileNotFoundError:
        print("File not found. Please ensure the file exists in the current directory.")
        return 0, []
    except Exception as e:
        print("An error occurred:", e)
        return 0, []

# Function to fetch class labels from a file based on given line numbers
def fetch_classes_from_file(filename, line_numbers):
    try:
        classes = []

        with open(filename, 'r') as file:
            for line_number, line in enumerate(file, start=1):
                if line_number in line_numbers:
                    class_label = int(line.strip())
                    classes.append(class_label)

        return classes

    except FileNotFoundError:
        print("File not found. Please ensure the file exists in the current directory.")
        return []
    except ValueError as e:
        print(f"Error converting class label to integer: {e}")
        return []
    except Exception as e:
        print("An error occurred:", e)
        return []

# Function to count the number of '1's in a list of class labels and calculate their percentage
def count_ones(classes):
    ones_count = classes.count(1)
    
    if len(classes) > 0:
        percentage = (ones_count / len(classes)) * 100
    else:
        percentage = 0
    
    return ones_count, percentage

# Function to count all occurrences of '1' in a file
def count_all_ones(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        ones_count = 0
        
        for line in lines:
            cleaned_line = line.strip()
            
            if '1' in cleaned_line:
                ones_count += cleaned_line.count('1')
        
        return ones_count

# Function to generate a table and save it based on BM25 results and class labels
def generate_and_save_table(results_diet_train_path, train_class_path, graphics_path):
    from tabulate import tabulate
    
    top_values = []
    relevant_files_counts = []
    relevant_files_percentages = []
    total_files_counts = []

    max_score = getMaxScore(results_diet_train_path)
    top_value = max_score

    while top_value >= -2:
        count, line_numbers = find_lines_above_threshold(results_diet_train_path, int(top_value))
        classes = fetch_classes_from_file(train_class_path, line_numbers)
        ones_count, percentage = count_ones(classes)
        total_files_count = count_lines(train_class_path) - 1

        top_values.append(top_value)
        relevant_files_counts.append(ones_count)
        relevant_files_percentages.append(percentage)
        total_files_counts.append(count)

        top_value -= 2

    table_data = []
    for i in range(len(top_values)):
        relevant_files_count = relevant_files_counts[i]
        count_all_ones_value = count_all_ones(train_class_path)
        success = "Yes" if relevant_files_count == count_all_ones_value else "No"
        if success == "Yes":
            eliminated_titles = total_files_count - total_files_counts[i]
        else:
            eliminated_titles = "-"
        table_data.append([top_values[i], relevant_files_count, relevant_files_percentages[i], total_files_counts[i], success, eliminated_titles])

    headers = ["Top Value", "Number of Relevant Files", "Percentage of Relevant Files", "Total Number of Files", "Success", "Titles Eliminated"]

    table_str = tabulate(table_data, headers=headers, tablefmt="fancy_grid", numalign="center", stralign="center")

    with open(graphics_path, "w", encoding="utf-8") as file:
        file.write(table_str)

    print("Table saved to " + graphics_path)

# Function to generate a table with a dynamic decrement based on max and min scores
def generate_and_save_table1(results_diet_train_path, train_class_path, graphics_path):
    top_values = []
    relevant_files_counts = []
    relevant_files_percentages = []
    total_files_counts = []

    max_score = getMaxScore(results_diet_train_path)
    min_score = getMinScore(results_diet_train_path)

    decrement_value = (max_score - min_score) / 20

    top_value = max_score

    while top_value >= min_score:
        count, line_numbers = find_lines_above_threshold(results_diet_train_path, int(top_value))
        classes = fetch_classes_from_file(train_class_path, line_numbers)
        ones_count, percentage = count_ones(classes)
        total_files_count = len(classes)

        top_values.append(top_value)
        relevant_files_counts.append(ones_count)
        relevant_files_percentages.append(percentage)
        total_files_counts.append(count)

        top_value -= decrement_value

    table_data = []
    for i in range(len(top_values)):
        table_data.append([top_values[i], relevant_files_counts[i], relevant_files_percentages[i], total_files_counts[i]])

    headers = ["Top Value", "Number of Relevant Files", "Percentage of Relevant Files", "Total Number of Files"]

    table_str = tabulate(table_data, headers=headers, tablefmt="fancy_grid", numalign="center", stralign="center")

    with open(graphics_path, "w", encoding="utf-8") as file:
        file.write(table_str)

    print("Table saved to " + graphics_path)

# Function to remove non-relevant articles based on BM25 scores and a specified threshold
def remove_non_relevant_articles(results_path, abstracts_titles_path, threshold, class_file_path, output_abstracts_titles_path, output_class_path, target_removals=140):
    """
    Removes non-relevant articles based on BM25 scores and a specified threshold, 
    from both abstracts+titles and class files.

    Parameters:
        results_path (str): Path to the file containing BM25 scores.
        abstracts_titles_path (str): Path to the file containing abstracts and titles.
        class_file_path (str): Path to the file containing class labels.
        output_abstracts_titles_path (str): Path where the filtered abstracts and titles will be saved.
        output_class_path (str): Path where the filtered class labels will be saved.
        threshold (float): Score threshold for determining non-relevant articles.
        target_removals (int): The exact number of articles to remove.
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        scores = [float(line.strip()) for line in f if line.strip()]

    low_score_indices = [idx for idx, score in enumerate(scores) if score < threshold]
    num_low_scores = len(low_score_indices)

    if num_low_scores == target_removals:
        indices_to_remove = low_score_indices
    elif num_low_scores >= target_removals:
        sorted_indices = sorted(low_score_indices, key=lambda idx: scores[idx])
        indices_to_remove = sorted_indices[:target_removals]
    else:
        print(f"Warning: Only {num_low_scores} documents have scores below the threshold {threshold}.")
        print(f"Proceeding to remove {num_low_scores} documents instead of {target_removals}.")
        indices_to_remove = low_score_indices

    indices_to_remove_set = set(indices_to_remove)
    
    with open(abstracts_titles_path, 'r', encoding='utf-8') as f:
        documents = f.readlines()

    with open(class_file_path, 'r', encoding='utf-8') as f:
        classes = f.readlines()

    if len(documents) != len(classes):
        raise ValueError("Mismatch between number of documents and class labels")

    filtered_documents = [doc for idx, doc in enumerate(documents) if idx not in indices_to_remove_set]
    filtered_classes = [cls for idx, cls in enumerate(classes) if idx not in indices_to_remove_set]

    with open(output_abstracts_titles_path, 'w', encoding='utf-8') as f:
        f.writelines(filtered_documents)

    with open(output_class_path, 'w', encoding='utf-8') as f:
        f.writelines(filtered_classes)

    removed_documents = len(documents) - len(filtered_documents)

    print(f"Total documents in original file: {len(documents)}")
    print(f"Documents removed: {removed_documents}")
    print(f"Documents remaining: {len(filtered_documents)}")
    print(f"Filtered documents saved to: {output_abstracts_titles_path}")
    print(f"Filtered classes saved to: {output_class_path}")
