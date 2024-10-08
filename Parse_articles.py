
# Function to convert the contents of a file to a string and replace commas followed by spaces with a single space
def text_to_string(filename):
    with open(filename, 'r', encoding='utf-8-sig') as file:
        text = file.read()
    text = text.replace(', ', ' ')
    return text

# Function to write a list of results (scores) to a file, with each score on a new line
def write_results_to_file(filename, results):
    with open(filename, 'w') as f:
        for result_array in results:
            for score in result_array:
                f.write(str(score) + '\n')

# Function to read all lines from a file and return them as a list
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

# Function to aggregate titles and abstracts into a single output file, combining them line by line
def aggregate_titles_and_abstracts(titles_file, abstracts_file, output_file):
    with open(titles_file, 'r', encoding='utf-8') as file:
        titles = file.readlines()
    
    with open(abstracts_file, 'r', encoding='utf-8') as file:
        abstracts = file.readlines()

    titles = [title.strip() for title in titles]
    abstracts = [abstract.strip() for abstract in abstracts]

    with open(output_file, 'w', encoding='utf-8') as file:
        for title, abstract in zip(titles, abstracts):
            file.write(f"{title}{abstract}\n")

# Function to read entries from a file, splitting them by newlines, and returning them as a list
def read_entries_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        entries = file.read().split('\n')
    return entries
