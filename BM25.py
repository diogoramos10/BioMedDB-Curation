from rank_bm25 import BM25Okapi, BM25L,BM25Plus
import os
from Parse_articles import text_to_string,write_results_to_file,aggregate_titles_and_abstracts,read_entries_from_file
from Graphics import remove_non_relevant_articles,getMaxScore,find_lines_above_threshold,fetch_classes_from_file,count_ones,generate_and_save_table,generate_and_save_table1

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))
# Define the path for 'Pollutants' folder
#Pollutants_Path = os.path.join(script_directory, 'Pollutants')

# DIET paths setup
Diet_path = os.path.join(script_directory, 'Diet') # Path to the 'Diet' folder

train_path = os.path.join(Diet_path, 'Train')  # Path for training data

# Paths to specific files within the training dataset of the Diet project
train_abstracts_path = os.path.join(train_path, 'diet_train_abstracts.txt') # Path to abstracts file
train_titles_path = os.path.join(train_path, 'diet_train_titles.txt') # Path to titles file
train_class_path = os.path.join(train_path,'diet_train_class.txt') # Path to class labels file
train_all_path = os.path.join(train_path, 'abstracts+titles.txt') # Combined file path (abstracts and titles)
graphics_path = os.path.join(train_path, 'graphicsBM25Plus.txt') # Path to save graphical results (BM25Plus)
results_path = os.path.join(train_path, 'results.txt') # Path to save results
output_path1 = os.path.join(train_path, 'abstracts+titles1.txt') # Filtered abstracts and titles file
output_path2 = os.path.join(train_path,'diet_train_class1.txt') # Filtered class labels file
results_diet_train_path = os.path.join(train_path,'results.txt') # Path for saving results of test data

# The following code block is commented out for testing paths in Diet dataset

"""test_path = os.path.join(Diet_path, 'Test') # Path for test dataset

test_abstracts_path = os.path.join(test_path, 'diet_test_abstracts.txt') # Test dataset abstracts path
test_titles_path = os.path.join(test_path, 'diet_test_titles.txt') # Test dataset titles path

test_all_path = os.path.join(test_path, 'abstracts+titles.txt') # Combined abstracts and titles for test data
graphics1_path = os.path.join(test_path, 'graphics.txt') # Graphics path for test data
test_class_path = os.path.join(test_path,'diet_test_class.txt') # Path for class labels in test data
queries_path = os.path.join(Diet_path,'Diet_queries.txt') # Query file path for diet data
results_diet_test_path = os.path.join(test_path,'results.txt') # Path for saving results of test data
output_path1 = os.path.join(test_path, 'abstracts+titles1.txt') # Filtered abstracts for test data
output_path2 = os.path.join(test_path, 'diet_test_class1.txt') # Filtered class labels for test data
"""

# Queries path for DIET dataset
queries_path = os.path.join(Diet_path,'Diet_queries.txt') 

# Reproducibility dataset paths setup

"""REPRO

Repro_Path = os.path.join(script_directory, 'Reproducibility') # Path for reproducibility dataset

abstracts_path = os.path.join(Repro_Path, 'reproducibility_abstracts.txt') # Reproducibility abstracts path
titles_path = os.path.join(Repro_Path, 'reproducibility_titles.txt') # Reproducibility titles path
all_path = os.path.join(Repro_Path, 'abstracts+titles.txt') # Combined abstracts and titles
class_path = os.path.join(Repro_Path,'reproducibility_class.txt') # Class labels path

queries_path = os.path.join(Repro_Path,'Repro_queries.txt') # Query file path for reproducibility dataset
results_path = os.path.join(Repro_Path,'results.txt') # Path for results
graphics_path = os.path.join(Repro_Path, 'graphics.txt') # Path for graphics
"""

# DPBS dataset paths

"""DPBS

DPBS_path = os.path.join(Pollutants_Path, 'DPBS') # Path for DPBS data

abstracts_path = os.path.join(DPBS_path, 'dbps_abstracts.txt') # Abstracts for DPBS
titles_path = os.path.join(DPBS_path, 'dbps_titles.txt') # Titles for DPBS
all_path = os.path.join(DPBS_path, 'abstracts+titles.txt') # Combined abstracts and titles
class_path = os.path.join(DPBS_path,'dbps_class.txt') # Class labels for DPBS

queries_path = os.path.join(DPBS_path,'dbps_queries.txt') # Queries file for DPBS
results_path = os.path.join(DPBS_path,'results.txt') # Results file for DPBS
graphics_path = os.path.join(DPBS_path, 'graphics.txt') # Path for graphical output
"""

# Polychlorinated dataset paths

"""Polychlorinated

Polychlorinated_path = os.path.join(Pollutants_Path, 'POLYCLORINATED') # Path for polychlorinated data

abstracts_path = os.path.join(Polychlorinated_path, 'polychlorinated_abstracts.txt') # Abstracts for polychlorinated data
titles_path = os.path.join(Polychlorinated_path, 'polychlorinated_titles.txt') # Titles for polychlorinated data
all_path = os.path.join(Polychlorinated_path, 'abstracts+titles.txt') # Combined abstracts and titles
class_path = os.path.join(Polychlorinated_path,'polychlorinated_class.txt') # Class labels for polychlorinated data

queries_path = os.path.join(Polychlorinated_path,'poly_queries.txt') # Queries file for polychlorinated data
results_path = os.path.join(Polychlorinated_path,'results.txt') # Results file for polychlorinated data
graphics_path = os.path.join(Polychlorinated_path, 'graphics.txt') # Path for graphical output
"""

# Polybrominated dataset paths

"""Polybrominated

Polybrominated_path = os.path.join(Pollutants_Path, 'POLYBROMINATED') # Path for polybrominated data

abstracts_path = os.path.join(Polybrominated_path, 'polybrominated_abstracts.txt') # Abstracts for polybrominated data
titles_path = os.path.join(Polybrominated_path, 'polybrominated_titles.txt') # Titles for polybrominated data
all_path = os.path.join(Polybrominated_path, 'abstracts+titles.txt') # Combined abstracts and titles
class_path = os.path.join(Polybrominated_path,'polybrominated_class.txt') # Class labels for polybrominated data

queries_path = os.path.join(Polybrominated_path,'poly_queries.txt') # Queries file for polybrominated data
results_path = os.path.join(Polybrominated_path,'results.txt') # Results file for polybrominated data
graphics_path = os.path.join(Polybrominated_path, 'graphics.txt') # Path for graphical output
"""

# PCB dataset paths

"""PCB
PCB_path = os.path.join(Pollutants_Path, 'PCB') # Path for PCB data

abstracts_path = os.path.join(PCB_path, 'pcb_abstracts.txt') # Abstracts for PCB data
titles_path = os.path.join(PCB_path, 'pcb_titles.txt') # Titles for PCB data
all_path = os.path.join(PCB_path, 'abstracts+titles.txt') # Combined abstracts and titles
class_path = os.path.join(PCB_path,'pcb_class.txt') # Class labels for PCB data

queries_path = os.path.join(PCB_path,'pcb_queries.txt') # Queries file for PCB data
results_path = os.path.join(PCB_path,'results.txt') # Results file for PCB data
graphics_path = os.path.join(PCB_path, 'graphics.txt') # Path for graphical output
"""

# Phthalates dataset paths

"""PHTALATES

PHTALATES_path = os.path.join(Pollutants_Path, 'PHTALATES') # Path for Phthalates data

abstracts_path = os.path.join(PHTALATES_path, 'phthalates_abstracts.txt') # Abstracts for phthalates data
titles_path = os.path.join(PHTALATES_path, 'phthalates_titles.txt') # Titles for phthalates data
all_path = os.path.join(PHTALATES_path, 'abstracts+titles.txt') # Combined abstracts and titles
class_path = os.path.join(PHTALATES_path,'phthalates_class.txt') # Class labels for phthalates data

queries_path = os.path.join(PHTALATES_path,'phtalates_queries.txt') # Queries file for phthalates data
results_path = os.path.join(PHTALATES_path,'results.txt') # Results file for phthalates data
graphics_path = os.path.join(PHTALATES_path, 'graphics.txt') # Path for graphical output
"""

# HCA dataset paths

"""HCA

HCA_path = os.path.join(Pollutants_Path, 'HCA') # Path for HCA data

abstracts_path = os.path.join(HCA_path, 'hca_abstracts.txt') # Abstracts for HCA data
titles_path = os.path.join(HCA_path, 'hca_titles.txt') # Titles for HCA data
all_path = os.path.join(HCA_path, 'abstracts+titles.txt') # Combined abstracts and titles
class_path = os.path.join(HCA_path,'hca_class.txt') # Class labels for HCA data

queries_path = os.path.join(HCA_path,'HCA_queries.txt') # Queries file for HCA data
results_path = os.path.join(HCA_path,'results.txt') # Results file for HCA data
graphics_path = os.path.join(HCA_path, 'graphics.txt') # Path for graphical output
"""

# PAH dataset paths

"""PAH

PAH_path = os.path.join(Pollutants_Path, 'PAH') # Path for PAH data

abstracts_path = os.path.join(PAH_path, 'pah_abstracts.txt') # Abstracts for PAH data
titles_path = os.path.join(PAH_path, 'pah_titles.txt') # Titles for PAH data
all_path = os.path.join(PAH_path, 'abstracts+titles.txt') # Combined abstracts and titles
class_path = os.path.join(PAH_path,'pah_class.txt') # Class labels for PAH data

queries_path = os.path.join(PAH_path,'PAH_queries.txt') # Queries file for PAH data
results_path = os.path.join(PAH_path,'results.txt') # Results file for PAH data
graphics_path = os.path.join(PAH_path, 'graphics.txt') # Path for graphical output
"""

# All datasets combined

"""All

All_path = os.path.join(script_directory, 'All') # Path for All datasets

queries_path = os.path.join(All_path,'All_queries.txt') # Queries file for All datasets

train_path = os.path.join(All_path, 'Train') # Path for train data for All datasets

train_class_path = os.path.join(train_path,'all_train_class.txt') # Train class file for All datasets
train_all_path = os.path.join(train_path, 'abstracts+titles2.txt') # Combined abstracts and titles for All datasets
graphics_path = os.path.join(train_path, 'graphics.txt') # Path for graphical output
results_diet_train_path = os.path.join(train_path,'results.txt') # Results file for All datasets

test_path = os.path.join(All_path, 'Test') # Path for test data for All datasets

test_all_path = os.path.join(test_path, 'abstracts+titles3.txt') # Combined abstracts and titles for test data
graphics1_path = os.path.join(test_path, 'graphicsAll.txt') # Graphics path for test data
test_class_path = os.path.join(test_path,'all_test_class.txt') # Class file for test data
results_diet_train_path = os.path.join(train_path,'results.txt') # Results file for train data
results_diet_test_path = os.path.join(test_path,'results.txt') # Results file for test data
"""

# DIET data aggregation (commented out)
aggregate_titles_and_abstracts(train_titles_path, train_abstracts_path,train_all_path)
#aggregate_titles_and_abstracts(test_titles_path, test_abstracts_path,test_all_path)

# Every other dataset
# aggregate_titles_and_abstracts(titles_path, abstracts_path,all_path)

# Read corpus entries (abstracts and titles) from file
# DIET
article_list = read_entries_from_file(train_all_path) 
#article_list = read_entries_from_file(test_all_path)

# For All datasets
#article_list = read_entries_from_file(train_all_path)
# article_list = read_entries_from_file(test_all_path)
# article_list = read_entries_from_file(all_path)

# Load queries as text
queries = text_to_string(queries_path)
# Tokenize the query by splitting into words
tokenized_query = queries.split(" ")

# Initialize an empty list to store BM25 scores
list_Scores = []

# Tokenize the corpus by splitting documents into words
tokenized_corpus = [doc.split(" ") for doc in article_list]
# Initialize BM25Okapi for ranking
bm25 = BM25Okapi(tokenized_corpus)
# Compute the scores of documents based on the query
doc_scores = bm25.get_scores(tokenized_query)
# Calculate average score of documents
avg_score = sum(doc_scores) / len(doc_scores) 
# Append scores to list_Scores
list_Scores.append(doc_scores)

# Write results for DIET (commented out)
write_results_to_file(results_diet_train_path, list_Scores)
#write_results_to_file(results_diet_test_path, list_Scores)

# Write results for All datasets
# write_results_to_file(results_diet_train_path, list_Scores)
# write_results_to_file(results_diet_test_path, list_Scores)

# For other datasets (commented out)
# write_results_to_file(results_path, list_Scores)

# Generate and save results for DIET
generate_and_save_table(results_diet_train_path, train_class_path,graphics_path)
#generate_and_save_table(results_diet_test_path, test_class_path,graphics1_path)

# Generate and save results for other datasets
# generate_and_save_table(results_path,test_class_path, graphics_path)

# Remove non-relevant articles based on a score threshold
#remove_non_relevant_articles(results_path, train_all_path, 21.9584,train_class_path, output_path1,output_path2)
