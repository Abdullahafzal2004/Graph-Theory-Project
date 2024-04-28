from Functions import *
import networkx as nx
import matplotlib.pyplot as plt

# Data Collection and Scrapping

# Combine and save data from scrapped articles to csv file "uncleaned_data.csv"
filename = 'uncleaned_data.csv'
filename_preprocess = 'preprocessed_data.csv'
combine_and_save_data(filename)

# Read the uncleaned data
uncleaned_data = pd.read_csv(filename)

# Separating into train and test data
train_set = uncleaned_data.iloc[:12]  # Access the first 12 rows
test_set = uncleaned_data.iloc[12:]   # Access the remaining rows


# Pre-Processing data on train
preprocessed_data = preprocess_data(train_set)

# Save preprocessed train dataset to CSV
save_preprocessed_data(preprocessed_data, filename_preprocess)

preprocessed_df = pd.DataFrame(preprocessed_data)
# filename_preprocess = 'preprocessed_data.csv'



# Graph construction

graphs_train_set = []
for index, row in preprocessed_df.iterrows():
    # Build the directed graph
    graph = construct_graph(row['content_tokens'])
    graphs_train_set.append(graph)


print("Graph of the first article in the training set")
plot_graph(graphs_train_set[2])

# Feature Extraction via Common Subgraphs:

training_labels = train_set['label'].tolist()
classifier = train_classifier(graphs_train_set, training_labels)
