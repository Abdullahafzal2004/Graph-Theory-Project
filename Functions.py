import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import networkx as nx
import matplotlib.pyplot as plt
from gspan_mining import gSpan
import re
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

nltk.download('punkt')
nltk.download('stopwords')

def combine_and_save_data(filename):
    # Read the three CSV files
    df1 = pd.read_csv("articles/disease_articles.csv")
    df2 = pd.read_csv("articles/science_edu_articles.csv")
    df3 = pd.read_csv("articles/fashion_articles.csv", encoding='latin1')

    # Initialize an empty dataframe to store the combined data
    combined_df = pd.DataFrame()

    # Iterate over the rows of the dataframes and concatenate them row by row
    for i in range(5):
        combined_df = pd.concat([combined_df, df1.iloc[[i]]])
        combined_df = pd.concat([combined_df, df2.iloc[[i]]])
        combined_df = pd.concat([combined_df, df3.iloc[[i]]])

    # Save the combined dataframe to a new CSV file
    combined_df.to_csv(filename, index=False)

    # Print the length of the combined dataframe
    # print("Length of combined_df:", len(combined_df))


# Tokenization
def tokenize(text):
    return nltk.word_tokenize(text.lower())

# Stop-word removal
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens

# Stemming
def stem_tokens(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

# Function to remove numbers, special characters, commas, full stops, and emojis
def clean_text(text):
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove special characters and emojis
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove commas and full stops
    text = text.replace(',', '').replace('.', '')
    return text

def preprocess_data(data):
    preprocessed_data = []

    for index, row in data.iterrows():
        # Clean title and content
        cleaned_title = clean_text(row['title'])
        cleaned_content = clean_text(row['content'])

        # Tokenize cleaned title and content
        title_tokens = tokenize(cleaned_title)
        content_tokens = tokenize(cleaned_content)

        # Remove stop words
        title_tokens = remove_stopwords(title_tokens)
        content_tokens = remove_stopwords(content_tokens)

        # Stem tokens
        title_tokens = stem_tokens(title_tokens)
        content_tokens = stem_tokens(content_tokens)

        # Count words in content
        words_count = len(content_tokens)

        # Append preprocessed data to the list
        preprocessed_data.append({
            'label': row['label'],
            'title_tokens': title_tokens,
            'content_tokens': content_tokens,
            'words_count': words_count
        })

    return preprocessed_data

def save_preprocessed_data(data, output_file):
    # Convert preprocessed data into a DataFrame
    preprocessed_df = pd.DataFrame(data)

    # Save the preprocessed data to a CSV file
    preprocessed_df.to_csv(output_file, index=False)

def generate_graph(data):
    graph = nx.DiGraph()
    for token_list in data['content_tokens']:  # Assuming 'content_tokens' contains lists of tokens
        # Add nodes for unique terms
        for term in token_list:
            graph.add_node(term)
        
        # Add edges between consecutive terms
        for i in range(len(token_list) - 1):
            current_term = token_list[i]
            next_term = token_list[i + 1]
            graph.add_edge(current_term, next_term)
    return graph

# Function to build directed graph
def construct_graph(tokens):
    graph = nx.DiGraph()
    for i in range(len(tokens) - 1):
        if not graph.has_edge(tokens[i], tokens[i+1]):
            graph.add_edge(tokens[i], tokens[i+1], weight=1)
        else:
            graph.edges[tokens[i], tokens[i+1]]['weight'] += 1
    return graph

# Function to plot the graph
def plot_graph(graph):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='lightgreen', node_size=2000, edge_color='blue', linewidths=2, font_size=12, font_weight='bold')
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_color='red', font_size=10)
    plt.show()
# Function to train the classifier
def train_classifier(training_graphs, training_labels):
    # Compute the feature vectors for the training graphs
    feature_vectors = []
    for graph in training_graphs:
        feature_vector = []
        # Append the entire graph structure to the feature vector
        # feature_vector.append(graph)
        # Append the set of nodes to the feature vector
        feature_vector.append(len(graph.nodes()))
        feature_vector.append(len(graph.edges()))  # Number of edges
        feature_vector.append(compute_distances(graph, training_graphs)[0])  # Index of the closest match
        feature_vector.append(np.mean(list(dict(graph.degree()).values())))  # Mean degree
        feature_vector.append(np.std(list(dict(graph.degree()).values())))  # Standard deviation of degree
        feature_vector.append(np.mean(list(nx.clustering(graph).values())))  # Mean clustering coefficient
        feature_vector.append(np.std(list(nx.clustering(graph).values())))  # Standard deviation of clustering coefficient
        feature_vectors.append(feature_vector)

    # Train a k-NN classifier
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(feature_vectors, training_labels)

    return classifier
# Function to compute distances between test graph and all training graphs
def compute_distances(test_graph, training_graphs):
    distances = []
    for training_graph in training_graphs:
        mcs = compute_mcs(test_graph, training_graph)
        # Convert MCS to a distance measure and normalize
        distance = 1 - (mcs / max(len(test_graph.edges()), len(training_graph.edges())))
        distances.append(distance)
    return distances
# Function to compute MCS between two graphs
def compute_mcs(graph1, graph2):
    # Extract the set of edges from each graph
    edges1 = set(graph1.edges())
    edges2 = set(graph2.edges())

    # Compute the set of common edges between the two graphs
    common_edges = edges1.intersection(edges2)

    # Construct a new graph using the common edges, representing the maximal common subgraph
    mcs_graph = nx.Graph(list(common_edges))

    # Calculate the graph distance (negative size of the maximal common subgraph)
    distance = -len(mcs_graph.edges())

    return distance