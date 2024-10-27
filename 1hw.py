#!/usr/bin/env python3

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import random
import nltk

from collections import Counter, defaultdict
import matplotlib.pyplot as plt


# Load documents

newsgroups_train = fetch_20newsgroups(subset='train')
print(len(newsgroups_train.data), " documents loaded.")

print("Example document:")
print(newsgroups_train.data[0])

# nltk.download('wordnet')


# Preprocess documents - lemmatization and stemming

def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

processed_docs = list(map(preprocess, newsgroups_train.data))

print("Example document - lemmatized and stemmed:")
print(processed_docs[0])


# Construct dictionary

dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

print("Dictionary size: ", len(dictionary))

# Filter words in documents

docs = list()
maxdoclen = 0
for doc in processed_docs:
    docs.append(list(filter(lambda x: x != -1, dictionary.doc2idx(doc))))
    maxdoclen = max(maxdoclen, len(docs[-1]))

print("Example document - filtered:")
print(docs[0])

print("Maximum document length:", maxdoclen)


# Set the hyperparameters

iterations = 100
topics = 20
alpha = 0.01
gamma = 0.01

doc_cnt = len(docs)
wrd_cnt = len(dictionary)

print(doc_cnt, wrd_cnt)


def initialize_lda(documents, K, alpha, gamma):
    """
    Initialize the LDA parameters.
    Parameters:
        documents (list of list of int): List of documents with words represented by integers.
        K (int): Number of topics.
        alpha (float): Document-topic distribution prior.
        gamma (float): Topic-word distribution prior.
    Returns:
        word_topic_counts (np.array): Counts of topics assigned to each word in each document.
        doc_topic_counts (np.array): Document-topic counts.
        topic_word_counts (np.array): Topic-word counts.
        topic_counts (np.array): Total counts of each topic.
    """
    D = len(documents)  # Number of documents
    V = max(max(doc) for doc in documents) + 1  # Vocabulary size based on max word id

    # Initialize count matrices
    doc_topic_counts = np.zeros((D, K)) + alpha  # Document-topic counts
    topic_word_counts = np.zeros((K, V)) + gamma  # Topic-word counts
    topic_counts = np.zeros(K) + V * gamma  # Total number of words in each topic

    # Random topic assignments to each word
    word_topic_assignments = []
    for d, doc in enumerate(documents):
        current_doc_assignments = []
        for word in doc:
            topic = np.random.randint(0, K)  # Randomly assign a topic
            current_doc_assignments.append(topic)

            # Update counts
            doc_topic_counts[d, topic] += 1
            topic_word_counts[topic, word] += 1
            topic_counts[topic] += 1

        word_topic_assignments.append(current_doc_assignments)

    return doc_topic_counts, topic_word_counts, topic_counts, word_topic_assignments


def gibbs_sampling(documents, K, doc_topic_counts, topic_word_counts, topic_counts, word_topic_assignments):
    """
    Perform a single iteration of Gibbs sampling.
    Parameters:
        documents (list of list of int): List of documents with words represented by integers.
        K (int): Number of topics.
        doc_topic_counts (np.array): Document-topic counts.
        topic_word_counts (np.array): Topic-word counts.
        topic_counts (np.array): Total counts of each topic.
        word_topic_assignments (list of list of int): Current topic assignments per word in each document.
    """
    for d, doc in enumerate(documents):
        for i, word in enumerate(doc):
            current_topic = word_topic_assignments[d][i]

            # Decrement counts for the current word's topic assignment
            doc_topic_counts[d, current_topic] -= 1
            topic_word_counts[current_topic, word] -= 1
            topic_counts[current_topic] -= 1

            # Compute topic probabilities
            topic_probs = ((topic_word_counts[:, word] / topic_counts) *
                           (doc_topic_counts[d, :] / np.sum(doc_topic_counts[d, :])))
            topic_probs /= np.sum(topic_probs)  # Normalize to make a probability distribution

            # Sample a new topic
            new_topic = np.random.choice(K, p=topic_probs)
            word_topic_assignments[d][i] = new_topic  # Assign new topic

            # Increment counts with the new topic assignment
            doc_topic_counts[d, new_topic] += 1
            topic_word_counts[new_topic, word] += 1
            topic_counts[new_topic] += 1


def run_lda(documents, K=20, alpha=0.1, gamma=0.1, iterations=100):
    """
    Run LDA model with Gibbs sampling for a given number of iterations.
    Parameters:
        documents (list of list of int): List of documents with words represented by integers.
        K (int): Number of topics.
        alpha (float): Document-topic distribution prior.
        gamma (float): Topic-word distribution prior.
        iterations (int): Number of Gibbs sampling iterations.
    Returns:
        doc_topic_counts (np.array): Document-topic counts.
        topic_word_counts (np.array): Topic-word counts.
        topic_counts (np.array): Total counts of each topic.
        word_topic_assignments (list of list of int): Final topic assignments per word in each document.
    """
    # Initialize LDA
    doc_topic_counts, topic_word_counts, topic_counts, word_topic_assignments = initialize_lda(documents, K, alpha,
                                                                                               gamma)

    # Run Gibbs sampling for specified iterations
    for iter in range(iterations):
        gibbs_sampling(documents, K, doc_topic_counts, topic_word_counts, topic_counts, word_topic_assignments)

        # Optionally, log progress or visualize intermediate results here
        if iter in [0, 1, 5, 10, 20, 50, 100]:
            print(f"Iteration {iter}: Gibbs sampling in progress")

    return doc_topic_counts, topic_word_counts, topic_counts, word_topic_assignments


# Fetch and preprocess your dataset
newsgroups_data = fetch_20newsgroups(subset='train')
vectorizer = CountVectorizer(max_features=1000, stop_words='english')  # Restrict to 1000 most common words
documents = [list(row.nonzero()[1]) for row in vectorizer.fit_transform(newsgroups_data.data)]

# Run the LDA model
doc_topic_counts, topic_word_counts, topic_counts, word_topic_assignments = run_lda(documents, K=20, alpha=0.1, gamma=0.1, iterations=100)
