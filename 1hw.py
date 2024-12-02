#!/usr/bin/env python3

from sklearn.datasets import fetch_20newsgroups
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load documents
newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True)
newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True)

# Set the hyperparameters
iterations = 100
topics = 20
alpha = 0.05
gamma = 0.01

chosen_topics = [0, 12, 14]
temp_dir = "/lnet/work/people/lutsai/unlp/temp"
plot_dir = "/lnet/work/people/lutsai/unlp/plot"


def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


def prepare_data(newsgroups_source, output_dir, dictionary=None, save=False):
    preprocess_out = f"{output_dir}/dataset.csv"
    dict_out = f"{output_dir}/vocab.dict"

    print(f"Data\tsize: {len(newsgroups_source.filenames)}")

    print(f"Data\ttopics: {len(newsgroups_source.target_names)}")
    for i, topic in enumerate(newsgroups_source.target_names):
        print(f"{i} - {topic}")

    print("Data\tExample document:")
    print(newsgroups_source.data[0])

    print(f"Data\tshape: {newsgroups_source.filenames.shape}")
    print(f"Data\tTarget shape: {newsgroups_source.target.shape}")

    print("- + - + - Running stemming and lemmatization - + - + - ")

    processed_docs = list(map(preprocess, newsgroups_source.data))
    print("Data\tExample document - lemmatized and stemmed:")
    print(processed_docs[0])

    # Construct dictionary
    print("- + - + - Creating vocabulary dictionary - + - + - ")
    if dictionary is None and save:
        dictionary = gensim.corpora.Dictionary(processed_docs)
        dictionary.save(dict_out)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    print("Data\tDictionary size: ", len(dictionary))

    # Filter words in documents

    print(f"- + - + - Filtering words - + - + - ")
    docs = list()
    maxdoclen = 0
    for doc in processed_docs:
        docs.append(list(filter(lambda x: x != -1, dictionary.doc2idx(doc))))
        maxdoclen = max(maxdoclen, len(docs[-1]))
    print("Data\tExample document - filtered:")
    print(docs[0])

    print("Data\tMaximum document length:", maxdoclen)

    doc_cnt = len(docs)
    wrd_cnt = len(dictionary)

    print(f"Data\tDocument count: {doc_cnt}")
    print(f"Data\tWord count: {wrd_cnt}")

    return docs, dictionary


def initialize_lda(documents, K, alpha, gamma, output_dir):
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
    V = int(max(max(doc) for doc in documents) + 1)  # Vocabulary size based on max word id
    print(f"Initializing LDA params for {D} documents with {V} vocabulary size...")


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

    DK_data = pd.DataFrame(doc_topic_counts)
    KV_data = pd.DataFrame(topic_word_counts)
    K_data = pd.DataFrame(topic_counts)
    VK_data = pd.DataFrame(word_topic_assignments)

    print(f"DOC-Topic:\t{DK_data.shape}")
    print(f"Topic-VOCAB:\t{KV_data.shape}")
    print(f"Topic:\t{K_data.shape}")
    print(f"VOCAB-Topic:\t{VK_data.shape}")

    return doc_topic_counts, topic_word_counts, topic_counts, word_topic_assignments


def new_doc_topic(documents, doc_topic_counts, topic_word_counts, topic_counts, word_topic_assignments, K, alpha, gamma):
    vocab_size = topic_word_counts.shape[0]
    res = []
    for d, doc_words in enumerate(documents):
        Nd = np.sum(doc_topic_counts[d, :])
        doc_a = []
        for n, dw in enumerate(doc_words):
            # print(d, n)
            assigned = word_topic_assignments[d][n]
            best_k = np.argmax(assigned)

            doc_topic_counts[d, best_k] -= 1

            recomp = np.zeros(K)
            for k in range(K):
                recomp[k] = ((alpha + doc_topic_counts[d, k]) / (K * alpha + Nd - 1)) * ((gamma + topic_word_counts[k, dw]) / (vocab_size * gamma + topic_counts[k]))

            doc_a.append(recomp)
            best_k = np.argmax(recomp)

            doc_topic_counts[d, best_k] += 1
        res.append(doc_a)
    return doc_topic_counts, res

def gibbs_sampling(documents, K, doc_topic_counts, topic_word_counts, topic_counts, word_topic_assignments, alpha, gamma):
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
    print(f"- + - + - Gibbs sampling - + - + - ")
    for d, doc in enumerate(documents):
        for i, word in enumerate(doc):
            current_topic = word_topic_assignments[d][i]

            # Decrement counts for the current word's topic assignment
            # print(doc_topic_counts[d, current_topic])
            doc_topic_counts[d, current_topic] -= 1
            topic_word_counts[current_topic, word] -= 1
            topic_counts[current_topic] -= 1

            topic_probs = (((topic_word_counts[:, word] + alpha) / (alpha * topic_counts + len(doc) - 1)) *
                           ((doc_topic_counts[d, :] + gamma) / (topic_word_counts.shape[0] * gamma + topic_counts)))

            topic_probs /= np.sum(topic_probs)  # Normalize to make a probability distribution

            # Sample a new topic
            new_topic = np.random.choice(K, p=topic_probs)
            word_topic_assignments[d][i] = new_topic  # Assign new topic

            # Increment counts with the new topic assignment
            doc_topic_counts[d, new_topic] += 1
            topic_word_counts[new_topic, word] += 1
            topic_counts[new_topic] += 1


def run_lda(documents, test_docs, dictionary, output_dir, K=20, alpha=0.1, gamma=0.1, iterations=100):
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
                                                                                               gamma, output_dir)
    entropy_record = []
    get_longest_document_plot(doc_topic_counts, "init")
    # freq_words(topic_word_counts, chosen_topics, dictionary)
    # Run Gibbs sampling for specified iterations

    test_doc_topic_counts, ttwc, ttc, test_word_topic_assignments = initialize_lda(test_docs, topics, alpha, gamma, temp_dir)
    test_doc_topic_counts, test_word_topic_assignments = new_doc_topic(test_docs, test_doc_topic_counts, topic_word_counts, topic_counts, test_word_topic_assignments, topics, gamma, alpha)
    entropy = comp_word_entropy(test_docs, topic_word_counts, test_doc_topic_counts, gamma, alpha)
    print(f"Initial test entropy:\t{entropy}")
    print(f"Initial test perplexity:\t{2 ** entropy}")


    print(f"- + - + - Running LDA - + - + - ")

    for iter in range(iterations):
        gibbs_sampling(documents, K, doc_topic_counts, topic_word_counts, topic_counts,
                       word_topic_assignments, alpha, gamma)
        print(f"\t{iter} iteration completed")

        cur_entropy = comp_topic_entropy(topic_word_counts, gamma)
        entropy_record.append(cur_entropy)

        print(f"\t{iter} iteration word entropy saved")

        # Optionally, log progress or visualize intermediate results here
        if iter in [0, 1, 4, 9, 19, 49, 99]:
            # print(f"Iteration {iter}: Gibbs sampling in progress")
            get_longest_document_plot(doc_topic_counts, iter+1)
            print(f"\t{iter} iteration freq words saved")
            # print(entropy_record)

        if iter == 50:
            test_doc_topic_counts, test_word_topic_assignments = new_doc_topic(test_docs, test_doc_topic_counts, topic_word_counts, topic_counts, test_word_topic_assignments, topics, gamma, alpha)
            entropy = comp_word_entropy(test_docs, topic_word_counts, test_doc_topic_counts, gamma, alpha)
            print(f"50 iteration test entropy:\t{entropy}")
            print(f"50 iteration test perplexity:\t{2 ** entropy}")

    freq_words(topic_word_counts, chosen_topics, dictionary)

    plot_entropy(entropy_record)

    return doc_topic_counts, topic_word_counts, topic_counts, word_topic_assignments


def get_longest_document_plot(doc_topic_counts, gibbs_iter):
    doc_total = doc_topic_counts.sum(axis=-1)
    longest_id, size = np.argmax(doc_total), np.max(doc_total)
    topic_distrib = doc_topic_counts[longest_id, :]

    # Creating histogram
    fig, axs = plt.subplots(1, 1,
                            figsize=(10, 7),
                            tight_layout=True)

    N, bins, patches = axs.hist(topic_distrib, bins=doc_topic_counts.shape[1], color="yellow", edgecolor="black")

    # Adding labels and title
    plt.xlabel('Topics')
    plt.ylabel('Word counts')
    plt.title(f'Distribution over topics - {gibbs_iter} iteration')
    i = 0
    for count, bin, patch in zip(N, bins, patches):
        plt.text(bin, count, f'{i}', ha='center', va='bottom')
        i += 1

    # Show plot
    plt.savefig(f'{plot_dir}/{str(gibbs_iter)}_long_distrib.png')
    # plt.show()


def comp_topic_entropy(topic_word_counts, gamma):
    vocab_size = topic_word_counts.shape[1]

    entropies = []
    for k_topic_words in topic_word_counts:
        topic_total = np.sum(k_topic_words)
        probs_record = []
        for w in k_topic_words:
            cond_prob = (gamma + w) / (gamma * vocab_size + topic_total)
            probs_record.append(cond_prob)

        probs_record = np.array(probs_record)
        entropy_k = - np.sum(probs_record * np.log2(probs_record))
        entropies.append(entropy_k)

    return entropies


def comp_word_entropy(test_docs, topic_word_counts, test_doc_topic_counts, gamma, alpha):
    vocab_size = topic_word_counts.shape[1]
    doc_size = test_doc_topic_counts.shape[0]
    k_size = topic_word_counts.shape[0]

    def d_w_prob(d, w):
        res = 0

        d_word_total = np.sum(test_doc_topic_counts[d, :])
        for k in range(k_size):
            k_vocab_total = topic_word_counts[k, :].sum()

            cond_prob = (gamma + topic_word_counts[k, w]) / (gamma * vocab_size + k_vocab_total)
            doc_prob = (alpha + test_doc_topic_counts[d, k]) / (k_size * alpha + d_word_total)
            res += cond_prob * doc_prob
        return res

    words_total = np.sum(np.sum(test_doc_topic_counts, axis=-1))
    entropy = 0
    for d, doc_words in enumerate(test_docs):
        for n, dw in enumerate(doc_words):
            dw_prob = abs(d_w_prob(d, dw))
            if dw_prob > 100:
                print(dw_prob)
            entropy += np.log2(dw_prob)

    entropy = - entropy / words_total

    return entropy


def freq_words(topic_word_counts, topics, vocab_dict, n=20):
    res = {}
    for k, k_topic_words in enumerate(topic_word_counts):
        if k in topics:
            ind = np.argpartition(k_topic_words, n)[-n:]
            top20_w = k_topic_words[ind]

            tokens = [vocab_dict[i] for i in ind]
            print(tokens)

            # Creating histogram
            fig, axs = plt.subplots(1, 1,
                                    figsize=(10, 7),
                                    tight_layout=True)

            N, bins, patches = axs.hist(top20_w, bins=20, color="orange",
                                        edgecolor="black")

            # Adding labels and title
            plt.xlabel('Words')
            plt.ylabel('Frequency')
            plt.title(f'Most freq words in {k} topic')
            i = 0
            for count, bin, patch in zip(N, bins, patches):
                plt.text(bin, count, f'{tokens[i]}', ha='center', va='bottom')
                i += 1

            # Show plot
            plt.savefig(f'{plot_dir}/{k}_freq_words.png')
            # plt.show()

            print(top20_w)

            topic_dict = {}
            for token, freq in zip(tokens, top20_w):
                topic_dict[token] = freq

            res[k] = topic_dict

    return res


def plot_entropy(entropy_record):
    entropy_values = np.array(entropy_record)
    # Create the figure and plot all 20 topics
    plt.figure(figsize=(10, 6))

    # Plot each topic's entropy values over iterations
    for topic in range(entropy_values.shape[1]):
        plt.plot(entropy_values[:, topic], label=f'Topic {topic + 1}')

    # Adding labels and title
    plt.xlabel('Gibbs sampling iteration')
    plt.ylabel('Entropy')
    plt.title('Entropy over vocabulary for 20 Topics')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), ncol=1, fontsize='small')

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/entropy_over_iterations.png", dpi=300)

    # plt.show()


# dk_out, vk_out, k_out = f"{temp_dir}/dk.csv", f"{temp_dir}/kv.csv", f"{temp_dir}/k.csv"
# dict_out = f"{temp_dir}/vocab.dict"

print("TRAIN")
train_docs, vocabulary = prepare_data(newsgroups_train, temp_dir, None, True)

print("TEST")
test_docs, _ = prepare_data(newsgroups_test, temp_dir, vocabulary, False)

# Run the LDA model
doc_topic_counts, topic_word_counts, topic_counts, word_topic_assignments = run_lda(train_docs, test_docs, vocabulary, temp_dir,
                                                                                    K=topics, alpha=0.1, gamma=0.1, iterations=100)
