import numpy as np
import random
from collections import Counter, defaultdict
import time
import traceback

input_file = "data_small.txt"
gold_file = "data_small_gold.txt"

alpha = 100
Pc = 0.5
Pcont = 0.99
T = 1
iterations = 50

annealing = False

output_file = f"{iterations}i_{alpha}a_{str(Pc)}pc_{str(Pcont)}pcnt_{'ann_' if annealing else ''}data_out.txt"




with open(input_file, "r") as f:
    texts = f.readlines()

text = "".join(texts)
# text = text.replace("\n", "")

print(text)
characters = list(set(text))
print(characters)
N = len(text)
C = len(characters)

temperature_max = N * 1.5



def load_txt(file):
    spaces = [1]
    wc = 0

    with open(file) as f:
        for line in f:
            for word in line.strip().split(" "):
                if word == '':
                    continue
                for i in range(len(word) - 1):
                    spaces.append(0)
                spaces.append(1)
                wc += 1
    f.close()
    return spaces, wc


def eval(gold_spaces, gold_wc, test_spaces, test_wc):
    if len(test_spaces) != len(gold_spaces):
        print("WARNING: Different sizes of test and gold files: TEST:", len(test_spaces), "GOLD:", len(gold_spaces))

    begin_ok = 0
    correct_count = 0
    for i in range(len(gold_spaces)):
        if gold_spaces[i] == 1 and test_spaces[i] == 1:
            if begin_ok == 1:
                correct_count += 1
            begin_ok = 1
        elif gold_spaces[i] != test_spaces[i]:
            begin_ok = 0

    precision = correct_count / test_wc
    recall = correct_count / gold_wc
    # print(precision)
    # print(recall)
    f1 = 2 * precision * recall / (precision + recall)

    print("Precision:", precision, "Recall:", recall, "F1-score:", f1)


def random_permut(N):
    numbers = list(range(1, N - 1))
    random.shuffle(numbers)
    return numbers[:N]


def count_words(text, separators):
    words = []
    w = ""
    for i, char in enumerate(text):
        w += char
        if separators[i] == 1:
            words.append(w)
            w = ""
    counts = Counter(words)
    return counts, words


def word_base_prob(word):
    k = len(word)
    return (1/C)**k * Pc**(k-1) * (1 - Pc)


def word_neighbours(i, separators, text):
    cur = separators[i]
    prev_c, next_c = separators[i-1], separators[i+1]

    ps, pe, ns, ne = None, i, i+1, None

    if next_c == 1:
        ne = i+1

    if prev_c == 1:
        ps = i

    moved_n = 1
    moved_p = cur

    if not ps:
        while i - moved_p >= 0 and separators[i - moved_p] != 1:
            moved_p += 1
        ps = i - moved_p + 1

    if not ne:
        while i + moved_n != N and separators[i + moved_n] != 1:
            moved_n += 1
        ne = i + moved_n


    # print(prev_c, cur, next_c)
    # print(ps, pe, i, ns, ne)

    prev_word = text[ps:pe + 1]
    next_word = text[ns:ne + 1]

    # print(prev_word, next_word, separators[ps:pe+1], separators[ns:ne+1])
    # print(text[ps:ne+1], separators[ps:ne+1])
    # print(text[ps-2:ne + 3], separators[ps-2:ne + 3])
    # raise NotImplementedError

    return prev_word, next_word


def CRP(text, ann=False):
    separators = [1 if random.random() > 0.5 else 0 for c in text]

    w_counts, words = count_words(text, separators)
    word_counts = defaultdict(lambda: 0)
    word_counts.update(w_counts)

    st = " ".join(words)
    with open(output_file, "w") as file:
        file.write(st)

    t = sum(separators)
    # print(word_counts)
    print(len(word_counts.keys()), sum(word_counts.values()))

    s, wc = load_txt(output_file)
    gold_s, gold_wc = load_txt(gold_file)

    print(len(s), wc)
    print(len(gold_s), gold_wc)

    for iter in range(iterations):
        start_time = time.perf_counter()
        permuts = random_permut(N)
        temp = temperature_max
        for num, i in enumerate(permuts):
            # if num % 1000 == 0:
            #     print((time.perf_counter() - start_time) / 60, "minutes\n\n", num, N, t, "\n")
            prev_w, next_w = word_neighbours(i, separators, text)
            joined = prev_w + next_w

            if separators[i] == 0:
                if word_counts[joined] > 0:
                    w_counts[joined] -= 1
                    t -= 1
            else:
                word_counts[prev_w] -= 1
                word_counts[next_w] -= 1
                t -= 2

            w_0 = (word_counts[joined] + alpha * word_base_prob(joined)) / (alpha + t)
            w_1 = (word_counts[prev_w] + alpha * word_base_prob(prev_w)) / (alpha + t) * \
                  (word_counts[next_w] + alpha * word_base_prob(next_w)) / (alpha + 1 + t) * Pcont


            if ann:
                w_0 = w_0 ** (1 / temp)
                w_1 = w_1 ** (1 / temp)
                temp -= 1

            separators[i] = 1 if random.random() < (w_1 / (w_1 + w_0)) else 0
            # raise NotImplementedError

            # print(num, i, N, t, prev_w, next_w, joined, text[i], text[max(i-3, 0):min(i+3, N-1)], separators[i])

            if separators[i] == 0:
                word_counts[joined] += 1
                t += 1
            else:
                word_counts[prev_w] += 1
                word_counts[next_w] += 1
                t += 2

        end_time = time.perf_counter()

        elapsed_time = (end_time - start_time) / 60
        print(f"\n{iter} Iteration took {elapsed_time:.6f} minutes to execute\n")

        w_counts, words = count_words(text, separators)
        t = sum(separators)
        seg_text = " ".join(words)

        with open(output_file, "w") as file:
            file.write(seg_text)

        s, wc = load_txt(output_file)
        print(len(s), wc)
        try:
            eval(gold_s, gold_wc, s, wc)
        except Exception as e:
            print(e)
            traceback.print_exc()

    return text


seg_text = CRP(text, False)

with open(output_file, "w") as f:
    f.write(seg_text)
