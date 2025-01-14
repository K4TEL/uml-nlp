#!/usr/bin/env python3

import sys

gold_spaces = [1]
gold_wc = 0

with open(sys.argv[1]) as f:
    for line in f:
        for word in line.strip().split(" "):
            if word == '':
                continue
            for i in range(len(word) - 1):
                gold_spaces.append(0)
            gold_spaces.append(1)
            gold_wc += 1
f.close()

test_spaces = [1]
test_wc = 0

with open(sys.argv[2]) as f:
    for line in f:
        for word in line.strip().split(" "):
            print(word)
            if word == '':
                continue
            for i in range(len(word) - 1):
                test_spaces.append(0)
            test_spaces.append(1)
            test_wc += 1
f.close()

#print(gold_spaces)
print(test_spaces)

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
print(precision)
print(recall)
f1 = 2 * precision * recall / (precision + recall)

print("Precision:", precision, "Recall:", recall, "F1-score:", f1)
