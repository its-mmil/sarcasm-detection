import argparse
import csv
# import numpy as np
# import tensorflow as tf

def parseArguments(): # Taken from HW6
    parser = argparse.ArgumentParser()
    #parser.add_argument("--test", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()
    return args

def get_data(path):
    with open(path, 'r') as file:
        r = csv.reader(file)
        all_text, sarcasm_labels, all_sentiment_words, all_non_sentiment_words, literal_labels, implied_labels = [], [], [], [], [], []
        for row in r:
            text = row[0].split(" ")
            all_text.append(text)
            sarcasm_label = int(row[1])
            sarcasm_labels.append(sarcasm_label)
            sentiment_words = row[2].split(" ")
            all_sentiment_words.append(sentiment_words)
            non_sentiment_words = row[3].split(" ")
            all_non_sentiment_words.append(non_sentiment_words)
            literal_label = int(row[4])
            literal_labels.append(literal_label)
            implied_label = int(row[5])
            implied_labels.append(implied_label)
        return all_text, sarcasm_labels, all_sentiment_words, all_non_sentiment_words, literal_labels, implied_labels

def main(args):
    train_file = "data/train_preprocessed.csv"
    test_file = "data/test_preprocessed.csv"
    train_text, train_sarcasm_labels, train_sent_words, train_non_sent_words, train_lit_labels, train_imp_labels = get_data(train_file)
    test_text, test_sarcasm_labels, test_sent_words, test_non_sent_words, test_lit_labels, test_imp_labels = get_data(test_file)

    # Build vocab:
    all_words = []
    for text in train_text:
        all_words += text
    for text in test_text:
        all_words += text
    all_words = sorted(set(all_words)) # Inspired by HW4
    vocabulary = {word: id for id, word in enumerate(all_words)}
    for to_translate in [train_text, test_text, train_sent_words, test_sent_words, train_non_sent_words, test_non_sent_words]: # Apply to all words in dataset
        to_translate[:] = [[vocabulary[x] for x in row if x != ""] for row in to_translate] # Convert all words into their ids in the vocab

if __name__ == "__main__": # Taken from HW6
    args = parseArguments()
    main(args)