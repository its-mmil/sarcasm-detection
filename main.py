import argparse
import csv
import losses
import random
import numpy as np
import tensorflow as tf

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

def train(args, model, all_input, lit_input, imp_input, lit_labels, imp_labels, sarc_labels):
    # Make sure inputs and labels are all the same length
    assert len(all_input) == len(lit_input) and len(imp_input) == len(lit_labels) and len(imp_labels) == len(sarc_labels) and len(sarc_labels) == len(all_input)
    
    batch_size = args.batch_size
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    input_len = len(all_input)

    # Choose random input order
    all_data = list(zip(all_input, lit_input, imp_input, lit_labels, imp_labels, sarc_labels))
    random.shuffle(all_data)
    all_input = [row[0] for row in all_data]
    lit_input = [row[1] for row in all_data]
    imp_input = [row[2] for row in all_data]
    lit_labels = [row[3] for row in all_data]
    imp_labels = [row[4] for row in all_data]
    sarc_labels = [row[5] for row in all_data]

    first_ind = 0 # 1st index of batch data
    last_ind = batch_size # index after last index of batch data
    accuracies = []
    while first_ind < input_len:
        if last_ind > input_len: # If taking a full batch would index out of bounds, end at last index
            last_ind = input_len
        # Grab a batch of data for each input and label
        all_input_b = all_input[first_ind : last_ind]
        lit_input_b = lit_input[first_ind : last_ind]
        imp_input_b = imp_input[first_ind : last_ind]
        lit_labels_b = lit_labels[first_ind : last_ind]
        imp_labels_b = imp_labels[first_ind : last_ind]
        sarc_labels_b = sarc_labels[first_ind : last_ind]
        
        # with tf.GradientTape() as tape:
        #     lit_pred, imp_pred, sarc_pred = model.call(all_input_b, lit_input_b, imp_input_b)
        #     loss = losses.total_loss(lit_labels_b, lit_pred, imp_labels_b, imp_pred, sarc_labels_b, sarc_pred)
        # gradients = tape.gradient(loss, model.trainable_variables)
        # optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # TODO: some accuracy function to write. append to accuracy list, prob need 3 separate for each part

        first_ind += batch_size
        last_ind += batch_size
    # TODO: print average accuracies for epoch

def test(model, all_input, lit_input, imp_input, lit_labels, imp_labels, sarc_labels):
    #lit_pred, imp_pred, sarc_pred = model.call(all_input_b, lit_input_b, imp_input_b)
    pass # TODO: calculate and return accuracy
    
def main(args):
    train_file = "data/train_preprocessed.csv"
    test_file = "data/test_preprocessed.csv"
    train_text, train_sarc_labels, train_sent_words, train_non_sent_words, train_lit_labels, train_imp_labels = get_data(train_file)
    test_text, test_sarc_labels, test_sent_words, test_non_sent_words, test_lit_labels, test_imp_labels = get_data(test_file)

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

    # Train model
    for _ in range(args.num_epochs):
        train(args, None, train_text, train_sent_words, train_non_sent_words, train_lit_labels, train_imp_labels, train_sarc_labels) #TODO: input model
    # Test model
    test(None, train_text, train_sent_words, train_non_sent_words, train_lit_labels, train_imp_labels, train_sarc_labels) #TODO: input model
if __name__ == "__main__": # Taken from HW6
    args = parseArguments()
    main(args)