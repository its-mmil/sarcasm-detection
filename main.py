import argparse
import csv
import losses
import random
import numpy as np
import tensorflow as tf
from model import Model

def parseArguments(): # Taken from HW6
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.003)
    parser.add_argument("--input_width", type=int, default=140)
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
    lit_accs, imp_accs, sarc_accs = [], [], []
    while first_ind < input_len:
        if last_ind > input_len: # If taking a full batch would index out of bounds, end at last index
            last_ind = input_len
        # Grab a batch of data for each input and label
        all_input_b = tf.convert_to_tensor(all_input[first_ind : last_ind])
        lit_input_b = tf.convert_to_tensor(lit_input[first_ind : last_ind])
        imp_input_b = tf.convert_to_tensor(imp_input[first_ind : last_ind])
        lit_labels_b = tf.convert_to_tensor(lit_labels[first_ind : last_ind])
        imp_labels_b = tf.convert_to_tensor(imp_labels[first_ind : last_ind])
        sarc_labels_b = tf.convert_to_tensor(sarc_labels[first_ind : last_ind])
        
        with tf.GradientTape() as tape:
            lit_pred, imp_pred, sarc_pred = model.call((lit_input_b, imp_input_b, all_input_b))
            loss = losses.total_loss(lit_labels_b, lit_pred, imp_labels_b, imp_pred, sarc_labels_b, sarc_pred, loss_coefs=[.25,.25,.5])
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Compute accuracies for each component
        lit_acc = binary_accuracy(lit_labels_b, lit_pred)
        imp_acc = binary_accuracy(imp_labels_b, imp_pred)
        sarc_acc = binary_accuracy(sarc_labels_b, sarc_pred)
        lit_accs.append(lit_acc)
        imp_accs.append(imp_acc)
        sarc_accs.append(sarc_acc)

        first_ind += batch_size
        last_ind += batch_size
    
    avg_lit_acc = np.mean(lit_accs)
    print(f"Literal Channel Accuracy: {avg_lit_acc}")
    avg_imp_acc = np.mean(imp_accs)
    print(f"Implied Channel Accuracy: {avg_imp_acc}")
    avg_sarc_acc = np.mean(sarc_accs)
    print(f"Overall Sarcasm Accuracy: {avg_sarc_acc}")

def test(model, all_input, lit_input, imp_input, lit_labels, imp_labels, sarc_labels):
    lit_pred, imp_pred, sarc_pred = model.call((tf.convert_to_tensor(lit_input), tf.convert_to_tensor(imp_input), tf.convert_to_tensor(all_input)), training=False)
    lit_acc = binary_accuracy(tf.convert_to_tensor(lit_labels), lit_pred)
    imp_acc = binary_accuracy(tf.convert_to_tensor(imp_labels), imp_pred)
    sarc_acc = binary_accuracy(tf.convert_to_tensor(sarc_labels), sarc_pred)
    print("Testing:")
    print(f"Literal Channel Accuracy: {lit_acc}")
    print(f"Implied Channel Accuracy: {imp_acc}")
    print(f"Overall Sarcasm Accuracy: {sarc_acc}")

    
# Taken from HW3
def binary_accuracy(y_true, y_pred):
    correct_predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


# def custom_loss(y_true, y_pred): # Custom intermediary loss function to satisfy keras's compile requirements
#     lit_labels = y_true[0]
#     imp_labels = y_true[1]
#     sarc_labels = y_true[2]
#     lit_pred = y_pred[0]
#     imp_pred = y_pred[1]
#     sarc_pred = y_pred[2]
#     return losses.total_loss(lit_labels, lit_pred, imp_labels, imp_pred, sarc_labels, sarc_pred)

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
    vocabulary = {word: id + 1 for id, word in enumerate(all_words)} # Assign ids, leaving 0 unassigned
    for to_translate in [train_text, test_text, train_sent_words, test_sent_words, train_non_sent_words, test_non_sent_words]: # Apply to all words in dataset
        to_translate[:] = [[vocabulary[x] for x in row if x != ""] for row in to_translate] # Convert all words into their ids in the vocab
        to_translate[:] = [row + [0] * (args.input_width - len(row)) for row in to_translate] # Pad word counts to be input_width words (140 by default, the maximum a tweet can have)

    # One-hot encode all labels:
    one_hotted = [tf.one_hot(x, 2).numpy().tolist() for x in (train_lit_labels, train_imp_labels, train_sarc_labels, test_lit_labels, test_imp_labels, test_sarc_labels)]
    train_lit_labels, train_imp_labels, train_sarc_labels, test_lit_labels, test_imp_labels, test_sarc_labels = one_hotted

    model = Model(vocab_size=len(vocabulary) + 1)
    #model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss = custom_loss, metrics = tf.keras.metrics.BinaryAccuracy())
    #model.fit(x=(train_sent_words, train_non_sent_words, train_text), y=(train_lit_labels, train_imp_labels, train_sarc_labels), epochs=args.num_epochs, batch_size=args.batch_size)
    #model.fit(x=all_inputs, y=all_inputs, epochs=args.num_epochs, batch_size=args.batch_size)

    # Train model
    for epoch in range(1, args.num_epochs + 1):
        print(f"Epoch {epoch}:")
        train(args, model, train_text, train_sent_words, train_non_sent_words, train_lit_labels, train_imp_labels, train_sarc_labels)
    # Test model
    test(model, test_text, test_sent_words, test_non_sent_words, test_lit_labels, test_imp_labels, test_sarc_labels)

if __name__ == "__main__": # Taken from HW6
    args = parseArguments()
    main(args)