from sentiment import Sentiment
import csv
from string import ascii_letters, digits

# Preprocesses input by removing punctuation and splitting contractions (can't/cant -> can not). 
# Most importantly, applies the sentiment analyzer to pick out and record words that convey sentiment and determines the data's literal sentiment and implied sentiment.


analyzer = Sentiment()

def preprocess(text):
    global analyzer
    CONTRACTIONS = {"cant":"can not", "cannot":"can not", "dont":"do not", "doesnt":"does not", "didnt": "did not", "isnt": "is not", "arent":"are not", "wasnt": "was not", "werent":"were not", "wont":"will not", "shouldnt":"shouldnt", "couldnt":"could not", "wouldnt":"would not", "aint":"is not", "mustnt": "must not", "mightnt":"might not", "shant":"shall not", "hasnt":"has not", "havent":"have not", "hadnt":"had not"}
    LOOK_DIST = 4 # Number of words ahead of "not" to provide as context to determine if not is a sentiment word.
    text = text.strip().lower()
    text = "".join([char for char in text if char in ascii_letters + digits + " "]) # Remove punctuation and uninterpretable characters

    words = text.split(" ")
    output_text = []
    sentiment_words = []
    non_sentiment_words = []
    positivity = 0 # Determines literal sentiment by adding 1 for every positive word and subtracting 1 for every negative
    active_not = 0 # When > 0, there was a "not" recently flipping the sentiment of the next active_not words
    for i, word in enumerate(words):
        if word == '':
            continue
        expanded = CONTRACTIONS.get(word)  # If word is contraction involving not, replace word with expanded form.
        word = expanded if expanded is not None else word
        query = word
        if expanded is not None or word == "not": # If word is not or contraction involving not
            query = " ".join(["not"] + words[i + 1: min(i + LOOK_DIST + 1, len(words))]) # Form a query of "not" and the LOOK_DIST words that follow
        sentiment = analyzer.analyze([query]).get("sentiments") # Analyze query and retrieve sentiment result.
        if sentiment == ["positive"] or sentiment == ["negative"]:
            if expanded is not None: # Only include the "not" part of the contraction as a sentiment word
                sentiment_words.append("not")
            else:
                sentiment_words.append(word)
            
            if expanded or word == "not": # If "not" is the current word, toggle that not is active
                active_not = LOOK_DIST if active_not > 0 else 0 # "Not" will flip sentiment of next LOOK_DIST words, or "not" is cancelled if active (double negative)
            else:
                if (sentiment == ["positive"] and active_not <= 0) or (sentiment == ["negative"] and active_not > 0):
                    positivity += 1
                else:
                    positivity -= 1
                active_not -= 1
        else:
            non_sentiment_words.append(word)
        
        output_text.append(word)
    
    return " ".join(output_text), " ".join(sentiment_words), " ".join(non_sentiment_words), positivity


# Generate a new csv of preprocessed inputs with sentiment words recorded in the csv so they do not have to be found while training
def preprocess_csv(path, start_row=0):
    with open(path, 'r', errors='ignore') as input: 
        r = csv.reader(input)
        with open(path[:-4] + '_preprocessed.csv', 'w', errors='ignore', newline='') as output:
            w = csv.writer(output)
            for _ in range(start_row): # Skip rows before start
                next(r)

            for row in r:
                if len(row) == 0 or row[0] == "": # Skip rows with no data
                    continue
                text = row[0]
                sarcasm_label = row[1]
                new_text, sentiment_words, non_sentiment_words, positivity = preprocess(text)
                literal_label = int(positivity >= 0) # Literal label just reflects whether the text contains fewer negative words than positive
                if sarcasm_label == "1": # If text is sarcastic, implied label should be opposite of literal, otherwise equal
                    implied_label = int(positivity < 0)
                else:
                    implied_label = literal_label
                
                new_row = [new_text, sarcasm_label, sentiment_words, non_sentiment_words, literal_label, implied_label]
                w.writerow(new_row)

preprocess_csv("../data/train.csv", start_row=1) # These lines ran once to create train_preprocessed.csv and test_preprocessed.csv
preprocess_csv("../data/test.csv", start_row=1)
    