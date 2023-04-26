from sentiment import Sentiment
import csv
from string import ascii_letters, digits

# Preprocesses input by removing punctuation and splitting contractions (can't/cant -> can not). 
# Most importantly, applies the sentiment analyzer to pick out and record words that convey sentiment.


analyzer = Sentiment()

def preprocess(text):
    global analyzer
    CONTRACTIONS = {"cant":"can not", "cannot":"can not", "dont":"do not", "doesnt":"does not", "didnt": "did not", "isnt": "is not", "arent":"are not", "wasnt": "was not", "werent":"were not", "wont":"will not", "shouldnt":"shouldnt", "couldnt":"could not", "wouldnt":"would not", "aint":"is not", "mustnt": "must not", "mightnt":"might not", "shant":"shall not", "hasnt":"has not", "havent":"have not", "hadnt":"had not"}
    LOOK_DIST = 4 # Number of words ahead of "not" to provide as context to determine if not is a sentiment word.
    #analyzer = Sentiment()
    text = text.strip().lower()
    text = "".join([char for char in text if char in ascii_letters + digits + " "]) # Remove punctuation and uninterpretable characters

    words = text.split(" ")
    output_text = []
    sentiment_words = []
    for i, word in enumerate(words):
        if word == '':
            continue
        expanded = CONTRACTIONS.get(word)#word.replace("'", ""))  # If word is contraction involving not, replace word with expanded form.
        word = expanded if expanded is not None else word
        query = word
        if expanded is not None or word == "not": # If word is not or contraction involving not
            query = " ".join(["not"] + words[i + 1: min(i + LOOK_DIST + 1, len(words))]) # Form a query of "not" and the LOOK_DIST words that follow
        sentiment = analyzer.analyze([query]).get("sentiments") # Analyze query and retrieve sentiment result.
        if sentiment is None:
            print(words)
            raise Exception
        if sentiment == ["positive"] or sentiment == ["negative"]:
            if expanded is not None: # Only include the "not" part of the contraction as a sentiment word
                sentiment_words.append("not")
            else:
                sentiment_words.append(word)
        output_text.append(word)
    
    return " ".join(output_text), " ".join(sentiment_words)


# Generate a new csv of preprocessed inputs with sentiment words recorded in the csv so they do not have to be found while training
def preprocess_csv(path, start_row=0):
    with open(path, 'r', errors='ignore') as input: 
        r = csv.reader(input)
        with open(path[:-4] + '_preprocessed.csv', 'w', errors='ignore', newline='') as output:
            w = csv.writer(output)
            for _ in range(start_row): # Skip rows before start
                next(r)

            for row in r:
                if len(row) == 0:
                    continue
                text = row[0]
                new_text, sentiment_words = preprocess(text)
                new_row = [new_text, row[1], sentiment_words]
                w.writerow(new_row)

preprocess_csv("../data/train.csv", start_row=1) #These lines ran once to create train_preprocessed.csv and test_preprocessed.csv
preprocess_csv("../data/test.csv", start_row=1)
    