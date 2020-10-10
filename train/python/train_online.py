import numpy as np
import re
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import pyprind
import joblib
import pickle

stop = stopwords.words("english")

def tokenizer(text):
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall("(?::|;|-)(?:-)?(?:\)|\(|D|P)", text)
    text = (re.sub("[\W]+", " ", text.lower()) + " ".join(emoticons).replace("-", ""))
    tokenized = [w for w in text.split() if not w in stop]
    return tokenized

def stream_docs(path):
    with open(path, "r", encoding="utf-8") as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)

    except StopIteration:
        return None, None
    
    return docs, y

def train(
    movie_data_path,
    batch_size,
    num_iter,
    random_seed,
    test_size,
    save_dir,
    model_name,
    ):
    pbar = pyprind.ProgBar(num_iter)

    classes = np.array([0,1])

    vect = HashingVectorizer(
        decode_error="ignore",
        n_features=2**21,
        preprocessor=None,
        tokenizer=tokenizer,
    )
    clf = SGDClassifier(loss="log", random_state=random_seed)

    doc_stream = stream_docs(movie_data_path)

    print("Start train...")
    for _ in range(num_iter):
        X_train, y_train = get_minibatch(doc_stream, batch_size)
        if not X_train:
            break
        X_train = vect.transform(X_train)
        clf.partial_fit(X_train, y_train, classes=classes)
        pbar.update()
    print("Train finish")

    print("Evaluate on test set.")
    X_test, y_test = get_minibatch(doc_stream, test_size)
    X_test = vect.transform(X_test)
    test_score = clf.score(X_test, y_test)
    print(f"Test Accuracy: {test_score}")

    print("Update by test set.")
    clf.partial_fit(X_test, y_test)
    save_path = os.path.join(save_dir, model_name + ".joblib")
    print(f"Save model to {save_path}")
    joblib.dump(clf, save_path)

    save_path = os.path.join(save_dir, model_name + "_stopwords.pkl")
    print(f"Save stop words to {save_path}")
    pickle.dump(stop, open(save_path, "wb"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--movie_data_path", type=str, default="./csv/movie_data.csv")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--num_iter", type=int, default=45)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--test_size", type=int, default=5000)
    parser.add_argument("--save_dir", type=str, default="./model")
    parser.add_argument("--model_name", type=str, required=True)

    args = parser.parse_args()
    args_dict = vars(args)

    train(**args_dict)

