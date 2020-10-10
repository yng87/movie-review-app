from sklearn.feature_extraction.text import HashingVectorizer
import pickle
import os
import re

cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(os.path.join(cur_dir, 'model', 'lr_online_stopwords.pkl'), 'rb'))

def tokenizer(text):
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall("(?::|;|-)(?:-)?(?:\)|\(|D|P)", text)
    text = (re.sub("[\W]+", " ", text.lower()) + " ".join(emoticons).replace("-", ""))
    tokenized = [w for w in text.split() if not w in stop]
    return tokenized

vect = HashingVectorizer(
    decode_error="ignore",
    n_features=2**21,
    preprocessor=None,
    tokenizer=tokenizer,
)