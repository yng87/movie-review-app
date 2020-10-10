from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import joblib
import sqlite3
import os
import numpy as np

from vectorizer import vect

app = Flask(__name__)

### Preparing the classifier
cur_dir = os.path.dirname(__file__)
clf = joblib.load(os.path.join(cur_dir, 'model', 'lr_online.joblib'))

db = os.path.join(cur_dir, 'reviews.sqlite')
if not os.path.isfile(db):
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute(
        "CREATE TABLE review_db"\
        " (review TEXT, sentiment INTEGER, date TEXT)"\
    )
    conn.commit()
    conn.close()

def classify(document):
    label = {0: "negative", 1: "positive"}
    X = vect.transform([document])
    y = clf.predict_proba(X)
    proba = np.max(y)
    pred_class = np.argmax(y)
    return label[pred_class], proba

def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y], classes=np.array([0,1]))

def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute(
        "INSERT INTO review_db"\
        " (review, sentiment, date) VALUES"\
        " (?, ?, DATETIME('now'))", (document, y)
    )
    conn.commit()
    conn.close()

### Flask

class ReviewForm(Form):
    moviereview = TextAreaField(
        '',
        validators=[
            validators.DataRequired(),
            validators.length(min=15)
        ]
    )

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        y, proba = classify(review)
        return render_template(
            'results.html',
             content=review,
             prediction=y,
             probability=round(proba*100, 2)
             )
    
    return render_template('reviewform.html', form=form)

@app.route('/thanks', methods=['Post'])
def feedback():
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']

    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    train(review, y)
    sqlite_entry(db, review, y)

    return render_template('thanks.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))