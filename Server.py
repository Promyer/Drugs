#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn
import re
import _pickle
from nltk import word_tokenize
from flask import Flask
from flask import jsonify
from flask import request


# In[2]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


# In[3]:


file = open('Classifier.model', 'rb')
model = _pickle.load(file)
file.close()
file = open('one_hot_encoder.transformer', 'rb')
encoder = _pickle.load(file)
file.close()


# In[4]:


app = Flask(__name__)

def my_tokenizer(text):
    return list(filter(lambda x: (x != ',' and x != '.' and len(x) > 3), word_tokenize(text)))

@app.route("/drugs", methods=['POST'])
def asker():
    text = request.form['text']
    words = np.array(my_tokenizer(text)).reshape((-1,1))
    one_hot_text = encoder.transform(words)
    y_pred = np.array(model.predict(one_hot_text))
    words = words.reshape((1,-1))[0]
    answer = jsonify(list(words[y_pred == 1]))
    return answer
if __name__ == "__main__":
        app.run(port=8000)

