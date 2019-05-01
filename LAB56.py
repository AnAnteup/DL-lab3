import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense,Dropout,Embedding,LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from tqdm import tqdm
# import nltk
# nltk.download('all')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
train= pd.read_csv("./train.tsv")
test = pd.read_csv("./test.tsv")
lemmatizer = WordNetLemmatizer()
def preprocessing_sentences(df):
    pre = []
    for sent in tqdm(df['Phrase']):
        # remove html content
        review_text = BeautifulSoup(sent).get_text()
        # remove non-alphabetic characters
        review_text = re.sub("[^a-zA-Z]", " ", review_text)
        # tokenize the sentences
        words = word_tokenize(review_text.lower())
        # lemmatize
        for i in words:
            lemma_words =lemmatizer.lemmatize(i)
        pre.append(lemma_words)
    return (pre)
train_sentences = preprocessing_sentences(train)
test_sentences = preprocessing_sentences(test)
print(len(train_sentences))
print(len(test_sentences))
target=train.Sentiment.values
y_target=to_categorical(target)
num_classes=y_target.shape[1]
X_train,X_val,y_train,y_val=train_test_split(train_sentences,y_target,test_size=0.2,stratify=y_target)

unique_words = set()
len_max = 0

for sent in tqdm(X_train):

    unique_words.update(sent)
    if (len_max < len(sent)):
        len_max = len(sent)

# length of the list of unique_words gives the no of unique words
print(len(list(unique_words)))
print(len_max)
tokenizer = Tokenizer(num_words=len(list(unique_words)))
tokenizer.fit_on_texts(list(X_train))
#Converts a text to a sequence of words
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(test_sentences)
#Pads sequences to the same length.
X_train = sequence.pad_sequences(X_train, maxlen=len_max)
X_val = sequence.pad_sequences(X_val, maxlen=len_max)
X_test = sequence.pad_sequences(X_test, maxlen=len_max)
print(X_train.shape,X_val.shape,X_test.shape)

model=Sequential()
model.add(Embedding(len(list(unique_words)),300,input_length=len_max))
model.add(LSTM(128,dropout=0.2, recurrent_dropout=0.3,return_sequences=True))
model.add(LSTM(64,dropout=0.2, recurrent_dropout=0.3,return_sequences=False))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.005),metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=4, batch_size=256, verbose=1)
score,acc = model.evaluate(X_val,y_val,verbose=2,batch_size=32)
print(score)
print(acc)