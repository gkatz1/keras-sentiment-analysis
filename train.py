import os
import sys
import pandas as pd
import numpy as np
import re
from sklearn.utils import resample
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

DATA_DIR = "data/"
PRINT_STATS = True
STRIP_SPECIAL_CHARS = re.compile("[^A-Za-z0-9 ]+")
MAX_REVIEW_LEN = 48

def train():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.tsv'), sep="\t")

    if PRINT_STATS:
        print train.head()['Phrase']
        print train.loc[train.SentenceId == 2]
        print train['Sentiment'].value_counts()

    train['clean_review'] = clean_sentences(train.Phrase.values)

    if PRINT_STATS:
        print train.head()['clean_review']


    train_2 = train[train['Sentiment']==2]
    train_1 = train[train['Sentiment']==1]
    train_3 = train[train['Sentiment']==3]
    train_4 = train[train['Sentiment']==4]
    train_5 = train[train['Sentiment']==0]
    train_2_sample = resample(train_2,replace=True,n_samples=75000,random_state=123)
    train_1_sample = resample(train_1,replace=True,n_samples=75000,random_state=123)
    train_3_sample = resample(train_3,replace=True,n_samples=75000,random_state=123)
    train_4_sample = resample(train_4,replace=True,n_samples=75000,random_state=123)
    train_5_sample = resample(train_5,replace=True,n_samples=75000,random_state=123)

    df_upsampled = pd.concat([train_2, train_1_sample,train_3_sample,train_4_sample,train_5_sample])

    if PRINT_STATS:
        print df_upsampled['Sentiment'].value_counts()
        print df_upsampled.head()


    from keras.utils import to_categorical
    X = df_upsampled['clean_review']
    #test_set = test['clean review']
    #Y = train['Sentiment']
    Y = to_categorical(df_upsampled['Sentiment'].values)
    if PRINT_STATS:
        print(Y)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, random_state=123)

    if PRINT_STATS:
        print(X_train.shape,Y_train.shape)
        print(X_val.shape,Y_val.shape)


    # ~~~~~ MODEL ~~~~~~~
    num_unique_word = 13728
    max_features = num_unique_word
    max_words = MAX_REVIEW_LEN
    batch_size = 128
    epochs = 3
    num_classes=5

    # tokenizing
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Dense, GRU, Embedding
    from tensorflow.python.keras.optimizers import Adam
    from tensorflow.python.keras.preprocessing.text import Tokenizer
    from tensorflow.python.keras.preprocessing.sequence import pad_sequences

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_val = tokenizer.texts_to_sequences(X_val)

    # padding
    from keras.preprocessing import sequence,text
    from keras.preprocessing.text import Tokenizer
    from keras.models import Sequential
    from keras.preprocessing.sequence import pad_sequences

    X_train = sequence.pad_sequences(X_train, maxlen=max_words)
    X_val = sequence.pad_sequences(X_val, maxlen=max_words)

    if PRINT_STATS:
        print(X_train.shape,X_val.shape)


    from keras.preprocessing import sequence,text
    from keras.preprocessing.text import Tokenizer
    from keras.models import Sequential
    from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D,Flatten,MaxPooling1D,GRU,SpatialDropout1D,Bidirectional
    from keras.callbacks import EarlyStopping
    from keras.utils import to_categorical
    from keras.losses import categorical_crossentropy
    from keras.optimizers import Adam
    from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
    import matplotlib.pyplot as plt

    model1=Sequential()
    model1.add(Embedding(max_features,100,mask_zero=True))

    model1.add(LSTM(64,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
    model1.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
    model1.add(Dense(num_classes,activation='softmax'))


    model1.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
    model1.summary()

    model1.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=epochs, batch_size=batch_size, verbose=1)


def clean_sentences(phrases):
    cleaned = []
    for phrase in phrases:
        phrase = phrase.lower().replace("<br />", " ")
        phrase = re.sub(STRIP_SPECIAL_CHARS, "", phrase.lower())
        cleaned.append(phrase)
    
    return np.array(cleaned)


def main():
    train()


if __name__ == '__main__':
    main()
