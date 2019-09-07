import re
import nltk
import pandas as pd 
import numpy as np 
from sklearn import metrics
from nltk.corpus import stopwords
from keras.optimizers import Adam
from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Dropout,Embedding,LSTM

# data-cleaning, According to the keras library,modify the data format
def dataClean(raw): 

    
    sentence = raw.copy()
    # delete the number and punctuation
    sentence= re.sub('[^a-zA-Z]', ' ',sentence)
    # Lower-case
    sentence = sentence.lower()
    # split by spacing
    sentence = sentence.split()
    # delete the stop words
    # lemmatizer = WordNetLemmatizer()
    # sentence= [lemmatizer.lemmatize(w) for w in sentence if not w in set(stopwords.words('english'))]
    return (' '.join(sentence))

# read data
def readFile(filename,max = '0'):

    
    openfile=open(filename,'r',encoding="utf-8")
    lines = openfile.readlines()
    data = []
    for line in lines:
        line = line.replace('\n','')
        if line != '':
            temp = line.split('\t')
            data.append(temp)
            if temp[1] == max and max != '0':
                return data
    return data

#split the dataset
def dataSplit(doc,size = 0.2):

    # extract the input data
    doc = np.array(doc)
    x = doc[1:,2]
    X = []
    for i in range(len(x)):
        temp = dataClean(x[i])
        X.append(temp.split(' '))
    if size == 0:
        return X

    # extract the label datda    
    Y = doc[1:,3]    
    return train_test_split(X,Y,test_size=size,random_state= 1)

# Change the structure of data Y in order to match the LSTM model 
def YSplit(y):

    
    Y = []
    
    # five elements, each element maps a type of sentiment (0,1,2,3,4)
    form = [0,0,0,0,0]
    for i in range(len(y)):
        temp = form.copy()
        num = int(y[i])
        temp[num] = 1
        Y.append(temp)

    Y = np.array(Y)
    return Y

# the beginning of the experiment 
print("Start!")

# choose the file path, read data and split the data
doc_train = readFile("train.tsv")    
x_train,x_val,y_train,y_val = dataSplit(doc_train)
print("Loaded training data")

doc_test = readFile("test.tsv")    
x_test = dataSplit(doc_test,size = 0)
print("Loaded testing data")

# extract all the words and delete the repeating words
unique_words = set()
len_max = 0
for sent in x_train:
    
    unique_words.update(sent)
    if(len_max<len(sent)):
        len_max = len(sent)

print("Word List:", len(unique_words))
print("Maximum word:", len_max)

tokenizer = Tokenizer(num_words=len(list(unique_words)))
tokenizer.fit_on_texts(list(x_train))

#  Serialization
X_train = tokenizer.texts_to_sequences(x_train)
X_val = tokenizer.texts_to_sequences(x_val)
X_test = tokenizer.texts_to_sequences(x_test)

# transform to the np array
X_train = sequence.pad_sequences(X_train, maxlen=len_max)
X_val = sequence.pad_sequences(X_val, maxlen=len_max)
X_test = sequence.pad_sequences(X_test, maxlen=len_max)

print(X_train.shape,X_val.shape,X_test.shape)

Y_train = YSplit(y_train)
Y_val = YSplit(y_val)

earlyStop = EarlyStopping(min_delta = 0.001, mode = 'max', monitor='val_acc', patience = 2)
callback = [earlyStop]

# constitute the LSTM network
model=Sequential()
model.add(Embedding(len(list(unique_words)),300,input_length=len_max))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr =0.003,beta_1=0.8),metrics=['accuracy'])
model.summary()


history=model.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=10, batch_size=256, verbose=1, callbacks= callback)


y_pred=model.predict_classes(X_test)
sub_file = pd.DataFrame(y_pred)
# predict and save
# sub_file.to_csv('submission.csv',index=False)

# each epoch record
epoch_count = range(1, len(history.history['loss']) + 1)

print("Done!")

# figure
import matplotlib.pyplot as plt

plt.plot(epoch_count, history.history['loss'], 'r--')
plt.plot(epoch_count, history.history['val_loss'], 'b-')

plt.plot(epoch_count, history.history['acc'], 'gray')
plt.plot(epoch_count, history.history['val_acc'], 'green')

plt.legend(['Training Loss', 'Validation Loss','Training Acc', 'Validation Acc'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()