from keras.preprocessing.text import Tokenizer,one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

from enum import Enum
class BIO(Enum):
    O   = 0
    B_H = 1
    I_H = 2
    B_T = 3
    I_T = 4
    B_O = 5
    I_O = 6

file=open('data/dse_text.txt','r')
texts=file.readlines()

tokenizer=Tokenizer(num_words=None,lower=True)
tokenizer.fit_on_texts(texts)
sequences=tokenizer.texts_to_sequences(texts)

print(len(sequences)) #6936
word_index=tokenizer.word_index
print("found %s unique tokens"%len(word_index)) # 13547

data = pad_sequences(sequences)


print('shape of data tensor: ', data.shape)
#print('shape of label tensor: ',labels.shape)

#split the data into training set and a test set, a dev test
indices=np.arange(data.shape[0])
print('indices -->',indices)
print('features -->',data.shape[1])

data=data[indices]
nb_test_samples=int(0.15 * data.shape[0])
nb_train_samples= int(0.7 * data.shape[0])
print('training set size -->',nb_train_samples)
print('testing set size -->', nb_test_samples)

x_train= data[:nb_train_samples]
x_test= data[nb_train_samples+1:nb_train_samples+nb_test_samples]
x_dev= data[-nb_test_samples:]

print('x_train shape: ',x_train.shape)
print('x_test shape: ',x_test.shape)

train_labels= to_categorical(np.random.randint(1,7,size=(6629,1)))
test_labels= to_categorical(np.random.randint(1,7,size=(1419,1)))

print(train_labels)
##########how to deal with bio tags?#####################33333333
batch_size=64

###########embedding layer
from gensim.models.keyedvectors import KeyedVectors
from keras.models import Sequential
from keras.layers import Dense, LSTM

word_vectors=KeyedVectors.load_word2vec_format('../jointExtraction/GoogleNews-vectors-negative300.bin',binary=True)
embedding= word_vectors.get_keras_embedding()
# return a keras embedding layer with weights set as the word2vec models learned word embeddings
model=Sequential()
model.add(embedding)
model.add(LSTM(batch_size=64,units=6629,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(7, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,train_labels,batch_size=batch_size,epochs=10)
score,acc=model.evaluate(x_test,test_labels,batch_size=batch_size)
print('TEst score:',score)
print('Test accuracy: ',acc)