from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding

# =======================
# INPUT DATA
# =======================


docs = ['not good']

# define class labels (pos = 1, neg = 0)
labels = array([0])

# tokenize and integer encode input
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)

# pad input
padded_docs = pad_sequences(encoded_docs, maxlen=4, padding='post')

# =======================
# GLOVE AND MODEL SET UP
# =======================

# load glOve embedding to memory
embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# create matrix of one embedding for each word in the training dataset
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# define model
model = Sequential()
e = Embedding(vocab_size, 100, weights=[
              embedding_matrix], input_length=4, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# summarize the model
print(model.summary())

# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)

# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

# save model
model.save('sentiment_model.h5')
