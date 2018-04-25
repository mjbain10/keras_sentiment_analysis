from numpy import array
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

model = load_model('sentiment_model.h5')

inputText = ['not good']

# tokenize input
t = Tokenizer()
t.fit_on_texts(inputText)
vocab_size = len(t.word_index) + 1

# integer encoding the input
encoded_input = t.texts_to_sequences(inputText)

print(encoded_input)

# pad input
padded_input = pad_sequences(encoded_input, maxlen=4, padding='post')

# predict
# prediction = model.predict_classes(padded_input)
# print(prediction)

# evaluate the model
labels = array([0])
loss, accuracy = model.evaluate(padded_input, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))
