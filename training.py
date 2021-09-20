import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent['tag']))

        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])




words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


# initializing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")





# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")









"""
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

ignore_letters = ['?',',','.','!']

words = list()
classes = list()
documents = list()

for intent in intents["intents"]:
	for pattern in intent["patterns"]:
		word_list = nltk.word_tokenize(pattern)
		words.extend(word_list)
		documents.append((word_list,intent["tag"]))
		if intent["tag"] not in classes:
			classes.append(intent["tag"])


words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words,open("words.pkl",'wb'))
pickle.dump(classes,open("classes.pkl",'wb'))


training = list()

output_empty =[0]*len(classes)

for document in documents:
	bag = list()
	word_patterns = document[0]
	word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
	for word in words:
		if word in word_patterns:
			bag.append(1)
		else:
			bag.append(0)
	output_row = list(output_empty)
	output_row[classes.index(document[1])]=1
	training.append([bag,output_row])

random.shuffle(training)
training = np.array(training,dtype="object")

x_train = list(training[:,0])
y_train = list(training[:,1])

model = Sequential()
model.add(Dense(128,input_shape=(len(x_train[0]),),activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]),activation="softmax"))


sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov = True)

model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics= ["accuracy"])
hist = model.fit(np.array(x_train),np.array(y_train),epochs= 200,batch_size=5,verbose=1)
model.save("chatbotmodel.h5",hist)
print("Done")
"""