__author__ = 'sam'

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.engine.training import slice_X

from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
import argparse

parser = argparse.ArgumentParser(description='Keras pokemon type1 classification example')
parser.add_argument('--predict', type=int,
                    help='input number of predictions'
                    )
parser.add_argument('--train', type=int,
                    help='input number of iterations'
                    )
args = parser.parse_args()
num_predict = args.predict
if(num_predict and num_predict > 100):
    print("Limiting prediction to 100 samples")
    num_predict = 100

num_iteration = args.train


#encode/decode type 1 classes into categories.
class Type1Table(object):
    '''
    Given a set of classes, Enumerate them to indices and save reverse lookup (indices to class)
    '''
    def __init__(self,Type1classes):
        self.type1classes = sorted(set(Type1classes))
        self.type1to_indices = dict((type1, i) for i, type1 in enumerate(self.type1classes))
        self.indicesto_type1 = dict((i, type1) for i, type1 in enumerate(self.type1classes))

    def encode(self, C):
        X = np.zeros((len(self.type1classes)))
        X[self.type1to_indices[C]] = 1
        return X

    def decode(self, index):
        return self.indicesto_type1[index]

df = pd.read_csv("Pokemon.csv",keep_default_na=False,na_values=['SAM'])

#shuffle the dataframe as the type1s are not properly mixed.
df = df.reindex(np.random.permutation(df.index))

type1s = df['Type 1'].unique()
#encode and decode classes.
type1table = Type1Table(type1s)


TRAINING_SIZE = 700
BATCH_SIZE = 21
NB_CLASSES = len(type1s)
NB_EPOCH = 10



#Label generation
tmpY = df['Type 1'].tolist()
tmpYTest = tmpY[TRAINING_SIZE:]
y = np.zeros((len(tmpY), len(type1s)), dtype=np.int)
for i,value in enumerate(tmpY):
    y[i] =  type1table.encode(value)

#DROP COLUMNS THAT ARE NOT REQUIRED.
df.drop('Type 1', axis=1,inplace=True)
df.drop('#',axis=1,inplace=True)
df.drop('Name', axis=1, inplace=True)

# FUNCTION TO NORMALIZE ALL INPUT COLUMNS
def normalize_column(df,column):
    mean = pd.Series(df[column].mean())
    std = pd.Series(df[column].std())
    df[column] = df[column].apply(lambda x: (x-mean)/std)

# REPLACE CATEGORICAL INPUTS TO THEIR FREQUENCIES
def replace_column_to_frequency(valuefreq,column):
    df[column] = df[column].apply(lambda x: valuefreq[x])

#column list.
columns = list(df.columns.values)
#print(columns)

#GET FREQUENCIES OF CATEGORICAL COLUMNS
#type2 column
type2freq = {}
type2freq = df['Type 2'].value_counts().to_dict()

#Generation
gen2freq = {}
gen2freq = df['Generation'].value_counts().to_dict()

#Legendary
legendary2freq = {}
legendary2freq = df['Legendary'].value_counts().to_dict()

#replace categorical columns to frequencies of each class.
replace_column_to_frequency(type2freq,'Type 2')
replace_column_to_frequency(gen2freq,'Generation')
replace_column_to_frequency(legendary2freq,'Legendary')

#normalize column to unit normal.
for column in columns:
    normalize_column(df,column)


# Get the input vector CHANGE TO NUMPY INPUT.
X = df.values
X = X.astype('float32')

# Make the input vector test set and training set.
X_test = X[TRAINING_SIZE:]
X = X[:TRAINING_SIZE]
#print("INPUT SHAPE IS ",X.shape)

# Make the output vector test and training set.
y_test = y[TRAINING_SIZE:]
y = y[:TRAINING_SIZE]



class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


#Train the model
def Train(num_train):
    print('Build model...')
    model = Sequential()
    model.add(Dense(128,input_shape=X.shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(NB_CLASSES))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    # Train the model using adam optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    startVal = 0
    for iteration in range(1, num_train):
        print()
        print('-' * 50)

        #Loop through and change validation set in each epoch
        X_val = X[startVal:startVal+70]
        y_val = y[startVal:startVal+70]
        X_train = X[0:startVal]
        X_train_tail = X[startVal+70:700]
        if( X_train.size and X_train_tail.size):
            X_train= np.concatenate((X_train,X_train_tail), axis=0)
        elif not X_train.size:
            X_train = X_train_tail

        y_train = y[0:startVal]
        y_train_tail = y[startVal+70:700]
        if( y_train.size and y_train_tail.size):
            y_train= np.concatenate((y_train,y_train_tail), axis=0)
        elif not y_train.size:
            y_train = y_train_tail


        print('Iteration', iteration)

        #train for the batch
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH,
                  validation_data=(X_val, y_val))
        ###
        # Select 10 samples from the validation set at random so we can visualize errors
        for i in range(10):

            correct = tmpY[startVal+i]

            preds = model.predict_classes(X_val[np.array([i])], verbose=0)

            print(preds[0])
            guess = type1table.decode(preds[0])
            print('T', correct)
            print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)
            print('---')
            print("done")

        startVal += 70
        if(startVal == 700):
            startVal = 0
    #save the model so we can reuse for prediction
    model.save('my_model.h5')

#time to test/predict
def predict_type1s(predict):
    #note the model provided here is the one i trained.
    #you can change to the one you trained and saved.
    model = load_model('my_model_batch_normalization.h5')
    print("Loaded model from disk")
    count = 0

    for ind in range(predict):
        correct = tmpYTest[ind]

        # predict
        preds = model.predict_classes(X_test[np.array([ind])], verbose=0)

        #print(preds[0])
        guess = type1table.decode(preds[0])
        print('T', correct)
        if(correct == guess):
            count += 1
        print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)
        print('---')
        print("done")

    print("prediction accuracy is " ,count/predict)


if(num_predict):
    predict_type1s(num_predict)
else:
    Train(num_iteration)