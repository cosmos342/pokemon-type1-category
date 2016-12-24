
/

# Pokemon type1 classification </br>
Predict pokemon type1 category </br>
Predicts the type of Pokemon from its stats </br>
This is for Siraj's Pokemon Classifier Contest : https://www.youtube.com/watch?v=0xVqLJe9_CY </br>
The dataset is taken from Kaggle. Also provided here as Pokemon.csv : https://www.kaggle.com/abcsds/pokemon 

#Dependencies

Tensorflow (pip install tensorflow) </br>
NumPy (pip install numpy) </br>
Pandas (pip install pandas) </br>
keras(pip install keras) (set the tensorflow backend in ~/.keras/keras.json file)</br>

#Demo

Run in terminal:  </br>
**For training** </br>
$ python pokemonclassification.py --train 40 </br>
**For prediction** </br>
$ python pokemonclassification.py --predict 40 </br>

#Results

The neural network built on Keras/Tensorflow is trained to an accuracy of over 90% in a few minutes on CPU. After training,user can run prediction. For prediction, model predicts by loading  a pretrained and saved model which is trained over 175 epochs on CPU. 

#Credit

Siraj Raval for the challenge, keras examples for initial code.

#Notes
In the input, removed type1 field. </br>
Changed input in categorial columns to frequencies. Replaced null fields with null frequency. </br>
Unit normalized the input(input-mean())/std(). </br>
Changed the input from pandas to np array. </br>
From 800 input samples seperated 100 as test samples so that they are not exposed to the model while training </br>

