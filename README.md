# pokemon-type1-category </br>
predict pokemon type1 category </br>
Predicts the type of Pokemon from its stats </br>
This is a submission for Siraj's Pokemon Classifier Contest : https://www.youtube.com/watch?v=0xVqLJe9_CY </br>
The dataset is taken from Kaggle : https://www.kaggle.com/abcsds/pokemon 

#Dependencies

Tensorflow (pip install tensorflow) </br>
NumPy (pip install numpy) </br>
Pandas (pip install pandas) </br>
keras(pip install keras) </br>

#Demo

Run in terminal:  </br>
For training </br>
$ python pokemonclassification.py --train 40 </br>
For prediction </br>
$ python pokemonclassification.py --predict 40 </br>

Results

The neural network built on Keras/Tensorflow is trained to an accuracy of over 90% in a few minutes on CPU. After training, the user can input the stats of a pokemon, and the model will predict its type. Comes with a pretrained model. During prediction model predicts based on a pretrained model which is trained over 175 batch iterations on CPU.

#Credit

Siraj Raval for the challenge, keras examples for initial code.
