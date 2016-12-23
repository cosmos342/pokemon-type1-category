# pokemon-type1-category
predict pokemon type1 category
Predicts the type of Pokemon from its stats 
This is a submission for Siraj's Pokemon Classifier Contest : https://www.youtube.com/watch?v=0xVqLJe9_CY 
The dataset is taken from Kaggle : https://www.kaggle.com/abcsds/pokemon
#Dependencies

Tensorflow (pip install t) 
NumPy (pip install numpy) 
Pandas (pip install pandas) 
keras(pip install keras)

#Demo

Run in terminal: 
For training
$ python pokemonclassification.py --train 40
For prediction
$ python pokemonclassification.py --predict 40

Results

The neural network built on Keras/Tensorflow is trained to an accuracy of over 90% in a few minutes on CPU. After training, the user can input the stats of a pokemon, and the model will predict its type. 

I thank Siraj Raval for the contest and video explanations, keras examples for code.
