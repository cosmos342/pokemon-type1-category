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
* In the input, removed type1 field. </br>
* Changed input in categorial columns to frequencies. Replaced null fields with null frequency. </br>
* Unit normalized the input(input-mean())/std(). </br>
* Changed the input from pandas to np array. changed input from float64 to float32 </br>
* From 800 input samples seperated 100 as test samples so that they are not exposed to the model while training </br>
* For training, kept train input at 630x10(10 fields) and output at 18(18 type1 category). </br>
* Converted 18 type1 category to 18 field np array(with one field set to 1 and rest to zero for each type). So output shape would become 630x18 for training </br>
* Initially only used one set of 70 samples as validation samples for all epochs. Here the training accuracy was greater than the
validation accuracy. </br>
* Divided training input to 630 training samples and 70 validation samples for each epoch and rotated it through each epoch so all the input samples are taken as training aswell as validation samples in different epochs. Here the validation accuracy went up compared to training accuracy </br>
* Initially tried 2 Fully connected layers with dropout of 0.5 after activation and trained. It gave good validation accuracy of around 60%. However test accuracy was only around 20% </br>
* Then removed the dropout layer and added batchnormalization before preactivation before each layer. This seemed to make a big difference as the training and validation accuracy went up to 90% within 40 epochs. And test accuracy also went up to 90%.
