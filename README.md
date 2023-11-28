# AI-Project: Laeeb (Group 10)
## Team Members
- 2018331115: Mizbah Uddin Junaed
- 2018331003: Shahrab Khan Sami
- 2018331027: Bipul Karmokar
- 2018331053: Saikat Hossain Rony

##  Description
A hand gesture recognition system provides a natural, innovative, and modern way of non-verbal communication. This is an Al system that will be given as input an image of a wrist in a particular gesture, it will identify it and output the name of the gesture.

## Overview
The model is trained first and then used to classify new hand gestures by inputting an image of the gesture and using the output of the final layer as features to train a classifier. This classifier can then be used to predict the class label of a new gesture. We used a popular CNN model named VGG-16.

For presenting a practical use of our project, we have given a demo of an application of the model. It is a calculator that uses different hand gestures to input the numbers and operators and then gives the output. It is demostrated in this video: <br>
[![Laeeb-demo](https://i9.ytimg.com/vi/eh76ANa9Blo/mqdefault.jpg?sqp=CNTRmZ8G-oaymwEmCMACELQB8quKqQMa8AEB-AHUBoAC4AOKAgwIABABGGUgZShlMA8=&rs=AOn4CLBOBS4YPqn5Y53aVybvE75Odyf79Q)](https://youtu.be/eh76ANa9Blo)

### Frontend: 
Clone this repository and run the command `streamlit run front.py`


## Dataset
[Kaggle: Hand Gesture Recognition](https://www.kaggle.com/datasets/aryarishabh/hand-gesture-recognition-dataset)
### To use dataset in google colab:
- Go to the 'Account' tab of your [Kaggle](https://www.kaggle.com) user profile `https://www.kaggle.com/<username>/account` and select 'Create API Token'
- Upload the `kaggle.json` file to colab's `/content` folder
- Uncomment the first cell and run all
