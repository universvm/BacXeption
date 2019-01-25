<h1 align="center"><img src="/logo.png">

Deep Learning for bacterial classification</h1>

# BacXeption

BacXeption is a Deep Learning template of image segmentation functions and a 
Convolutional Neural Network (CNN) built on Keras for bacterial image 
classification. It uses the Xception architecture with pre-trained weights (https://arxiv.org/abs/1610.02357). 
## Examples

## 1. Getting Started
This project requires Python 3.6+
### 1.1 Pre-requisites

Install the prerequisites with PIP
```
pip install -r requirements.txt
```

### 1.2 Running the trained model

1. Place the raw images in `data/test_data/`
2. Run `python main.py`

This should output labelled images with a .txt file of the coordinates of 
each box in the `output/$DATE_TIME` folder. Example: 

<h1 align="center"><img src="https://i.imgur.com/8cChQ8s.png"></h1>


## 2. Training your own model

### 2.1 Two categories
1. Replace the images in the `data/0/` and `data/1/` with your 
images. 
2. Run `python train.py`
3. Move the `output/$DATE_TIME/model.json` and `output/$DATE_TIME/model.h5` 
in the `model/` folder.
4. Follow the instructions in section 1.2 

### 2.1 >Two categories
1. Change `NUM_CLASSES` in config.py to the number of classes wanted. 
2. Add your data in the `data/` folder. Each category should have a separate
 folder name, these must be integers starting from 0 (eg. `0/`,`1/`,`2/` for
  3 categories) 
3. Follow the instructions in section 2.1

## 3. Contributing
Pull requests and suggestions are always welcome. 

## 4. Additional information

### Authors
Leonardo Castorina - [universVM](https://github.com/universvm)

## Acknowledgments

[Dr. Teuta Pilizota](http://pilizotalab.bio.ed.ac.uk) - Proposing the 
problem and useful discussions. 

[Dario Miroli](https://github.com/DarioMiroli) – For introducing me to Keras
 and debugging early versions of BacXeption
 
[François Chollet](https://github.com/fchollet) – Developing Keras and 
Xception 

