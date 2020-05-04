
# Identification of COVID-19 patients from chest X-ray images
In this assignment, deep neural networks were used to identify Covid-19 patients from their chest X-ray image. Transfer learning was exploited to decrease the training time and increase the network peformance. VGG-16 and ResNet-18 networks, pre-trained on ImageNet Dataset, were fine tuned for this puropose. 
# Part-A: Classification of Normal Vs Infected Classes
In this task, we'll classify images to only two classes normal or infected. The code for this part is available in `covid19_classification.ipynb`
## Data
The dataset used in Part-A of this assignment can be found at this [Gdrive Link](https://drive.google.com/a/itu.edu.pk/uc?id=1-HQQciKYfwAO3oH7ci6zhg45DduvkpnK)
The images were downscaled to 256 by 256 pixels using *LANCZOS* interpolation
## Task 1 (Fine tuning of only the last FC layers)
In this task, I removed the last FC layers of both VGG and ResNet network and added two FC layers. First FC layer had 190 units while the second had 2 units.
### VGG Network
**Test accuracy = 92%**

Confusion Matrix:

|          |          |         |
|----------|----------|---------|
| Infected | 5.2e+02  | 98      |
| Normal   | 17       | 8.7e+02 |
|          | infected | Normal  |

Weights of this model are at [this link](https://drive.google.com/open?id=1-4dezrry054UffxyhWqN7Gcg3gVwgU6T)
### ResNet Network
**Test accuracy = 92%**

Confusion Matrix:

|          |          |         |
|----------|----------|---------|
| Infected | 5.2e+02  | 98      |
| Normal   | 17       | 8.7e+02 |
|          | infected | Normal  |
#### Weights
Weights of this model are at [this link](https://drive.google.com/open?id=1-8j4bm5yeOzNVsrDBqgf4k5QjiJqgGJe)

## Task 2
These models were fine tuned under different configurations. At first, only last conv layer was fine tuned, then last 3 conv layers were fine tuned and finally all the network was fine tuned.
Following are the results, confusion matrices and weights when all the models were fine tuned
### VGG Network
**Test accuracy = 97%**

Confusion Matrix:

|          |          |         |
|----------|----------|---------|
| Infected | 6e+02  | 18      |
| Normal   | 23       | 8.6e+02 |
|          | infected | Normal  |
#### Weights

Weights of this model are at [this link](https://drive.google.com/open?id=1uRpzAqoQolGbNYJtm1r2yVXPKCSRTACd)
### ResNet Network
**Test accuracy = 96%**

Confusion Matrix:

|          |          |         |
|----------|----------|---------|
| Infected | 5.8e+02  | 37      |
| Normal   | 19       | 8.7e+02 |
|          | infected | Normal  |
#### Weights
Weights of this model are at [this link](https://drive.google.com/open?id=1-1Xfgw9EVeRjkRRA6In07UbE7DfW1rel)

# Part B: Classification of X-ray image into Normal,Covid and Pneumonia Classes
In this experiment, we'll classify an X-ray image into normal, pneumonia or covid. Covid being a special case of pneumonia so any image belonging to this class will carry the label of both covid and pneumonia. In this way, this is a multilabel multicalss classification problem. We'll leverage the pre-trained VGG-16 and ResNet-18 network trained on ImageNet dataset. The dataset used for training has a high class imbalance, so we'll also alpha balanced [Focal Loss](https://arxiv.org/pdf/1708.02002.pdf) besides Binary Cross Entropy Loss. The code for this part is available in `covid19_classification_focal_loss.ipynb`
## Dataset
Dataset consists of labeled train and validation set while unlabeled test set.
Train and validation has 3 classes: Normal, Covid and pneumonia. 
The distribution is as follows:

|       | Normal | Covid | Pneumonia |
|-------|--------|-------|-----------|
| Train | 4000   | 200   | 2000      |
| Valid | 400    | 28    | 200       |

The dataset used in Part-B of this assignment can be found at this [Gdrive Link](https://drive.google.com/open?id=1eytbwaLQBv12psV8I-aMkIli9N3bf8nO&authuser=1)

## VGG-16 Model
I used the pre-defined pytorch model of VGG-16 and removed all of its (last) FC layers. I performed different experiments for various hyper-parameters and I'll report only the best ones here.
### Binary Cross Entropy Loss
I replaced the last FC layers with 3 FC layer. First layer having 200 neuron, second 60 and third 3. First and second FC layer had ReLU activation function followed by Dropout layer with drop probability of 0.5. 
I fine tuned the network for 20 epochs with batch size of 64. Stochastic Gradient descent with momentum of 0.9 and learning rate of 0.001 was used.

**Accuracy on Validation Set = 95.2 %**

**F1 score on Validation Set = 93.0 %**

#### Confusion Matrix
Following are confusion matrices of each class for validation set.

Covid Class

|          |          |         |
|----------|----------|---------|
| Negative | 6e+02  | 21      |
| Positive   | 0       | 7 |
|          | Negative  | Positive  |


Pneumonia Class

|          |          |         |
|----------|----------|---------|
| Negative | 4e+02  | 32      |
| Positive   | 4       | 2e+02 |
|          | Negative  | Positive  |


Normal Class

|          |          |         |
|----------|----------|---------|
| Negative | 2e+02  | 4      |
| Positive   | 30       | 4e+02 |
|          | Negative  | Positive  |

### Alpha balanced Focal Loss
I implemented the alpha balanced focal loss and tried different settings of FC layers, alpha weights and gamma. The best performing model had 3 FC layers with 300, 80 and 3 neurons respectively. First and second layer was followed by ReLU activation and Dropout with keep probability of 0.5.
gamma was 2.5 and alpha weights for covid, normal and pneumonia class were 1.2,0.01 and 0.5 respectively.

The network was fine tuned for 20 epochs with batch size of 64. Stochastic Gradient descent with momentum of 0.9 and learning rate of 0.001 was used.

**Accuracy on Validation Set = 96.3 %**

**F1 score on Validation Set = 94.7 %**

#### Confusion Matrix
Following are confusion matrices of each class for validation set.

Covid Class

|          |          |         |
|----------|----------|---------|
| Negative | 6e+02  | 4      |
| Positive   | 2       | 24 |
|          | Negative  | Positive  |


Pneumonia Class

|          |          |         |
|----------|----------|---------|
| Negative | 3.9e+02  | 23      |
| Positive   | 10       | 2.0e+02 |
|          | Negative  | Positive  |


Normal Class

|          |          |         |
|----------|----------|---------|
| Negative | 2e+02  | 7      |
| Positive   | 24       | 3.9e+02 |
|          | Negative  | Positive  |
#### Weights
The weights of this network can be found at thsi [gdrive link](https://drive.google.com/file/d/1-2tynXlui2l7FpNhY8YK7tJIktTD1L11/view?usp=sharing)

## ResNet-18 Model
I used the pre-defined pytorch model of ResNet-18 and removed all of its (last) FC layers. I performed different experiments for various hyper-parameters and I'll report only the best ones here.
### Binary Cross Entropy Loss
I replaced the last FC layers with 3 FC layer. First layer having 200 neuron, second having 50 and third having 3. First and second FC layer had ReLU activation function followed by Dropout layer with drop probability of 0.5. 
I fine tuned the network for 20 epochs with batch size of 64. Stochastic Gradient descent with momentum of 0.9 and learning rate of 0.001 was used.

**Accuracy on Validation Set = 93.8 %**

**F1 score on Validation Set = 91.0 %**

#### Confusion Matrix
Following are confusion matrices of each class for validation set.

Covid Class

|          |          |         |
|----------|----------|---------|
| Negative | 6e+02  | 28      |
| Positive   | 0       | 0 |
|          | Negative  | Positive  |


Pneumonia Class

|          |          |         |
|----------|----------|---------|
| Negative | 3.8e+02  | 28      |
| Positive   | 16       | 2e+02 |
|          | Negative  | Positive  |


Normal Class

|          |          |         |
|----------|----------|---------|
| Negative | 2e+02  | 16      |
| Positive   | 28       | 3.8e+02 |
|          | Negative  | Positive  |

### Alpha balanced Focal Loss
I implemented the alpha balanced focal loss and tried different settings of FC layers, alpha weights and gamma. The best performing model had 3 FC layers with 200, 50 and 3 neurons respectively. First and second layer was followed by ReLU activation and Dropout with keep probability of 0.5.
gamma was 2.0 and alpha weights for covid, normal and pneumonia class were 1,0.05 and 0.1 respectively.

The network was fine tuned for 20 epochs with batch size of 64. Stochastic Gradient descent with momentum of 0.9 and learning rate of 0.001 was used.

**Accuracy on Validation Set = 93.5 %**

**F1 score on Validation Set = 91.0 %**

#### Confusion Matrix
Following are confusion matrices of each class for validation set.

Covid Class

|          |          |         |
|----------|----------|---------|
| Negative | 6e+02  | 28      |
| Positive   | 0       | 0 |
|          | Negative  | Positive  |


Pneumonia Class

|          |          |         |
|----------|----------|---------|
| Negative | 3.8e+02  | 28      |
| Positive   | 17       | 2e+02 |
|          | Negative  | Positive  |


Normal Class

|          |          |         |
|----------|----------|---------|
| Negative | 2e+02  | 11      |
| Positive   | 33       | 3.9e+02 |
|          | Negative  | Positive  |


#### Weights
The weights of this network can be found at this [gdrive line](https://drive.google.com/file/d/141bZ4VMGdmKND2XF1p41MxaoANUf-ibs/view?usp=sharing)
