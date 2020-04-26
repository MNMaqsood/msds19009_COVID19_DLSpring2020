# msds19009_COVID19_DLSpring2020
This repository contains code and results for COVID-19 classification assignment by Deep Learning Spring 2020 course offered at Information Technology University, Lahore, Pakistan. This assignment is only for learning purposes and is not intended to be used for clinical purposes.

## Identification of COVID-19 patient by using chest X-ray images by exploiting Transfer Learning on VGG-16 and ResNet-18 models(pre-trained on ImageNet Dataset)
## Data
The dataset used in this notebook can be found at this [Gdrive Link](https://drive.google.com/a/itu.edu.pk/uc?id=1-HQQciKYfwAO3oH7ci6zhg45DduvkpnK)
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

Weights of this model are at [this link](https://drive.google.com/open?id=1uRpzAqoQolGbNYJtm1r2yVXPKCSRTACd)
### ResNet Network
**Test accuracy = 96%**
Confusion Matrix:

|          |          |         |
|----------|----------|---------|
| Infected | 5.8e+02  | 37      |
| Normal   | 19       | 8.7e+02 |
|          | infected | Normal  |

Weights of this model are at [this link](https://drive.google.com/open?id=1-1Xfgw9EVeRjkRRA6In07UbE7DfW1rel)

