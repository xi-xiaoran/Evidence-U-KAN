# Evidence U-KAN for Trustworthy Medical Image Segmentation

## Project Description:

Trustworthy medical image segmentation aims at deliver accurate and reliable results for clinical decision-making. Most existing methods adopt the evidence deep learning (EDL) paradigm due to its computational efficiency and theoretical robustness. However, the EDL-based methods often neglect leveraging uncertainty maps -rich in attention cues- to refine ambiguous boundary segmentation. To address this, we propose a Progressive Evidence Uncertainty guided Attention (PEUA) mechanism to guide the model to focus on the feature representation learning of hard samples. Unlike conventional approaches, PEUA iteratively refines attention using uncertainty maps while employing low-rank learning to denoise attention weights, enhancing feature learning for challenging regions. Concurrently, standard EDL methods suppress non-ground-truth evidence indiscriminately via KL regularization, distorting attention in ambiguous areas. We thus introduce a semantic-preserving evidence learning (SAEL) strategy, integrating a semantic-smooth evidence generator and a fidelity-enhancing regularization term to retain critical semantics. Finally, by embedding PEUA and SAEL with the state-of-the-art U-KAN, we proposes Evidence U-KAN, a novel solution for trustworthy medical image segmentation. Extensive experiments on 4 datasets demonstrate superior accuracy and reliability over the competing methods.

![EUGA](https://github.com/xi-xiaoran/Evidence-U-KAN/blob/main/Plot/EUGA.png)
![SAEL](https://github.com/xi-xiaoran/Evidence-U-KAN/blob/main/Plot/SAEL.png)
![Display image of segmentation effect](https://github.com/xi-xiaoran/Evidence-U-KAN/blob/main/Plot/result.PNG)

## Installation:

This project requires the following Python libraries:

torch 1.13.1  
torchvision 0.14.1  
numpy 1.21.6  
scikit-learn 1.0.2  

(Anyway, the author is using the version above)

## Usage:

### folder
- `data` : Used for storing data
- `deal_data` : Includes data preprocessing
- `Loss_functions` : Includes all loss functions
- `models` : Contains definitions for all models
- `Plot` : Include the images presented in the paper
- `save_models` : Used to save model parameters
- `see` : Used to save visualized training results
- `show` : Used to save comparison results of different models
### py file
- `ASSD` : Calculate ASSD value
- `Bys.py` : Includes Bayesian based model settings
-`Bys_test.py` : Testing of Bayesian Model
- `Bys_train`.py : Training of Bayesian Model
- `Cross_dataset.py` : Setting up cross dataset training
- `main.py` : Include the settings of our model
- `Plotandsave.py` : Compare different models and visualize them, and save the results to the show folder
- `Test_model.py` : Testing our model
- `train_model.py` : Training our model

** Please note that if there is no corresponding folder, it must be created first **
