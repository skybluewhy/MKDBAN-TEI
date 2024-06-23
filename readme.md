# MKDBAN-TEI
MKDBAN-TEI is designed for predicting TCR-epitope binding. We encode TCR and epitope sequences using a learnable residue embedding matrix and employ CNN layers to extract features. An interpretable bilinear attention network isthen used to capture the interaction patterns between TCR and epitope. To improve the model’s performance and generalization capability, we introduce three types of protein sequence features and train additional multiple teacher models to capture biologically meaningful binding 
patterns from different features. Next, the knowledge distillation technique is utilized to transfer the knowledge from trained multi-teacher models to the student model.

# Dependencies
TEINet is writen in Python based on Pytorch. The required software dependencies are listed below:

```
torch
Numpy
pandas
scikit-learn
```
# Data
All the data used in the paper is collected from public databases. We also upload the processed data in the data package.

# Usage of MKDBAN-TEI
Training TEINet:
```
python run_model.py --device "cuda" --epoch 50 --batch_size 64 --train_dataset "./data/train0.csv" --test_dataset "./data/test0.csv"
```
Predict TCR-epitope pairs:
```
python run_model.py --device "cuda" --epoch 50 --batch_size 64 --train_dataset "./data/train0.csv" --test_dataset "./data/test0.csv" --only_test True
```
