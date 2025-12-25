EuroSAT Land Use Classification with CNN
This project implements a Convolutional Neural Network (CNN) for Land Use / Land Cover (LULC) classification using the EuroSAT RGB dataset.
Both baseline and improved training pipelines are implemented and compared to analyze the impact of training strategies on model generalization.


1. Environment Setup
Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


2. Dataset
Dataset: EuroSAT RGB
Source: HuggingFace (blanchon/EuroSAT_RGB)
Total images: 27,000
Number of classes: 10 (balanced)
 Note:
The dataset is automatically downloaded when training or evaluation starts.
No manual download is required.


3. Dataset Splitting Strategy
The dataset is split in a stratified manner:
Train: 70%
Validation: 15%
Test: 15%
Implementation:

src/dataset_eurosat.py → get_dataloaders()

Data Augmentation
Applied only to training data:
Random horizontal flip
Random rotation
Validation and test sets use no augmentation.


4. Baseline Model Training

4.1 Baseline Definition

Simple CNN architecture
Fixed learning rate
No learning rate scheduling
No early stopping

4.2 Run Baseline Training

python -m src.train_baseline | tee results/baseline_log.txt

4.3 Output

Training and validation loss/accuracy printed per epoch
Best model saved as:

baseline_best.pth

5. Improved Model Training

5.1 Improvements Applied

Early stopping (validation loss–based)
Learning rate scheduler (ReduceLROnPlateau)
Weight decay (L2 regularization)
Same CNN architecture (for fair comparison)

5.2 Run Improved Training

python -m src.train_improved | tee results/improved_27k_log.txt

5.3 Output

Training stops automatically if validation loss does not improve
Best model saved as:

baseline_improved_best.pth

6. Evaluation (Test Performance & Confusion Matrix)

To evaluate the trained model on the test set and generate the confusion matrix:

python -m src.evaluate

Outputs:
Test accuracy, precision, recall, F1-score
Confusion matrix saved to:

figures/confusion_matrix.png

7. Plotting Training Curves
To generate training and validation loss/accuracy plots:

python -m src.plot_logs

Outputs:

Training vs Validation Loss (Figure 2)
Training vs Validation Accuracy (Figure 3)
Saved under:
figures/

8. Prediction (Single Image Inference)
To run inference on a single image:

python -m src.predict path/to/image.jpg

The script outputs Top-3 predicted classes with probabilities.

9. Reproducibility Notes

Random seeds are fixed to ensure reproducible experiments.
Training, evaluation, and plotting are separated into different scripts for clarity.
All experiments can be reproduced using the commands above.

10. Demo 

A simple demo interface is provided for qualitative testing:

python app.py

11. Conclusion

This project demonstrates how controlled CNN training, proper dataset splitting, and regularization techniques such as learning rate scheduling and early stopping significantly improve generalization performance on satellite image classification tasks.