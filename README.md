EuroSAT Land Use Classification with CNN
This project implements a Convolutional Neural Network (CNN) for Land Use / Land Cover (LULC) classification using the EuroSAT RGB dataset.
Both baseline and improved training pipelines are provided and compared.

1. Environment Setup 
Create and activate virtual environment:

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

2. Dataset
The project uses EuroSAT RGB dataset from HuggingFace.
Total images: 27,000
Classes: 10 (balanced)
Source: blanchon/EuroSAT_RGB
Dataset is automatically downloaded when training starts.
No manual download is required.

3. Dataset Splitting Strategy
The dataset is split in a stratified manner:
Train: 70%
Validation: 15%
Test: 15%
This is implemented in:

src/dataset_eurosat.py → get_dataloaders()

Data augmentation is applied only to training data:
Random horizontal flip
Random rotation
Validation and test sets use no augmentation.

4. Baseline Model Training
    4.1 What Is Baseline?
        Simple CNN architecture
        No early stopping
        Fixed learning rate
        No learning rate scheduling
    4.2 Run Baseline Training
        python -m src.train_baseline | tee results/baseline_log.txt
    4.3 Output
        Training & validation loss/accuracy printed per epoch
        Best model saved as:
        baseline_best.pth
        Test performance printed at the end

5. Improved Model Training  
    5.1 Improvements Applied
        Early Stopping (prevents overfitting)
        Learning Rate Scheduler (ReduceLROnPlateau)
        Weight Decay (L2 regularization)
        Same CNN architecture for fair comparison
    
    5.2 Run Improved Training (27K dataset)

    python -m src.train_improved | tee results/improved_27k_log.txt

    5.3 Output
        Training stops automatically if validation loss does not improve
        Best model saved as:

        baseline_improved_best.pth

6. Plotting Loss and Accuracy Graphs
Generate training curves:

python -m src.plot_logs

7. How to Demonstrate in Class 

Open project folder

Activate environment:
source venv/bin/activate

Run improved training:
python -m src.train_improved

Show:
Epoch outputs
Early stopping trigger
Test accuracy

Show graphs in figures/

Explain:
Dataset split
Overfitting prevention
Performance improvement

8. Conclusion
This project demonstrates how controlled CNN training, proper dataset splitting, and regularization techniques such as early stopping significantly improve generalization performance on satellite image classification tasks.


