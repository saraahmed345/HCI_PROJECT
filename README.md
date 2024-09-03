"# HCI_PROJECT" 
EEG Signal Classification Using CSP and Machine Learning
Overview
This project implements an EEG signal classification pipeline using Common Spatial Patterns (CSP) for feature extraction and Gaussian Naive Bayes (GNB) and Support Vector Machine (SVM) classifiers for classification. The pipeline includes preprocessing steps such as bandpass filtering and dimensionality reduction based on correlation analysis.

Features
Bandpass Filtering: Applies a Butterworth bandpass filter to remove noise from EEG signals.
Feature Extraction: Uses Common Spatial Patterns (CSP) to extract relevant features from the EEG data.
Classification: Implements Gaussian Naive Bayes and Support Vector Machine (SVM) classifiers to classify EEG signals into 'left' and 'right' movements.
Hyperparameter Tuning: Performs hyperparameter tuning for the SVM classifier using GridSearchCV.
Prerequisites
Make sure you have the following libraries installed:

pandas
numpy
scikit-learn
scipy
mne
pickle (standard library)
You can install the required libraries using pip:

bash
Copy code
pip install pandas numpy scikit-learn scipy mne
Data
The dataset used in this project is assumed to be in CSV format and located at:
BCICIV_2a_44.csv
The CSV file should contain EEG signal data with columns corresponding to signal features and a 'label' column indicating the class ('left' or 'right').





Run the script using Python:

bash
Copy code
python hcifinalcode.py


Outputs
Classification Reports: Printed for both Gaussian Naive Bayes and SVM classifiers, showing precision, recall, and F1-score.
Confusion Matrices: Printed for both classifiers to visualize the classification performance.
Model File: The best SVM model is saved as best_svm_model.pkl.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Common Spatial Patterns (CSP) for feature extraction.
Support Vector Machine (SVM) and Gaussian Naive Bayes classifiers from scikit-learn.
Data preprocessing and filtering techniques.

