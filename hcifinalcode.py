# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from scipy.signal import butter, filtfilt
from mne.decoding import CSP
import pickle

# Preprocessing parameters
FS = 250
LOWCUT = 7
HIGHCUT = 30
CORRELATION_THRESHOLD = 0.01

# Butterworth bandpass filter design
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to apply bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

# Load data
file_path = r'C:\Users\DELL\Downloads\python_project7_final - Copy\BCICIV_2a_44.csv'
df = pd.read_csv(file_path)

# Select relevant labels and encode them
df = df[df['label'].isin(['left', 'right'])]
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Compute correlation matrix and drop low correlated columns
corr_matrix = df.corr()
low_corr_columns = corr_matrix[abs(corr_matrix['label']) < CORRELATION_THRESHOLD].index
X = df.drop(columns=['label'] + list(low_corr_columns))
y = df['label']

# Apply bandpass filter
X_filtered = X.apply(lambda col: butter_bandpass_filter(col, LOWCUT, HIGHCUT, FS), axis=0)

# Split data into training and testing sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=150, stratify=y)

# Convert DataFrame to numpy array
X_train_array = X_train.values.reshape(-1, X_train.shape[1], 1)
X_test_array = X_test.values.reshape(-1, X_test.shape[1], 1)

# Define the number of CSP components to retain
n_components = 4  # You can adjust this number based on your dataset

# Initialize CSP
csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)

# Fit CSP on training data
X_train_csp = csp.fit_transform(X_train_array, y_train)

# Apply CSP on testing data
X_test_csp = csp.transform(X_test_array)

# Initialize classifiers
nb_model = GaussianNB()
svm_model = SVC()

# Define hyperparameter grids for SVM
param_grid_svm = {
    'C': [ 0.1],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Perform GridSearchCV for SVM
grid_search_svm = GridSearchCV(svm_model, param_grid_svm, cv=StratifiedKFold(n_splits=5), verbose=2)
grid_search_svm.fit(X_train_csp, y_train)
best_svm_model = grid_search_svm.best_estimator_
# Evaluate SVM model with CSP features
y_pred_svm_csp = best_svm_model.predict(X_test_csp)

# Fit and evaluate Gaussian Naive Bayes with CSP features
nb_model.fit(X_train_csp, y_train)
y_pred_nb_csp = nb_model.predict(X_test_csp)


# Print classification report and confusion matrix for Naive Bayes with CSP
print("Classification Report for Gaussian Naive Bayes with CSP:")
print(classification_report(y_test, y_pred_nb_csp))
print("Confusion Matrix for Gaussian Naive Bayes with CSP:")
print(confusion_matrix(y_test, y_pred_nb_csp))


# Print classification report and confusion matrix for SVM with CSP
print("Classification Report for SVM with CSP:")
print(classification_report(y_test, y_pred_svm_csp))
print("Confusion Matrix for SVM with CSP:")
print(confusion_matrix(y_test, y_pred_svm_csp))
# Save the trained SVM model
with open('best_svm_model.pkl', 'wb') as f:
    pickle.dump(best_svm_model, f)
