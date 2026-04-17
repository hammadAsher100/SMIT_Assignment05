📋 Assignment Overview
This project implements a Stroke Prediction System using an Artificial Neural Network (ANN) to predict the likelihood of a patient having a stroke based on various health and demographic factors. The system is built using the Kaggle Stroke Prediction Dataset and includes comprehensive data preprocessing, model training, and evaluation.

🎯 Objective
Develop a binary classification model that predicts whether a patient is at risk of stroke (0 = No Stroke, 1 = Stroke) using patient health records and demographic information.

📊 Dataset Information
Source: Kaggle Stroke Prediction Dataset

Dataset Size: 5,110 samples with 12 features

Features Description
Feature	Type	Description
id	Integer	Unique patient identifier
gender	Categorical	Male, Female, or Other
age	Float	Patient age in years
hypertension	Binary	0 = No hypertension, 1 = Hypertension
heart_disease	Binary	0 = No heart disease, 1 = Heart disease
ever_married	Categorical	Yes or No
work_type	Categorical	Private, Self-employed, Govt_job, children, Never_worked
Residence_type	Categorical	Urban or Rural
avg_glucose_level	Float	Average glucose level in blood
bmi	Float	Body Mass Index
smoking_status	Categorical	formerly smoked, never smoked, smokes, Unknown
stroke	Binary	Target: 0 = No stroke, 1 = Stroke
Data Challenges
Class Imbalance: Only ~5% of samples are positive (stroke cases)

Missing Values: 201 missing values in BMI column

Mixed Data Types: Both numerical and categorical features

🛠️ Technologies Used
Python 3.8+

TensorFlow 2.x / Keras - Deep learning framework

Scikit-learn - Data preprocessing and evaluation

Pandas - Data manipulation

NumPy - Numerical operations

Matplotlib & Seaborn - Data visualization

Joblib - Model serialization

📁 Project Structure
text
stroke-prediction-ann/
│
├── stroke_prediction.ipynb          # Main Jupyter notebook
├── README.md                         # Project documentation
├── requirements.txt                  # Dependencies
│
├── models/                          # Saved models directory
│   ├── stroke_prediction_ann.h5    # Trained ANN model
│   ├── scaler.pkl                   # StandardScaler object
│   ├── label_encoders.pkl           # Label encoders
│   └── feature_columns.pkl          # Feature names
│
├── reports/                         # Evaluation reports
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
└── data/                           # Dataset directory
    └── healthcare-dataset-stroke-data.csv
🔧 Implementation Steps
1. Data Preprocessing
Handling Missing Values
Only BMI column contains 201 missing values (3.93% of data)

Missing BMI values filled with median to maintain data distribution

python
bmi_median = df['bmi'].median()
df['bmi'].fillna(bmi_median, inplace=True)
Encoding Categorical Variables
Label Encoding for binary categories: gender, ever_married, Residence_type

One-Hot Encoding for multi-category columns: work_type, smoking_status

Feature Scaling
Applied StandardScaler to numerical features: age, avg_glucose_level, bmi

Transformed to zero mean and unit variance

2. Train-Test Split
80% training, 20% testing

Stratified split to maintain class distribution

3. ANN Architecture
python
Model Architecture:
- Input Layer: (n_features,)
- Hidden Layer 1: 128 neurons, ReLU, BatchNorm, Dropout(0.3)
- Hidden Layer 2: 64 neurons, ReLU, BatchNorm, Dropout(0.3)
- Hidden Layer 3: 32 neurons, ReLU, BatchNorm, Dropout(0.2)
- Hidden Layer 4: 16 neurons, ReLU, BatchNorm, Dropout(0.2)
- Output Layer: 1 neuron, Sigmoid activation
Total Parameters: ~15,000 trainable parameters

4. Model Training Configuration
Optimizer: Adam

Loss Function: Binary Crossentropy

Metrics: Accuracy, Precision, Recall

Batch Size: 32

Epochs: 100 (with early stopping)

Validation Split: 20% of training data

Class Weights: Applied to handle class imbalance

5. Handling Class Imbalance
python
Class weights calculated using sklearn:
- Class 0 (No Stroke): ~0.526
- Class 1 (Stroke): ~9.474
6. Early Stopping
Monitors validation loss

Patience: 15 epochs

Restores best weights

📈 Evaluation Metrics
The model is evaluated using multiple metrics:

Primary Metrics
Accuracy: Overall correctness of predictions

Precision: Accuracy of positive predictions

Recall (Sensitivity): Ability to find positive cases

F1-Score: Harmonic mean of precision and recall

Secondary Metrics
Specificity: True negative rate

Negative Predictive Value (NPV)

ROC-AUC Score

Confusion Matrix Analysis

🚀 How to Run the Project
Prerequisites
Install Python 3.8+

Install required packages:

bash
pip install -r requirements.txt
Or install individually:

bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn joblib
Step-by-Step Execution
Clone/Download the project

Place the dataset in the data/ directory:

Download from Kaggle

Save as healthcare-dataset-stroke-data.csv

Run the Jupyter notebook:

bash
jupyter notebook stroke_prediction.ipynb
Execute cells in order from Shell 1 to Shell 15

Model will be saved in the models/ directory

🎯 Making Predictions
Using the Prediction Function
python
from stroke_prediction import predict_stroke_risk

# Example prediction
result = predict_stroke_risk(
    age=65,
    gender='Male',
    hypertension=1,
    heart_disease=0,
    ever_married='Yes',
    work_type='Private',
    residence_type='Urban',
    avg_glucose_level=120.5,
    bmi=28.5,
    smoking_status='formerly smoked'
)

print(result)
# Output: {'prediction': 'Stroke Risk', 'probability': 0.73, 'risk_percentage': '73.00%'}
Loading Saved Model
python
import tensorflow as tf
import joblib

# Load model
model = tf.keras.models.load_model('models/stroke_prediction_ann.h5')

# Load preprocessors
scaler = joblib.load('models/scaler.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')
📊 Expected Results
Based on typical runs with this dataset:

Metric	Value
Accuracy	85-92%
Precision	70-85%
Recall	65-80%
F1-Score	68-82%
ROC-AUC	0.85-0.92
Note: Results may vary due to random initialization and class imbalance handling

🔍 Key Findings
Class Imbalance Impact:

Model struggles to identify stroke cases due to severe imbalance (5% positive)

Class weights significantly improve recall but may reduce precision

Important Features:

Age is the strongest predictor

Average glucose level and hypertension also significant

BMI shows moderate correlation

Model Performance:

High specificity (>90%) but moderate sensitivity (70-80%)

Good for ruling out stroke, but may miss some cases

💡 Recommendations for Improvement
Data-Level Approaches:

SMOTE (Synthetic Minority Over-sampling) for balancing

Collect more stroke cases for better training

Model-Level Approaches:

Try different architectures (more/less layers)

Experiment with different activation functions

Use different class weight strategies

Algorithm Alternatives:

Random Forest or XGBoost for comparison

Ensemble methods combining multiple models

Feature Engineering:

Create interaction features (e.g., age × hypertension)

BMI categories instead of continuous values

📝 Limitations
Data Imbalance: Limited stroke cases affect model generalization

Missing Data: 201 missing BMI values, though handled with median

Unknown Smoking Status: Many patients have 'Unknown' status

Binary Output: Only predicts stroke risk, not severity

Limited Features: No lifestyle or genetic factors included

🔮 Future Work
Implement cross-validation for more robust evaluation

Try deep learning with more complex architectures

Deploy as web API using Flask/FastAPI

Create interactive dashboard with Streamlit

Add SHAP values for model interpretability

Experiment with different balancing techniques

Implement ensemble methods

📚 References
Kaggle Dataset: Stroke Prediction Dataset

TensorFlow Documentation: Keras Guide

Scikit-learn Documentation: Model Evaluation

👨‍💻 Author
Assignment Submission - Artificial Neural Network Implementation for Stroke Prediction

📄 License
This project is for educational purposes as part of the course assignment.

🤝 Acknowledgments
Kaggle for providing the dataset

TensorFlow and Scikit-learn teams for excellent ML libraries

📞 Support
For issues or questions regarding this implementation:

Check the Jupyter notebook for detailed comments

Verify all required packages are installed

Ensure dataset is correctly placed in the data directory

Note: This model is for educational purposes and should not be used for actual medical diagnosis without proper validation and clinical approval.

