# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a Random Forest Classifier trained using Scikit-learn 1.5.1 It applies OneHotEncoding for categorical variables and LabelBinarizer for the target variable. The data is split 80-20 for training and evaluation.

## Intended Use
This model predicts income levels based on demographic.it is intended for research and learning but should not be used for hiring, financial decisions, or policy-making.
## Training Data
The model was trained on the Census Income Dataset.The dataset contains 32,561 entries with attributes such as age, education, occupation, and hours worked per week. 
## Evaluation Data
A 20% split of the census dataset was used for validation. The same preprocessing steps were applied to ensure consistency. Model performance was measured using Precision, Recall, and F1-score.
## Metrics
The model’s key performance scores are: Precision: 0.7455, Recall: 0.6359, F1-score: 0.6864

## Ethical Considerations
The dataset has sensitive demographic features, which could cause biased predictions. Performance differs across race, gender, and education levels, so more bias mitigation is needed before real-world use.
## Caveats and Recommendations
The model is trained on 1994 census data, which may not reflect current economic conditions. Regular retraining with updated data and bias analysis is recommended before deployment.

