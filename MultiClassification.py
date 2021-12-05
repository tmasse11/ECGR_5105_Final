from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets
import UtilityFunctions
import ModelPipelines

# This script will use multiple classification models for each output class.
"""
Input Features: 
Reason_for_Stop{Speeding, Vehicle Regulatory, Vehicle Equipment, Stop Light/Sign, Safe Movement, Investigation, Other, SeatBelt, Driving While Impaired, Checkpoint}
Officer_Race {White, Black/African American, Asian/Pacific Islander, Hispanic/Latino, 2 or More, Not Specified, American Indian/Alaska Native}
Officer_Gender {Male, Female}
Officer_Years_of_Service
Driver_Race {Black, White, Other/Unknown, Asian, Native American}
Driver_Ethnicity {Hispanic, Non-Hispanic}
Driver_Gender {Male, Female}
Driver_Age
Was_a_Search_Conducted {yes, no}
CMPD_Division{South Division, Providence Division, North Division, University City Division, Independence Division}

Labels: 
Result_of_Stop {Verbal Warning, Citation Issued, Written Warning, No Action Taken, Arrest}

Test Cases: 
1. Logistic Regression with Normalized Preprocessing Scaler(80%, 20% split). 
2. Logistic Regression with K fold Cross Validation. 
3. Logistic Regression with PCA (80%, 20% Split).
4. Logistic Regression with PCA and K-fold Cross Validation. 
5. Naive Bayes with LDA Decomposition
6. Support Vector Classifier with Different Kernerlizations (80%, 20% Split).
"""

# Import DataSet.
dataset = pd.DataFrame(pd.read_csv(r"Officer_Traffic_Stops.csv"))

# Extract Features of Interest.
dataset.pop("Month_of_Stop")
dataset.pop("GlobalID")
dataset.pop("OBJECTID")

# Preprocess Categorical Input Features within Dataset.
dataset = UtilityFunctions.map_binary_input(dataset)
dataset = UtilityFunctions.dummy_variable_encode(dataset)

# Obtain Feature Names of Preprocessed Dataset.
features = list(dataset)
# Perform Min Max Scale of Input Features
dataset = UtilityFunctions.min_max_scale(dataset)

# Perform Test Cases.
# ModelPipelines.test_case_1(dataset, features, label_base_name="Result_of_Stop")
# # print('**********************************************')
# ModelPipelines.test_case_2(dataset, features, label_base_name="Result_of_Stop")

# print("***********************************************")
_, best_k_values, _, _, _, _ = ModelPipelines.test_case_3(dataset, features,
                                                           label_base_name="Result_of_Stop")
# print("************************************************")
ModelPipelines.test_case_4(dataset, features, best_pca_values=best_k_values, label_base_name="Result_of_Stop")
# test_case_5(dataset, features, naive_bayes_only=False, label_base_name="Result_of_Stop")
print("*****************************************************")
# ModelPipelines.test_case_6(dataset, features, label_base_name="Result_of_Stop")