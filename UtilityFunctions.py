import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import sklearn


def dummy_variable_encode(dataset):
    """
    Performs A one-hot encoding variant for all categorical input features within the dataset.
    :param dataset:
    :return: dataset_encoded : Dataset with One Hot Encoded Columns
    """

    dataset_encoded = pd.get_dummies(dataset, columns=['Reason_for_Stop'],  prefix='Reason_for_Stop',
                                            prefix_sep='.', dummy_na=False, )

    dataset_encoded = pd.get_dummies(dataset_encoded, columns=['Officer_Race'], prefix='Officer_Race',
                                            prefix_sep='.', dummy_na=False, )

    dataset_encoded = pd.get_dummies(dataset_encoded, columns=['Driver_Race'], prefix='Driver_Race',
                                            prefix_sep='.', dummy_na=False, )

    dataset_encoded = pd.get_dummies(dataset_encoded, columns=['CMPD_Division'], prefix='CMPD_Division',
                                            prefix_sep='.', dummy_na=False, )


    dataset_encoded = pd.get_dummies(dataset_encoded, columns=["Result_of_Stop"], prefix="Result_of_Stop",
                                                prefix_sep='.', dummy_na=False)

    return dataset_encoded


def map_binary_input(dataset):
    """
    :param dataset: 
    :param input_fields: 
    :param input_feature_name: 
    :return: dataset
    """""

    # Map Binary Feature Driver_Gender
    dataset["Driver_Gender"] = dataset["Driver_Gender"].map({'Female': 0, 'Male': 1})

    # Map Binary Feature Officer_Gender
    dataset["Officer_Gender"] = dataset["Officer_Gender"].map({'Female': 0, 'Male': 1})

    # Map Binary Feature Was_a_Search_Conducted
    dataset["Was_a_Search_Conducted"] = dataset["Was_a_Search_Conducted"].map({'Yes': 1, 'No': 0})

    # Map Binary Feature Driver_Ethnicity
    dataset["Driver_Ethnicity"] = dataset["Driver_Ethnicity"].map({'Non-Hispanic': 1, 'Hispanic': 0})

    return dataset


def min_max_scale(dataset):
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)
    return dataset
