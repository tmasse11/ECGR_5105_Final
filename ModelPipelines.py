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

def test_case_1(dataset, feature_names, label_base_name="Result_of_Stop"):
    """
    :Description:
    Logistic Regression with Normalized Preprocessing Scaler(80%, 20% split).
    One vs. All Approach will be used to perform multiclassification. A model for each class will be constructed.


    :param dataset: 2D Numpy Array. Contains input data and associated class labels.
    :param feature_names: List of feature names associated with input "dataset" This is needed to correlate the feature
    :param label_base_name: Name of the output label. Possible Labels are "Result_of_Stop" and "Was_a_Search_Conducted"
    to the respective column in the dataset array.
    :return:
    """
    # Split Data into training and validiation datasets.
    training, validation = train_test_split(dataset, train_size=0.8, test_size=0.2, random_state=42) # Random state is locked for Reproducibility.

    # Extract Labels from training and validation sets.

    if label_base_name == "Result_of_Stop":
        label_indicies = []
        label_classes = []
        for indx, feature_name in enumerate(feature_names):
            if label_base_name in feature_name:
                label_classes.append(feature_name)
                label_indicies.append(indx)

        num_of_classes = len(label_classes)
        training_labels = training[:, -num_of_classes:]
        validation_labels = validation[:, -num_of_classes:]
        training_input = training[:, :-num_of_classes]
        validation_input = validation[:, :-num_of_classes]

    elif label_base_name == "Was_a_Search_Conducted":
        label_indx = None
        for indx, feature_name in enumerate(feature_names):
            if "Was_a_Search_Conducted" in feature_name:
                label_indx = indx
                break
        training_labels = training[:, label_indx]
        validation_labels = validation[:, label_indx]
        training_input = np.delete(training, label_indx, axis=1)
        validation_input = np.delete(validation, label_indx, axis=1)
    else:
        print("WARNING: A Possible Label Name Wasn't Specified")

    # Create and train Logistic Regression Model for each Label Class.
    logistic_regression_models = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []

    if label_base_name == "Result_of_Stop":
        for indx, label_class in enumerate(label_classes):
            print("Performing Logistic Regression for Class : {}....".format(label_class))
            binary_classifier_model = LogisticRegression(C=10, random_state=0, max_iter=5000, solver='liblinear')
            binary_classifier_model.fit(training_input, training_labels[:, indx])
            predictions = binary_classifier_model.predict(validation_input)

            # Obtain Test Metrics.
            accuracy_score = metrics.accuracy_score(validation_labels[:, indx], predictions)
            recall_score = metrics.recall_score(validation_labels[:, indx], predictions)
            precision_score = metrics.precision_score(validation_labels[:, indx], predictions)
            accuracy_scores.append(accuracy_score)
            precision_scores.append(precision_score)
            recall_scores.append(recall_score)

            # Store Model.
            logistic_regression_models.append(binary_classifier_model)

            print("Metrics for Logistic Classifier")
            print("Accuracy of Classifier: {:.4f}".format(accuracy_score))
            print("Precision of Classifier: {:.4f}".format(precision_score))
            print("Recall of Classifier: {:.4f}".format(recall_score))
            print("================================================================")

    elif label_base_name == "Was_a_Search_Conducted":
        print("Performing Logistic Regression for Class : {}....".format(label_base_name))
        binary_classifier_model = LogisticRegression(C=10, random_state=0, max_iter=5000, solver='liblinear')
        binary_classifier_model.fit(training_input, training_labels)
        predictions = binary_classifier_model.predict(validation_input)

        # Obtain Test Metrics.
        accuracy_score = metrics.accuracy_score(validation_labels, predictions)
        recall_score = metrics.recall_score(validation_labels, predictions)
        precision_score = metrics.precision_score(validation_labels, predictions)
        accuracy_scores.append(accuracy_score)
        precision_scores.append(precision_score)
        recall_scores.append(recall_score)

        # Store Model.
        logistic_regression_models.append(binary_classifier_model)

        print("Metrics for Logistic Classifier")
        print("Accuracy of Classifier: {:.4f}".format(accuracy_score))
        print("Precision of Classifier: {:.4f}".format(precision_score))
        print("Recall of Classifier: {:.4f}".format(recall_score))
        print("================================================================")
        accuracy_scores.append(accuracy_score)
        precision_scores.append(precision_score)
        recall_scores.append(recall_score)

    return accuracy_scores, precision_scores, recall_scores, logistic_regression_models

def test_case_2(dataset, feature_names, label_base_name="Result_of_Stop", k_value=10):
    """
    :Description:
    Logistic Regression with Normalized Preprocessing Scaler(K-fold split).
    One vs. All Approach will be used to perform multiclassification. A model for each class will be constructed.


    :param dataset: 2D Numpy Array. Contains input data and associated class labels.
    :param feature_names: List of feature names associated with input "dataset" This is needed to correlate the feature
    :param label_base_name: Name of the output label.
    to the respective column in the dataset array.
    :return:
    """
    # Extract Labels from training and validation sets.
    label_indicies = []
    label_classes = []
    if label_base_name == "Result_of_Stop":
        for indx, feature_name in enumerate(feature_names):
            if label_base_name in feature_name:
                label_classes.append(feature_name)
                label_indicies.append(indx)

        num_of_classes = len(label_classes)
        input_data = dataset[:, :-num_of_classes]
        labels = dataset[:, -num_of_classes:]

    elif label_base_name == "Was_a_Search_Conducted":
        label_indx = None
        for indx, feature_name in enumerate(feature_names):
            if "Was_a_Search_Conducted" in feature_name:
                label_indx = indx
                break
        input_data = np.delete(dataset, label_indx, axis=1)
        labels = dataset[:, label_indx]

    else:
        print("WARNING: A Possible Label Name Wasn't Specified")

    # Create and train Logistic Regression Model for each Label Class.
    logistic_regression_models = []
    avg_accuracy_scores = []
    avg_precision_scores = []
    avg_recall_scores = []
    if label_base_name == "Result_of_Stop":
        for indx, label_class in enumerate(label_classes): # iterate through each class label. (One-vs-all Approach)
            print("Performing K Fold Validation for Class Label {}. K Value of {} is used.".format(label_class, k_value))
            scoring_tasks = {'accuracy' : make_scorer(accuracy_score),
                             'precision' : make_scorer(precision_score),
                             'recall' : make_scorer(recall_score)}

            kfold = KFold(n_splits=k_value, shuffle=True, random_state=42)
            binary_classifier_model = LogisticRegression(C=100, max_iter=1000, solver='liblinear')
            results = model_selection.cross_validate(binary_classifier_model, input_data, labels[:, indx], cv=kfold, scoring=scoring_tasks)
            logistic_regression_models.append(binary_classifier_model)
            print("Average Accuracy: {:.3f}".format(results['test_accuracy'].mean()))
            print("Average Precision: {:.3f}".format(results['test_precision'].mean()))
            print("Average Recall: {:.3f}".format(results['test_recall'].mean()))
            print("================================================================")
            avg_accuracy_scores.append(results['test_accuracy'].mean())
            avg_precision_scores.append(results['test_precision'].mean())
            avg_recall_scores.append(results['test_recall'].mean())

    elif label_base_name == "Was_a_Search_Conducted":
        print("Performing K Fold Validation for Class Label {}. K Value of {} is used.".format(label_base_name, k_value))
        scoring_tasks = {'accuracy': make_scorer(accuracy_score),
                         'precision': make_scorer(precision_score),
                         'recall': make_scorer(recall_score)}

        kfold = KFold(n_splits=k_value, shuffle=True, random_state=42)
        binary_classifier_model = LogisticRegression(C=100, max_iter=1000, solver='liblinear')
        results = model_selection.cross_validate(binary_classifier_model, input_data, labels, cv=kfold,
                                                 scoring=scoring_tasks)
        logistic_regression_models.append(binary_classifier_model)
        print("Average Accuracy: {:.3f}".format(results['test_accuracy'].mean()))
        print("Average Precision: {:.3f}".format(results['test_precision'].mean()))
        print("Average Recall: {:.3f}".format(results['test_recall'].mean()))
        print("================================================================")
        avg_accuracy_scores.append(results['test_accuracy'].mean())
        avg_precision_scores.append(results['test_precision'].mean())
        avg_recall_scores.append(results['test_recall'].mean())

    return avg_accuracy_scores, avg_precision_scores, avg_recall_scores, logistic_regression_models

def test_case_3(dataset, feature_names, label_base_name="Result_of_Stop", plot_figures=False):
    """
    :Description:
    Logistic Regression with Normalized Preprocessing Scaler(80%, 20% split).
    PCA Preprocessing will Be Performed.
    One vs. All Approach will be used to perform multiclassification. A model for each class will be constructed.

    :param dataset: 2D Numpy Array. Contains input data and associated class labels.
    :param feature_names: List of feature names associated with input "dataset" This is needed to correlate the feature
    :param label_base_name: Name of the output label.
    to the respective column in the dataset array.
    :return:
    """

    import warnings
    warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

    # Extract Labels from training and validation sets.
    if label_base_name == "Result_of_Stop":
        label_indicies = []
        label_classes = []
        for indx, feature_name in enumerate(feature_names):
            if label_base_name in feature_name:
                label_classes.append(feature_name)
                label_indicies.append(indx)
        num_of_classes = len(label_classes)
        input_data = dataset[:, :-num_of_classes]
        labels = dataset[:, -num_of_classes:]

        # Format Output Metrics for each Label Class.
        best_classifier_models = [None] * num_of_classes
        best_accuracy_scores = [None] * num_of_classes
        best_precision_scores = [None] * num_of_classes
        best_recall_scores = [None] * num_of_classes
        best_k_values = [None] * num_of_classes

    elif label_base_name == "Was_a_Search_Conducted":
        label_classes = [label_base_name]
        label_indx = None
        for indx, feature_name in enumerate(feature_names):
            if "Was_a_Search_Conducted" in feature_name:
                label_indx = indx
                break
        input_data = np.delete(dataset, label_indx, axis=1)
        labels = dataset[:, label_indx]

        # Format Output Metrics for each Label Class.
        best_classifier_models = []
        best_accuracy_scores = []
        best_precision_scores = []
        best_recall_scores = []
        best_k_values = []
    else:
        print("WARNING: A Possible Label Name Wasn't Specified")
        return None, None, None, None, None

    # Perform PCA compression/decompisition
    pca = PCA()
    principal_components = pca.fit_transform(input_data)

    # Split Data into training and validiation datasets.
    training_pcs, validation_pcs, training_labels, validation_labels = train_test_split(principal_components, labels, train_size=0.8, test_size=0.2, random_state=42) # Random state is locked for Reproducibility.

    max_possible_k_components = principal_components.shape[1]

    if label_base_name == "Result_of_Stop":
        for class_indx, label_class in enumerate(label_classes):
            print("==========================================================================================")
            print("Performing Logistic Regression with PCA for Class : {}....".format(label_class))
            best_k_value = 0
            precision_history = np.zeros(max_possible_k_components)
            recall_history = np.zeros(max_possible_k_components)
            accuracy_history = np.zeros(max_possible_k_components)
            best_accuracy = 0

            for num_of_principal_components in range(1, max_possible_k_components):
                binary_classifier_model = LogisticRegression(C=10, random_state=0, max_iter=5000, solver='liblinear')
                binary_classifier_model.fit(training_pcs[:, :num_of_principal_components], training_labels[:, class_indx])
                predictions = binary_classifier_model.predict(validation_pcs[:, :num_of_principal_components])

                # Obtain metrics for current logisstic regression model.
                precision_history[num_of_principal_components-1] = metrics.precision_score(validation_labels[:, class_indx], predictions)
                recall_history[num_of_principal_components-1] = metrics.recall_score(validation_labels[:, class_indx], predictions)
                accuracy_history[num_of_principal_components-1] = metrics.accuracy_score(validation_labels[:, class_indx], predictions)

                # Determine the best amount of principal components.
                if metrics.accuracy_score(validation_labels[:, class_indx], predictions) > best_accuracy:
                    best_accuracy = metrics.accuracy_score(validation_labels[:, class_indx], predictions)
                    best_k_value = num_of_principal_components
                    best_classifier_models[class_indx] = binary_classifier_model

            print("Best Num of Principal Components: ", best_k_value)
            print("Model Accuracy Score at Best K Value: {:.4f}".format(accuracy_history[best_k_value - 1]))
            print("Model Recall Score at Best K Value: {:.4f}".format(recall_history[best_k_value - 1]))
            print("Model Precision Score at Best K Value: {:.4f}".format(precision_history[best_k_value - 1]))

            # Store best Metrics for current logistic classifier.
            best_accuracy_scores[class_indx] = accuracy_history[best_k_value-1]
            best_recall_scores[class_indx] = recall_history[best_k_value-1]
            best_precision_scores[class_indx] = precision_history[best_k_value-1]
            best_k_values[class_indx] = best_k_value

            if plot_figures is True:
                plt.plot(range(1, max_possible_k_components + 1), precision_history, label="Precision")
                plt.plot(range(1, max_possible_k_components + 1), recall_history, label="Recall")
                plt.plot(range(1, max_possible_k_components + 1), accuracy_history, label="Accuracy")
                plt.xlabel("Number of Principal Components Used in Model")
                plt.ylabel("Metric Value")
                plt.legend()
                plt.title(
                    "Validation Precision, Recall, and Accuracy for Various Logistic Regression Models Trained with Different Number of Principal Components")
                plt.show()
            print("================================================================")

    elif label_base_name == "Was_a_Search_Conducted":
        print("==========================================================================================")
        print("Performing Logistic Regression with PCA for Class : {}....".format(label_base_name))
        best_k_value = 0
        precision_history = np.zeros(max_possible_k_components)
        recall_history = np.zeros(max_possible_k_components)
        accuracy_history = np.zeros(max_possible_k_components)
        best_accuracy = 0

        for num_of_principal_components in range(1, max_possible_k_components):
            binary_classifier_model = LogisticRegression(C=10, random_state=0, max_iter=5000, solver='liblinear')
            binary_classifier_model.fit(training_pcs[:, :num_of_principal_components], training_labels)
            predictions = binary_classifier_model.predict(validation_pcs[:, :num_of_principal_components])

            # Obtain metrics for current logisstic regression model.
            precision_history[num_of_principal_components - 1] = metrics.precision_score(
                validation_labels, predictions)
            recall_history[num_of_principal_components - 1] = metrics.recall_score(validation_labels,
                                                                                   predictions)
            accuracy_history[num_of_principal_components - 1] = metrics.accuracy_score(validation_labels,
                                                                                       predictions)

            # Determine the best amount of principal components.
            if metrics.accuracy_score(validation_labels, predictions) > best_accuracy:
                best_accuracy = metrics.accuracy_score(validation_labels, predictions)
                best_k_value = num_of_principal_components
                best_classifier_models = binary_classifier_model

        print("Best Num of Principal Components: ", best_k_value)
        print("Model Accuracy Score at Best K Value: {:.4f}".format(accuracy_history[best_k_value - 1]))
        print("Model Recall Score at Best K Value: {:.4f}".format(recall_history[best_k_value - 1]))
        print("Model Precision Score at Best K Value: {:.4f}".format(precision_history[best_k_value - 1]))

        # Store best Metrics for current logistic classifier.
        best_accuracy_scores.append(accuracy_history[best_k_value - 1])
        best_recall_scores.append(recall_history[best_k_value - 1])
        best_precision_scores.append(precision_history[best_k_value - 1])
        best_k_values.append(best_k_value)
        if plot_figures is True:
            plt.plot(range(1, max_possible_k_components + 1), precision_history, label="Precision")
            plt.plot(range(1, max_possible_k_components + 1), recall_history, label="Recall")
            plt.plot(range(1, max_possible_k_components + 1), accuracy_history, label="Accuracy")
            plt.xlabel("Number of Principal Components Used in Model")
            plt.ylabel("Metric Value")
            plt.legend()
            plt.title(
                "Validation Precision, Recall, and Accuracy for Various Logistic Regression Models Trained with Different Number of Principal Components")
            plt.show()

    return label_classes, best_k_values, best_accuracy_scores, best_precision_scores, best_recall_scores, best_classifier_models

def test_case_4(dataset, feature_names, label_base_name="Result_of_Stop", k_fold_value=10, best_pca_values=None, plot_figures=False):
    """
    :Description:
    Logistic Regression with Normalized Preprocessing with K-Fold Cross Validation.
    PCA Preprocessing will Be Peformed.
    One vs. All Approach will be used to perform multiclassification. A model for each class will be constructed.

    :param dataset: 2D Numpy Array. Contains input data and associated class labels.
    :param feature_names: List of feature names associated with input "dataset" This is needed to correlate the feature
    :param label_base_name: Name of the output label.
    to the respective column in the dataset array.
    :param best_pca_values: List that contains best amount of K components for each class label. The length of this list
    correspond to the amount of class labels. Typically, the best_k_values output from test case_3 is used as the input
    for this field.
    :param plot_figures: Boolean Flag to display corresponding plots. True to Plot, False to not Plot.

    :return:
        label_classes : List of class labels names
        best_accuracy_scores : List of best accuracy scores for all class labels. Each index represents metric value
        for a class. The order of the list matches the order of label_classes.
        best_precision_scores : List of best precision scores for all class labels. Each index represents metric value
        for a class. The order of the list matches the order of label_classes.
        best_recall_scores: List of best recall scores for all class labels. Each index represents metric value
        for a class. The order of the list matches the order of label_classes.
        best_classifier_models : List of Classifier models. Order of list matches order of label_classes list.
    """

    import warnings
    warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

    # Extract Labels from training and validation sets.
    if label_base_name == "Result_of_Stop":
        label_indicies = []
        label_classes = []
        for indx, feature_name in enumerate(feature_names):
            if label_base_name in feature_name:
                label_classes.append(feature_name)
                label_indicies.append(indx)
        num_of_classes = len(label_classes)
        input_data = dataset[:, :-num_of_classes]
        labels = dataset[:, -num_of_classes:]

        # Create and train Logistic Regression Model for each Label Class.
        best_classifier_models = [None] * num_of_classes
        best_accuracy_scores = [None] * num_of_classes
        best_precision_scores = [None] * num_of_classes
        best_recall_scores = [None] * num_of_classes
        best_k_values = [None] * num_of_classes

    elif label_base_name == "Was_a_Search_Conducted":
        label_classes = [label_base_name]
        label_indx = None
        for indx, feature_name in enumerate(feature_names):
            if "Was_a_Search_Conducted" in feature_name:
                label_indx = indx
                break
        input_data = np.delete(dataset, label_indx, axis=1)
        labels = dataset[:, label_indx]
        # Format Output Metrics for each Label Class.
        best_classifier_models = []
        best_accuracy_scores = []
        best_precision_scores = []
        best_recall_scores = []
        best_k_values = []

    else:
        print("WARNING: A Possible Label Name Wasn't Specified")
        return None, None, None, None, None

    # Perform PCA compression/decompisition
    pca = PCA()
    principal_components = pca.fit_transform(input_data)
    max_possible_k_components = principal_components.shape[1]

    if label_base_name == "Result_of_Stop":
        for class_indx, label_class in enumerate(label_classes):
            print("==========================================================================================")
            print("Performing Logistic Regression with PCA and K-fold Validaiton for Class : {}....".format(label_class))
            best_k_value = 0
            precision_history = np.zeros(max_possible_k_components)
            recall_history = np.zeros(max_possible_k_components)
            accuracy_history = np.zeros(max_possible_k_components)
            best_accuracy = 0

            scoring_tasks = {'accuracy': make_scorer(accuracy_score),
                             'precision': make_scorer(precision_score),
                             'recall': make_scorer(recall_score)}

            kfold = KFold(n_splits=k_fold_value, shuffle=True, random_state=42)

            if best_pca_values is None:
                for num_of_principal_components in range(1, max_possible_k_components):
                    print("Analyzing Principal Component : ", num_of_principal_components)
                    binary_classifier_model = LogisticRegression(C=10, random_state=0, max_iter=5000, solver='liblinear')
                    results = model_selection.cross_validate(binary_classifier_model,
                                                             principal_components[:, :num_of_principal_components],
                                                             labels[:, class_indx],
                                                             cv=kfold,
                                                             scoring=scoring_tasks)

                    # Obtain metrics for current logistic regression model.
                    precision_history[num_of_principal_components-1] = results['test_precision'].mean()
                    recall_history[num_of_principal_components-1] = results['test_recall'].mean()
                    accuracy_history[num_of_principal_components-1] = results['test_accuracy'].mean()

                    # Determine the best amount of principal components.
                    if results['test_accuracy'].mean() > best_accuracy:
                        best_accuracy = results['test_accuracy'].mean()
                        best_k_value = num_of_principal_components
                        best_classifier_models[class_indx] = binary_classifier_model

                print("Best Num of Principal Components: ", best_k_value)
                print("Model Mean Accuracy Score at Best K Value: {:.4f}".format(accuracy_history[best_k_value - 1]))
                print("Model Mean Recall Score at Best K Value: {:.4f}".format(recall_history[best_k_value - 1]))
                print("Model Mean Precision Score at Best K Value: {:.4f}".format(precision_history[best_k_value - 1]))

            else:

                binary_classifier_model = LogisticRegression(C=10, random_state=0, max_iter=5000, solver='liblinear')
                results = model_selection.cross_validate(binary_classifier_model,
                                                         principal_components[:, :best_pca_values[class_indx]],
                                                         labels[:, class_indx],
                                                         cv=kfold,
                                                         scoring=scoring_tasks)

                # Obtain metrics for current logistic regression model.
                num_of_principal_components = best_pca_values[class_indx]
                best_k_value = best_pca_values[class_indx]
                precision_history[num_of_principal_components - 1] = results['test_precision'].mean()
                recall_history[num_of_principal_components - 1] = results['test_recall'].mean()
                accuracy_history[num_of_principal_components - 1] = results['test_accuracy'].mean()

                print("Best Num of Principal Components: ", best_k_value)
                print("Model Mean Accuracy Score at Best K Value: {:.4f}".format(accuracy_history[best_k_value - 1]))
                print("Model Mean Recall Score at Best K Value: {:.4f}".format(recall_history[best_k_value - 1]))
                print("Model Mean Precision Score at Best K Value: {:.4f}".format(precision_history[best_k_value - 1]))

            # Store best Metrics for current logistic classifier.
            best_accuracy_scores[class_indx] = accuracy_history[best_k_value-1]
            best_recall_scores[class_indx] = recall_history[best_k_value-1]
            best_precision_scores[class_indx] = precision_history[best_k_value-1]
            best_k_values[class_indx] = best_k_values

            if plot_figures is True:
                plt.plot(range(1, max_possible_k_components + 1), precision_history, label="Precision")
                plt.plot(range(1, max_possible_k_components + 1), recall_history, label="Recall")
                plt.plot(range(1, max_possible_k_components + 1), accuracy_history, label="Accuracy")
                plt.xlabel("Number of Principal Components Used in Model")
                plt.ylabel("Metric Value")
                plt.legend()
                plt.title(
                    "Validation Precision, Recall, and Accuracy for Various Logistic Regression Models Trained with Different Number of Principal Components")
                plt.show()

            print("================================================================")
    elif label_base_name == "Was_a_Search_Conducted":
        print("==========================================================================================")
        print("Performing Logistic Regression with PCA and K-fold Validaiton for Class : {}....".format(label_base_name))
        best_k_value = 0
        precision_history = np.zeros(max_possible_k_components)
        recall_history = np.zeros(max_possible_k_components)
        accuracy_history = np.zeros(max_possible_k_components)
        best_accuracy = 0

        scoring_tasks = {'accuracy': make_scorer(accuracy_score),
                         'precision': make_scorer(precision_score),
                         'recall': make_scorer(recall_score)}

        kfold = KFold(n_splits=k_fold_value, shuffle=True, random_state=42)

        if best_pca_values is None:
            for num_of_principal_components in range(1, max_possible_k_components):
                print("Analyzing Principal Component : ", num_of_principal_components)
                binary_classifier_model = LogisticRegression(C=10, random_state=0, max_iter=5000, solver='liblinear')
                results = model_selection.cross_validate(binary_classifier_model,
                                                         principal_components[:, :num_of_principal_components],
                                                         labels,
                                                         cv=kfold,
                                                         scoring=scoring_tasks)

                # Obtain metrics for current logistic regression model.
                precision_history[num_of_principal_components - 1] = results['test_precision'].mean()
                recall_history[num_of_principal_components - 1] = results['test_recall'].mean()
                accuracy_history[num_of_principal_components - 1] = results['test_accuracy'].mean()

                # Determine the best amount of principal components.
                if results['test_accuracy'].mean() > best_accuracy:
                    best_accuracy = results['test_accuracy'].mean()
                    best_k_value = num_of_principal_components
                    best_classifier_models = binary_classifier_model

            print("Best Num of Principal Components: ", best_k_value)
            print("Model Mean Accuracy Score at Best K Value: {:.4f}".format(accuracy_history[best_k_value - 1]))
            print("Model Mean Recall Score at Best K Value: {:.4f}".format(recall_history[best_k_value - 1]))
            print("Model Mean Precision Score at Best K Value: {:.4f}".format(precision_history[best_k_value - 1]))

        else:

            binary_classifier_model = LogisticRegression(C=10, random_state=0, max_iter=5000, solver='liblinear')
            results = model_selection.cross_validate(binary_classifier_model,
                                                     principal_components[:, :best_pca_values[0]],
                                                     labels,
                                                     cv=kfold,
                                                     scoring=scoring_tasks)

            # Obtain metrics for current logistic regression model.
            num_of_principal_components = best_pca_values[0]
            best_k_value = best_pca_values[0]
            precision_history[num_of_principal_components - 1] = results['test_precision'].mean()
            recall_history[num_of_principal_components - 1] = results['test_recall'].mean()
            accuracy_history[num_of_principal_components - 1] = results['test_accuracy'].mean()

            print("Best Num of Principal Components: ", best_k_value)
            print("Model Mean Accuracy Score at Best K Value: {:.4f}".format(accuracy_history[best_k_value - 1]))
            print("Model Mean Recall Score at Best K Value: {:.4f}".format(recall_history[best_k_value - 1]))
            print("Model Mean Precision Score at Best K Value: {:.4f}".format(precision_history[best_k_value - 1]))

        # Store best Metrics for current logistic classifier.
        best_accuracy_scores.append(accuracy_history[best_k_value - 1])
        best_recall_scores.append(recall_history[best_k_value - 1])
        best_precision_scores.append(precision_history[best_k_value - 1])
        best_k_values.append(best_k_values)

        if plot_figures is True:
            plt.plot(range(1, max_possible_k_components + 1), precision_history, label="Precision")
            plt.plot(range(1, max_possible_k_components + 1), recall_history, label="Recall")
            plt.plot(range(1, max_possible_k_components + 1), accuracy_history, label="Accuracy")
            plt.xlabel("Number of Principal Components Used in Model")
            plt.ylabel("Metric Value")
            plt.legend()
            plt.title(
                "Validation Precision, Recall, and Accuracy for Various Logistic Regression Models Trained with Different Number of Principal Components")
            plt.show()

    return label_classes, best_accuracy_scores, best_precision_scores, best_recall_scores, best_classifier_models


def test_case_5(dataset, feature_names, label_base_name="Result_of_Stop", naive_bayes_only=False, one_v_all=True):
    """
    :Description:
    Naive Bayes with LDA with 80% 20% split.
    LDA Preprocessing will Be Peformed.
    One vs. All Approach will be used to perform multiclassification. A model for each class label will be constructed.

    :param dataset: 2D Numpy Array. Contains input data and associated class labels.
    :param feature_names: List of feature names associated with input "dataset" This is needed to correlate the feature
    :param label_base_name: Name of the output label.
    to the respective column in the dataset array.
    :param one_v_all: Boolean that tells the function whether to fit models using 1 v all approach or one model for all
    output labels.
    :return:
    label_classes-  List of Output Label Classes that were used in Model Fitting. If one_v_all is false or the output
    label is binary - this list will be of length one.
    accuracy_scores - list of accuracy scores. Length is the same as the amount of label classes.
    precision_scores - list of precision scores. Length is the same as the amount of label classes.
    recall_scores - list of recall scores. Length is the same as the amount of label classes.
    models - list of fitted models for each label class defined in label_classes.  Order of this list matches order of
    label_classes

    """

    import warnings
    warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

    # Extract Labels from training and validation sets.
    if label_base_name == "Result_of_Stop":
        label_indicies = []
        label_classes = []
        for indx, feature_name in enumerate(feature_names):
            if label_base_name in feature_name:
                label_classes.append(feature_name)
                label_indicies.append(indx)
        num_of_classes = len(label_classes)
        input_data = dataset[:, :-num_of_classes]
        labels = dataset[:, -num_of_classes:]
        if one_v_all is False:

            label_classes = ["non_one_v_all"]

    elif label_base_name == "Was_a_Search_Conducted":
        label_classes = [label_base_name]
        label_indx = None
        for indx, feature_name in enumerate(feature_names):
            if "Was_a_Search_Conducted" in feature_name:
                label_indx = indx
                break
        input_data = np.delete(dataset, label_indx, axis=1)
        labels = dataset[:, label_indx]
    else:
        print("WARNING: A Possible Label Name Wasn't Specified")
        return None, None, None, None, None

    # Split Data into training and validiation datasets.
    training_data, validation_data, training_labels, validation_labels = train_test_split(input_data, labels,
                                                                                        train_size=0.8, test_size=0.2,
                                                                                        random_state=42)  # Random state is locked for Reproducibility.

    # Construct lists for storing metrics.
    models = []
    precision_scores = []
    accuracy_scores = []
    recall_scores = []

    if label_base_name == "Result_of_Stop" and one_v_all is True:
        for class_indx, label_class in enumerate(label_classes):
            print("Performing Naive Bayes with LDA for Class : ", label_class)
            if naive_bayes_only is False:
                model = LinearDiscriminantAnalysis()
            else:
                model = GaussianNB()

            model.fit(training_data, training_labels[:, class_indx])
            predictions = model.predict(validation_data)

            # Obtain metrics for current Naive bayes model.
            precision = metrics.precision_score(validation_labels[:, class_indx], predictions)
            recall = metrics.recall_score(validation_labels[:, class_indx], predictions)
            accuracy = metrics.accuracy_score(validation_labels[:, class_indx], predictions)

            precision_scores.append(precision)
            accuracy_scores.append(accuracy)
            recall_scores.append(recall)
            models.append(model)

            print("Model  Accuracy Score: {:.4f}".format(accuracy))
            print("Model Recall Score: {:.4f}".format(recall))
            print("Model Precision Score: {:.4f}".format(precision))

            print("================================================================")

    elif label_base_name == "Was_a_Search_Conducted" or one_v_all is False:
        if one_v_all is False and label_base_name == "Result_of_Stop":
            label_base_name = "non_one_v_all" # Used for displaying of Non 1 vs all approach.

        print("Performing Naive Bayes with LDA for Class : ", label_base_name)
        if naive_bayes_only is False:
            model = LinearDiscriminantAnalysis()
        else:
            model = GaussianNB()

        model.fit(training_data, training_labels)
        predictions = model.predict(validation_data)

        # Obtain metrics for current Naive bayes model.
        precision = metrics.precision_score(validation_labels, predictions)
        recall = metrics.recall_score(validation_labels, predictions)
        accuracy = metrics.accuracy_score(validation_labels, predictions)

        precision_scores.append(precision)
        accuracy_scores.append(accuracy)
        recall_scores.append(recall)
        models.append(model)

        print("Model  Accuracy Score: {:.4f}".format(accuracy))
        print("Model Recall Score: {:.4f}".format(recall))
        print("Model Precision Score: {:.4f}".format(precision))

    return label_classes, accuracy_scores, precision_scores, recall_scores, models


def test_case_6(dataset, feature_names, label_base_name="Result_of_Stop", one_v_all=True):
    """
    :Description:
    Support Vector Classification with Kernel Methods. 80%, 20% split.

    One vs. All Approach will be used to perform multiclassification. A model for each class will be constructed.

    :param dataset: 2D Numpy Array. Contains input data and associated class labels.
    :param feature_names: List of feature names associated with input "dataset" This is needed to correlate the feature
    :param label_base_name: Name of the output label.
    to the respective column in the dataset array.
    :param one_v_all : Boolean that tells the function to perform one-v-all or combine all labels into one SVM model.

    :return:
    label_classes : List of Output Label Classes for the Dataset.
    svm_models : Nested Dictionary of SVM models and associated Performance Metrics.
            Format:
                 "class name" -> accuracy : List of accuarcies. Each index represents metric for index of kernel in kernel_name and svm_models
                                 precision :  List of precisions. Each index represents metric for index of kernel in kernel_name and svm_models
                                 recall :  List of recalls. Each index represents metric for index of kernel in kernel_name and svm_models
                                 svm_models : List of ML models. Each index represents model for index of kernel in kernel_name
                                 kernel_names : List of Kernel Names

    """

    import warnings
    warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

    # Extract Labels from training and validation sets.
    if label_base_name == "Result_of_Stop":
        label_indicies = []
        label_classes = []
        for indx, feature_name in enumerate(feature_names):
            if label_base_name in feature_name:
                label_classes.append(feature_name)
                label_indicies.append(indx)
        num_of_classes = len(label_classes)
        input_data = dataset[:, :-num_of_classes]
        labels = dataset[:, -num_of_classes:]

    elif label_base_name == "Was_a_Search_Conducted":
        label_classes = [label_base_name]
        label_indx = None
        for indx, feature_name in enumerate(feature_names):
            if "Was_a_Search_Conducted" in feature_name:
                label_indx = indx
                break
        input_data = np.delete(dataset, label_indx, axis=1)
        labels = dataset[:, label_indx]
    else:
        print("WARNING: A Possible Label Name Wasn't Specified")
        return None, None, None, None, None

    # Split Data into training and validiation datasets.
    training_data, validation_data, training_labels, validation_labels = train_test_split(input_data, labels,
                                                                                        train_size=0.8, test_size=0.2,
                                                                                        random_state=42)  # Random state is locked for Reproducibility.

    # Construct lists for storing metrics.
    svm_models = {}
    # Different Kernel Methods.
    kernel_methods = ['linear', 'poly', 'rbf', 'sigmoid']
    polynomial_degree = [2, 3]

    if label_base_name == "Result_of_Stop" and one_v_all is True:
        for class_indx, label_class in enumerate(label_classes):
            accuracy_scores = []
            precision_scores = []
            recall_scores = []
            models = []
            kernel_names =[]
            print("========================================================================")
            print("Pefroming SVM Classification for Class Label ", label_class)
            for kernel_method in kernel_methods:
                if kernel_method == 'poly':
                    for degree in polynomial_degree:
                        model = SVC(kernel=kernel_method, degree=degree)
                        print("************************************************************")
                        print("Kernel Method: ", kernel_method + "_" + str(degree))
                        kernel_names.append(kernel_method + "_" + str(degree))
                        model.fit(training_data, training_labels[:, class_indx])
                        predictions = model.predict(validation_data)
                        accuracy_scores.append(metrics.accuracy_score(validation_labels[:, class_indx], predictions))
                        precision_scores.append(metrics.precision_score(validation_labels[:, class_indx], predictions))
                        recall_scores.append(metrics.recall_score(validation_labels[:, class_indx], predictions))
                        models.append(model)

                        print("Accuracy Score: {:.4f}".format(metrics.accuracy_score(validation_labels[:, class_indx],
                                                                                     predictions)))
                        print("Precision Score: {:.4f}".format(metrics.precision_score(validation_labels[:, class_indx],
                                                                                       predictions)))
                        print("Recall Score: {:.4f}".format(metrics.recall_score(validation_labels[:, class_indx],
                                                                                 predictions)))
                else:
                    print("************************************************************")
                    print("Kernel Method: ", kernel_method)

                    model = SVC(kernel=kernel_method)
                    model.fit(training_data, training_labels[:, class_indx])

                    predictions = model.predict(validation_data)
                    accuracy_scores.append(metrics.accuracy_score(validation_labels[:, class_indx], predictions))
                    precision_scores.append(metrics.precision_score(validation_labels[:, class_indx], predictions))
                    recall_scores.append(metrics.recall_score(validation_labels[:, class_indx], predictions))
                    kernel_names.append(kernel_method)
                    models.append(model)
                    print("Accuracy Score: {:.4f}".format(metrics.accuracy_score(validation_labels[:, class_indx],
                                                                                 predictions)))
                    print("Precision Score: {:.4f}".format(metrics.precision_score(validation_labels[:, class_indx],
                                                                                   predictions)))
                    print("Recall Score: {:.4f}".format(metrics.recall_score(validation_labels[:, class_indx],
                                                                             predictions)))

            # Package dictionary for this function return.
            svm_models[label_class] = {}
            svm_models[label_class]["accuracy"] = accuracy_scores
            svm_models[label_class]["precision"] = precision_scores
            svm_models[label_class]["recall"] = recall_scores
            svm_models[label_class]["svm_models"] = models
            svm_models[label_class]["kernel_names"] = kernel_names

            print("================================================================")

    elif label_base_name == "Was_a_Search_Conducted" or one_v_all is False:
        if one_v_all is False and label_base_name == "Result_of_Stop":
            label_base_name = "labels" # Used for dictionary keying of Non 1 vs all approach.

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        models = []
        kernel_names = []
        print("========================================================================")
        print("Pefroming SVM Classification for Class Label ", label_base_name)
        for kernel_method in kernel_methods:
            if kernel_method == 'poly':
                for degree in polynomial_degree:
                    model = SVC(kernel=kernel_method, degree=degree)
                    print("************************************************************")
                    print("Kernel Method: ", kernel_method + "_" + str(degree))
                    kernel_names.append(kernel_method + "_" + str(degree))
                    model.fit(training_data, training_labels)
                    predictions = model.predict(validation_data)
                    accuracy_scores.append(metrics.accuracy_score(validation_labels, predictions))
                    precision_scores.append(metrics.precision_score(validation_labels, predictions))
                    recall_scores.append(metrics.recall_score(validation_labels, predictions))
                    models.append(model)

                    print("Accuracy Score: {:.4f}".format(metrics.accuracy_score(validation_labels,
                                                                                 predictions)))
                    print("Precision Score: {:.4f}".format(metrics.precision_score(validation_labels,
                                                                                   predictions)))
                    print("Recall Score: {:.4f}".format(metrics.recall_score(validation_labels,
                                                                             predictions)))
            else:
                print("************************************************************")
                print("Kernel Method: ", kernel_method)

                model = SVC(kernel=kernel_method)
                model.fit(training_data, training_labels)

                predictions = model.predict(validation_data)
                accuracy_scores.append(metrics.accuracy_score(validation_labels, predictions))
                precision_scores.append(metrics.precision_score(validation_labels, predictions))
                recall_scores.append(metrics.recall_score(validation_labels, predictions))
                kernel_names.append(kernel_method)
                models.append(model)
                print("Accuracy Score: {:.4f}".format(metrics.accuracy_score(validation_labels,
                                                                             predictions)))
                print("Precision Score: {:.4f}".format(metrics.precision_score(validation_labels,
                                                                               predictions)))
                print("Recall Score: {:.4f}".format(metrics.recall_score(validation_labels,
                                                                         predictions)))

        # Package dictionary for this function return.
        svm_models[label_base_name] = {}
        svm_models[label_base_name]["accuracy"] = accuracy_scores
        svm_models[label_base_name]["precision"] = precision_scores
        svm_models[label_base_name]["recall"] = recall_scores
        svm_models[label_base_name]["svm_models"] = models
        svm_models[label_base_name]["kernel_names"] = kernel_names


    return svm_models, label_classes
