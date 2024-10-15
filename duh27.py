import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import sklearn
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.svm


class my_svm:
    """
    A class to handle the creation, training, and evaluation of a Support Vector
    Machine (SVM) model for binary classification tasks, including data
    preprocessing, feature creation, cross-validation, and confusion matrix
    visualization.
    """

    def __init__(self, year: str) -> None:
        """
        Initialize the SVM model, set the data path for the specific year, and
        instantiate the SVM model with a linear kernel and a standard scaler for
        normalization.

        Args:
            year (str): The year for which the dataset will be used.
        """
        self.data_path = os.path.dirname(
            os.path.abspath(__file__)) + f"/data-{year}/"
        self.svm_model = sklearn.svm.SVC(kernel="linear")
        self.scaler = sklearn.preprocessing.StandardScaler()

    def preprocess(
        self, data: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Normalize the input data using a standard scaler, handle missing values,
        and align the labels with valid data entries.

        Args:
            data (np.ndarray): The input features for the SVM.
            labels (np.ndarray): The target labels for classification.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing normalized data
            and corresponding labels.
        """
        # Normalize data using standard scaler
        data_normalized = self.scaler.fit_transform(data)

        # Drop rows with missing values after normalization
        data_normalized = pd.DataFrame(data_normalized).dropna()

        # Match labels with the valid data indices
        valid_indices = data_normalized.index
        labels = labels[valid_indices]

        return data_normalized.values, labels

    def feature_creation(self, fs_value: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Create feature sets based on the given feature set value,
        loading positive and negative feature data from corresponding files.

        Args:
            fs_value (str): The feature set identifier to determine which
            features to load.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the feature matrix
            and labels (0 for negative, 1 for positive).
        """
        # Mapping of feature sets to corresponding files and column ranges
        feature_sets = {
            "FS-I": (
                "pos_features_main_timechange.npy",
                "neg_features_main_timechange.npy",
                slice(0, 18),
            ),
            "FS-II": (
                "pos_features_main_timechange.npy",
                "neg_features_main_timechange.npy",
                slice(18, 90),
            ),
            "FS-III": (
                "pos_features_historical.npy",
                "neg_features_historical.npy",
                slice(0, 1),
            ),
            "FS-IV": (
                "pos_features_maxmin.npy",
                "neg_features_maxmin.npy",
                slice(0, 18),
            ),
        }

        pos_features_file, neg_features_file, col_range = feature_sets[fs_value]

        # Load positive and negative features from files
        pos_features = np.load(
            self.data_path + pos_features_file)[:, col_range]
        neg_features = np.load(
            self.data_path + neg_features_file)[:, col_range]

        # Concatenate positive and negative features
        all_features = np.concatenate((pos_features, neg_features), axis=0)

        # Generate corresponding labels for positive and negative classes
        labels = np.concatenate(
            (np.ones(len(pos_features)), np.zeros(len(neg_features)))
        )

        return all_features, labels

    def visualize_confusion_matrix(self, cm: np.ndarray) -> None:
        """
        Visualize the confusion matrix using a heatmap to show the
        classification results of the SVM model.

        Args:
            cm (np.ndarray): Confusion matrix array.
        """
        plt.figure(figsize=(8, 6))
        seaborn.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Flare", "Flare"],
            yticklabels=["No Flare", "Flare"],
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    def cross_validation(
        self, data: np.ndarray, labels: np.ndarray
    ) -> tuple[float, float]:
        """
        Perform 10-fold cross-validation to evaluate the SVM model. Tracks
        accuracy and TSS scores, visualizes results across folds, and generates
        the confusion matrix.

        Args:
            data (np.ndarray): Input features for cross-validation.
            labels (np.ndarray): Target labels for cross-validation.

        Returns:
            tuple[float, float]: Average accuracy and average TSS score across
            all folds.
        """
        # Initialize k-fold cross-validator and lists to store scores
        kf = sklearn.model_selection.KFold(
            n_splits=10, shuffle=True, random_state=42)
        accuracies = []
        tss_scores = []

        for train_index, test_index in kf.split(data):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # Train the SVM model on the current fold
            self.training(X_train, y_train)

            # Predict on the test fold
            y_pred = self.svm_model.predict(X_test)

            # Calculate the TSS score and accuracy
            tss_score = self.tss(y_test, y_pred)
            accuracy = np.mean(y_test == y_pred)

            # Append fold results to lists
            tss_scores.append(tss_score)
            accuracies.append(accuracy)

        # Plot TSS scores across folds
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 11), tss_scores,
                 marker="o", linestyle="-", color="b")
        plt.title("TSS Scores Across K-Folds")
        plt.xlabel("Fold Number")
        plt.ylabel("TSS Score")
        plt.xticks(range(1, 11))
        plt.grid()
        plt.show()

        # Generate and visualize the confusion matrix
        cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
        self.visualize_confusion_matrix(cm)

        return np.mean(accuracies), np.mean(tss_scores)

    def training(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the SVM model on the provided training data.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training labels.
        """
        self.svm_model.fit(X_train, y_train)

    def tss(
        self, true_labels: np.ndarray,
        predicted_labels: np.ndarray
    ) -> float:
        """
        Calculate the True Skill Statistic (TSS) to evaluate model performance
        based on true positives, false negatives, false positives, and true
        negatives.

        Args:
            true_labels (np.ndarray): True labels for the test set.
            predicted_labels (np.ndarray): Predicted labels from the model.

        Returns:
            float: The TSS score, ranging from -1 to 1.
        """
        TP = sum((true_labels == 1) & (predicted_labels == 1))
        FN = sum((true_labels == 1) & (predicted_labels == 0))
        FP = sum((true_labels == 0) & (predicted_labels == 1))
        TN = sum((true_labels == 0) & (predicted_labels == 0))

        TSS = (TP / (TP + FN)) - (FP / (FP + TN))
        return TSS


def feature_experiment() -> str:
    """
    Run an experiment across different feature sets (FS-I, FS-II, FS-III, FS-IV)
    to find the feature set that results in the highest TSS score during
    cross-validation.

    Returns:
        str: The feature set that produced the highest TSS score.
    """
    highest_TSS = -2.0
    highest_TSS_fs = None

    # Loop through each feature set and perform the experiment
    for fs_value in ["FS-I", "FS-II", "FS-III", "FS-IV"]:
        print(f"Running experiment with feature set: {fs_value}")

        # Initialize the SVM model for the year 2010-15
        svm_model = my_svm(year="2010-15")

        # Create feature set and labels based on the current feature set
        data, labels = svm_model.feature_creation(fs_value)

        # Preprocess the data and labels (normalize and remove NaN values)
        data_preprocessed, labels_preprocessed = svm_model.preprocess(
            data, labels)

        # Perform cross-validation and get average accuracy and TSS score
        accuracy, tss_score = svm_model.cross_validation(
            data_preprocessed, labels_preprocessed
        )

        # Track the highest TSS score and corresponding feature set
        if tss_score > highest_TSS:
            highest_TSS = tss_score
            highest_TSS_fs = fs_value

        print(
            f"Feature Set {fs_value}: Avg Accuracy: {
                accuracy}, Avg TSS: {tss_score}\n"
        )

    return highest_TSS_fs


def data_experiment(best_fs: str) -> None:
    """
    Run experiments on different datasets (2010-15 and 2020-24) using the best
    feature set from the feature_experiment. Show results and class 
    distributions.

    Args:
        best_fs (str): The best feature set determined from feature_experiment.
    """
    print(f"Using Feature Set {best_fs}")

    # Loop through each dataset year and perform the experiment
    for year in ["2010-15", "2020-24"]:
        print(
            f"Running experiment with dataset: {
                year} and feature set: {best_fs}"
        )

        # Initialize the SVM model for the given year
        svm_model = my_svm(year)

        # Create features and labels for the given dataset
        data, labels = svm_model.feature_creation(best_fs)

        # Preprocess the data (normalize and remove NaN values)
        data_preprocessed, labels_preprocessed = svm_model.preprocess(
            data, labels)

        # Store labels for each year
        if year == "2010-15":
            labels_2010_2015 = labels_preprocessed
        else:
            labels_2020_2024 = labels_preprocessed

        # Perform cross-validation and get average accuracy and TSS score
        accuracy, tss_score = svm_model.cross_validation(
            data_preprocessed, labels_preprocessed
        )

        print(
            f"Dataset {year}: Avg Accuracy: {
                accuracy}, Avg TSS: {tss_score}\n"
        )

    # Show the class distribution across datasets
    show_class_distribution(labels_2010_2015, labels_2020_2024)


def show_class_distribution(
    labels_2010_2015: np.ndarray, labels_2020_2024: np.ndarray
) -> None:
    """
    Plot the class distribution (Flare/No Flare) for both datasets.

    Args:
        labels_2010_2015 (np.ndarray): The preprocessed labels
        for the 2010-2015 dataset.
        labels_2020_2024 (np.ndarray): The preprocessed labels
        for the 2020-2024 dataset.
    """
    # Calculate the counts for each class (Flare/No Flare) in both datasets
    counts_2010_2015 = {
        0: np.sum(labels_2010_2015 == 0),
        1: np.sum(labels_2010_2015 == 1),
    }

    counts_2020_2024 = {
        0: np.sum(labels_2020_2024 == 0),
        1: np.sum(labels_2020_2024 == 1),
    }

    # Create a DataFrame for plotting class distributions
    class_distribution = pd.DataFrame(
        {
            "Dataset": ["2010-2015", "2010-2015", "2020-2024", "2020-2024"],
            "Class": ["No Flare", "Flare", "No Flare", "Flare"],
            "Count": [
                counts_2010_2015[0],
                counts_2010_2015[1],
                counts_2020_2024[0],
                counts_2020_2024[1],
            ],
        }
    )

    # Pivot the DataFrame for easier plotting
    pivot_df = class_distribution.pivot(
        index="Dataset", columns="Class", values="Count"
    )

    # Plot the class distribution as a stacked bar chart
    pivot_df.plot(
        kind="bar", stacked=True, figsize=(10, 6), color=["#66c2a5", "#fc8d62"]
    )
    plt.title("Class Distribution for Datasets")
    plt.xlabel("Dataset")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.legend(title="Class", loc="upper right")
    plt.grid(axis="y")
    plt.show()


best_fs = feature_experiment()
data_experiment(best_fs)
