import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Project 1: Kannada MNIST - Classification Problem

# Load Kannada MNIST data
data = np.load('path/to/kannada-mnist.npz')
X_train, y_train = data['train_images'], data['train_labels']
X_test, y_test = data['test_images'], data['test_labels']

# Perform PCA to 10 components
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train.reshape(-1, 28 * 28))
X_test_pca = pca.transform(X_test.reshape(-1, 28 * 28))

# Define models for Project 1
models_project1 = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': GaussianNB(),
    'K-NN': KNeighborsClassifier(),
    'SVM': SVC(probability=True)
}

# Model Training and Evaluation for Project 1
for model_name, model in models_project1.items():
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)

    print(f"Project 1 - Model: {model_name}")
    print("Metrics:")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
    print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted')}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_pca), multi_class='ovr')
    print(f"RoC-AUC: {roc_auc}")
    print("\n")

# Repeat for different component sizes (15, 20, 25, 30)
for n_components in [15, 20, 25, 30]:
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train.reshape(-1, 28 * 28))
    X_test_pca = pca.transform(X_test.reshape(-1, 28 * 28))

    for model_name, model in models_project1.items():
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)

        print(f"Project 1 - Model: {model_name}, PCA Components: {n_components}")
        print("Metrics:")
        print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
        print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
        print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted')}")

        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)

        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_pca), multi_class='ovr')
        print(f"RoC-AUC: {roc_auc}")
        print("\n")

# Project 2: Toxic Tweets Dataset - NLP Problem

# Load Toxic Tweets data
df = pd.read_csv('path/to/toxic-tweets-dataset.csv')

# Text Feature Extraction for Project 2
vectorizers = {
    'Bag of Words': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

# Model Training and Evaluation for Project 2
for vectorizer_name, vectorizer in vectorizers.items():
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models for Project 2
    models_project2 = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Naive Bayes': MultinomialNB(),
        'K-NN': KNeighborsClassifier(),
        'SVM': SVC(probability=True)
    }

    for model_name, model in models_project2.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"Project 2 - Model: {model_name}, Vectorizer: {vectorizer_name}")
        print("Metrics:")
        print(f"Precision: {precision_score(y_test, y_pred)}")
        print(f"Recall: {recall_score(y_test, y_pred)}")
        print(f"F1-Score: {f1_score(y_test, y_pred)}")

        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)

        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print(f"RoC-AUC: {roc_auc}")
        print("\n")
