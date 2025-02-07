import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load dataset
def load_data():
    df = pd.read_csv("decision_tree_dataset.csv")
    return df

def split_data(df):
    X = df.drop(columns=["Purchase"])
    y = df["Purchase"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def calculate_information_gain(X_train, y_train, model):
    feature_importances = model.feature_importances_
    feature_names = X_train.columns
    info_gain = {feature: importance for feature, importance in zip(feature_names, feature_importances)}
    return info_gain

def train_model(X_train, y_train, params):
    model = DecisionTreeClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, report, y_pred

def plot_decision_tree(model, feature_names, info_gain):
    plt.figure(figsize=(12, 6))
    plot_tree(model, feature_names=feature_names, filled=True, class_names=["No", "Yes"])
    plt.title("Decision Tree Visualization")
    st.pyplot(plt)
    
    # Display Information Gain values
    st.subheader("Feature Importance (Information Gain)")
    info_gain_df = pd.DataFrame(info_gain.items(), columns=['Feature', 'Information Gain']).sort_values(by='Information Gain', ascending=False)
    st.table(info_gain_df)

def plot_data_changes(df, X_train, y_train, X_test, y_test, model):
    # Select only the first two features for visualization
    X_train_vis = X_train.iloc[:, :2]
    X_test_vis = X_test.iloc[:, :2]

    # Train a new DecisionTreeClassifier with only two features
    vis_model = DecisionTreeClassifier(max_depth=model.get_params()["max_depth"], random_state=42)
    vis_model.fit(X_train_vis, y_train)

    # Generate a mesh grid for decision boundary
    x_min, x_max = X_train_vis.iloc[:, 0].min() - 1, X_train_vis.iloc[:, 0].max() + 1
    y_min, y_max = X_train_vis.iloc[:, 1].min() - 1, X_train_vis.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1.0), np.arange(y_min, y_max, 1.0))

    # Predict on the grid
    Z = vis_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    
    # Scatter plot actual data
    sns.scatterplot(x=X_train_vis.iloc[:, 0], y=X_train_vis.iloc[:, 1], hue=y_train, edgecolor="k", ax=ax)
    
    plt.xlabel(X_train_vis.columns[0])
    plt.ylabel(X_train_vis.columns[1])
    plt.title("Decision Boundary of Decision Tree")

    st.pyplot(fig)


def main():
    st.title("Decision Tree Classifier with Configurable Parameters")
    df = load_data()
    st.sidebar.header("Model Parameters")
    criterion = st.sidebar.radio("Criterion", ["gini", "entropy"], index=0)
    splitter = st.sidebar.radio("Splitter", ["best", "random"], index=0)
    max_depth = st.sidebar.number_input("Max Depth", 0, 20, 0, step=1) or None
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 200, 2)
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 200, 1)
    max_features = st.sidebar.slider("Max Features", 1, df.shape[1]-1, 5)
    max_leaf_nodes = st.sidebar.number_input("Max Leaf Nodes", 0, 50, 0, step=1) or None
    min_impurity_decrease = st.sidebar.number_input("Min Impurity Decrease", 0.0, 1.0, 0.0, step=0.01)
    
    X_train, X_test, y_train, y_test = split_data(df)
    
    if st.button("Run Algorithm"):
        params = {
            "criterion": criterion,
            "splitter": splitter,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "max_leaf_nodes": max_leaf_nodes,
            "min_impurity_decrease": min_impurity_decrease
        }
        
        model = train_model(X_train, y_train, params)
        accuracy, report, y_pred = evaluate_model(model, X_test, y_test)
        info_gain = calculate_information_gain(X_train, y_train, model)
        
        st.subheader("Model Performance")
        st.write(f"Accuracy: {accuracy:.2f}")
        
        st.subheader("Classification Report")
        st.table(pd.DataFrame(report).transpose())
        
        st.subheader("Decision Tree Structure")
        plot_decision_tree(model, X_train.columns, info_gain)
        
        st.subheader("Dataset Visualization Before & After Classification")
        #plot_data_changes(df, X_train, y_train, X_test, y_test, y_pred)
        plot_data_changes(df, X_train, y_train, X_test, y_test, model)  # Pass the trained model, not y_pred

if __name__ == "__main__":
    main()
