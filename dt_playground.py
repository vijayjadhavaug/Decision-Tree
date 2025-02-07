import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Streamlit app
st.title("Decision Tree Hyperparameter Tuning")

# Sidebar for parameters
st.sidebar.header("Hyperparameter Tuning")
criterion = st.sidebar.radio("Criterion", ["gini", "entropy"], index=0)
splitter = st.sidebar.radio("Splitter", ["best", "random"], index=0)
max_depth = st.sidebar.number_input("Max Depth", 0, 20, 0, step=1)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 200, 2)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 200, 1)
max_features = st.sidebar.slider("Max Features", 1, 10, 5)
max_leaf_nodes = st.sidebar.number_input("Max Leaf Nodes", 0, 50, 0, step=1)
min_impurity_decrease = st.sidebar.number_input("Min Impurity Decrease", 0.0, 1.0, 0.0, step=0.01)

# Convert 0 values to None for parameters where 0 is not allowed
max_depth = None if max_depth == 0 else max_depth
max_leaf_nodes = None if max_leaf_nodes == 0 else max_leaf_nodes

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())
    
    # Assuming the last column is the target variable
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Plot dataset
    st.subheader("Dataset Scatter Plot")
    if X.shape[1] >= 2:
        sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y)
        st.pyplot(plt)
    else:
        st.write("Cannot plot scatter plot. Need at least two features.")
    
    # Run Algorithm button
    if st.sidebar.button("Run Algorithm"):
        clf = DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features if max_features <= X.shape[1] else None,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            random_state=42
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.subheader("Model Accuracy")
        st.write(f"Accuracy: {accuracy:.2f}")
        
        # Plot decision tree
        st.subheader("Decision Tree Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_tree(clf, filled=True, feature_names=X.columns, class_names=[str(c) for c in set(y)])
        st.pyplot(fig)
