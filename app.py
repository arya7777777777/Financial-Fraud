import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Financial Fraud Dashboard", layout="wide")
st.title("Financial Fraud Detection Dashboard")

# ✅ Load data from the same folder as app.py
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_fraud_survey.csv")
    return df

df = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
tabs = [
    "Data Visualization",
    "Classification",
    "Clustering",
    "Association Rule Mining",
    "Regression"
]
choice = st.sidebar.radio("Go to", tabs)

# Helper function for encoding categorical columns
le = LabelEncoder()
def encode_column(col):
    return le.fit_transform(col.astype(str))

# ✅ Data Visualization
if choice == "Data Visualization":
    st.subheader("Descriptive Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Age"], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("Income Distribution")
        fig, ax = plt.subplots()
        sns.boxplot(x=df["AnnualIncomeUSD"], ax=ax)
        st.pyplot(fig)

    st.write("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ✅ Classification
elif choice == "Classification":
    st.subheader("Classification Models")
    st.write("Train and evaluate multiple classifiers.")

    target = st.selectbox("Select target variable (must be categorical)", df.columns)
    if target:
        X = df.select_dtypes(include=np.number)
        y = encode_column(df[target])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        models = {
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        results = []
        y_probs = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:,1]
                y_probs[name] = y_prob
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            results.append((name, acc, prec, rec, f1))

        st.write("Performance Metrics")
        st.table(pd.DataFrame(
            results,
            columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"]
        ))

        selected_model = st.selectbox("Select model to show confusion matrix", list(models.keys()))
        if selected_model:
            model = models[selected_model]
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            st.write("Confusion Matrix")
            st.write(pd.DataFrame(cm))

        # ROC Curve
        st.write("ROC Curve (for models supporting probabilities)")
        fig, ax = plt.subplots()
        for name, y_prob in y_probs.items():
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        ax.plot([0,1], [0,1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)

# ✅ Clustering (skeleton)
elif choice == "Clustering":
    st.subheader("Clustering (K-Means)")
    st.write("This section will allow you to cluster data.")

# ✅ Association Rule Mining (skeleton)
elif choice == "Association Rule Mining":
    st.subheader("Association Rule Mining")
    st.write("This section will show frequent itemsets.")

# ✅ Regression (skeleton)
elif choice == "Regression":
    st.subheader("Regression Analysis")
    st.write("This section will show regression models.")
