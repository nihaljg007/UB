import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
import base64

st.set_page_config(layout="wide", page_title="Universal Bank — Marketing Dashboard", initial_sidebar_state="expanded")

@st.cache_data
def load_default_data():
    df = pd.read_csv("UniversalBank.csv")
    df.columns = [c.strip() for c in df.columns]
    return df

def prepare_features(df):
    cols_to_drop = [c for c in df.columns if c.strip().lower()=='id' or 'zip' in c.strip().lower()]
    X = df.drop(columns=cols_to_drop+['Personal Loan'], errors='ignore')
    # locate y robustly
    if 'Personal Loan' in df.columns:
        y = df['Personal Loan']
    else:
        y = df.loc[:, df.columns.str.lower()=="personal loan"].iloc[:,0]
    return X, y

def train_models(X, y, random_state=42):
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=random_state)
    }
    results = {}
    for name, model in models.items():
        model.fit(X, y)
        results[name] = model
    return results

def metrics_table(models, X_train, X_test, y_train, y_test):
    rows = []
    for name, model in models.items():
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, zero_division=0)
        recall = recall_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_test_proba)
        cv_acc = cross_val_score(model, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), cv=5, scoring='accuracy').mean()
        rows.append({
            "Algorithm": name,
            "Train Accuracy": train_acc,
            "Test Accuracy": test_acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "AUC": auc,
            "CV(5)-Accuracy": cv_acc
        })
    return pd.DataFrame(rows).set_index("Algorithm").round(4)

def plot_roc(models, X_test, y_test):
    plt.figure(figsize=(7,6))
    colors = {'Decision Tree':'tab:blue','Random Forest':'tab:green','Gradient Boosting':'tab:red'}
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:,1]
        else:
            y_proba = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=colors.get(name))
    plt.plot([0,1],[0,1],'k--', label='Random Chance')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — All Models")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

def plot_confusion_matrix(cm, title="Confusion matrix", cmap="Blues"):
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=True, xticklabels=['Pred 0','Pred 1'], yticklabels=['True 0','True 1'])
    plt.title(title)
    st.pyplot(plt.gcf())
    plt.clf()

def download_df_as_csv(df, filename="predictions.csv"):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, file_name=filename, mime='text/csv')

st.title("Universal Bank — Marketing Intelligence Dashboard")
st.markdown("A Streamlit app to explore customer data, train tree models, and predict Personal Loan acceptance.")

menu = st.sidebar.radio("Navigation", ["Overview (EDA)","Modeling (Train & Metrics)","Predict & Download","About"])

df_default = load_default_data()

if menu == "Overview (EDA)":
    st.header("Exploratory analysis — 5 action-oriented charts")
    st.markdown("Use these charts to generate marketing actions: segmentation, targeting, and channel strategies.")

    df = df_default.copy()

    st.subheader("Filters (applied to charts)")
    col1, col2, col3 = st.columns(3)
    with col1:
        min_income = int(df['Income'].min())
        max_income = int(df['Income'].max())
        income_range = st.slider("Income range ($000)", min_income, max_income, (min_income, max_income))
    with col2:
        ed_levels = sorted(df['Education'].unique())
        sel_ed = st.multiselect("Education levels", ed_levels, default=ed_levels)
    with col3:
        family_levels = sorted(df['Family'].unique())
        sel_fam = st.multiselect("Family size", family_levels, default=family_levels)

    dff = df[(df['Income']>=income_range[0]) & (df['Income']<=income_range[1]) & (df['Education'].isin(sel_ed)) & (df['Family'].isin(sel_fam))]

    st.markdown("### 1) Income buckets — conversion rate & counts (Action: target high-conversion buckets)")
    dff['income_bucket'] = pd.qcut(dff['Income'], q=10, duplicates='drop')
    conv = dff.groupby('income_bucket')['Personal Loan'].agg(['mean','count']).reset_index().rename(columns={'mean':'conversion_rate'})
    fig, ax = plt.subplots(figsize=(10,4))
    ax2 = ax.twinx()
    sns.barplot(x='income_bucket', y='count', data=conv, ax=ax, color='tab:orange')
    sns.pointplot(x='income_bucket', y='conversion_rate', data=conv, ax=ax2, color='tab:blue')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xlabel("Income bucket (deciles)")
    ax.set_ylabel("Count")
    ax2.set_ylabel("Conversion rate")
    st.pyplot(fig)
    plt.clf()

    st.markdown("### 2) Education × Family — segment conversion heatmap (Action: segment-based offers)")
    seg = dff.groupby(['Education','Family'])['Personal Loan'].mean().unstack()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(seg, annot=True, fmt=".3f", cmap='YlGnBu', ax=ax)
    ax.set_title("Conversion rate by Education (rows) and Family (columns)")
    st.pyplot(fig)
    plt.clf()

    st.markdown("### 3) CCAvg vs Income — highlight accepted customers (Action: cross-sell credit products)")
    fig, ax = plt.subplots(figsize=(8,5))
    accepted = dff[dff['Personal Loan']==1]
    notaccepted = dff[dff['Personal Loan']==0]
    ax.scatter(notaccepted['Income'], notaccepted['CCAvg'], alpha=0.3, label='No', s=20)
    ax.scatter(accepted['Income'], accepted['CCAvg'], alpha=0.8, label='Yes', s=30, edgecolor='k')
    if len(accepted)>2:
        z = np.polyfit(accepted['Income'], accepted['CCAvg'], 1)
        p = np.poly1d(z)
        xs = np.linspace(dff['Income'].min(), dff['Income'].max(), 100)
        ax.plot(xs, p(xs), color='red', linestyle='--', label='Accepted trend')
    ax.set_xlabel("Income ($000)")
    ax.set_ylabel("CCAvg ($000/month)")
    ax.legend()
    st.pyplot(fig)
    plt.clf()

    st.markdown("### 4) Product affinity vs conversion (Action: select likely channels for offers)")
    prod_match = []
    for c in dff.columns:
        low = c.lower()
        if 'cd' in low and 'account' in low:
            prod_match.append(c)
        elif low.strip() == 'online':
            prod_match.append(c)
        elif 'credit' in low and 'card' in low:
            prod_match.append(c)
        elif 'secur' in low and 'account' in low:
            prod_match.append(c)
    prod_match = sorted(list(set(prod_match)), key=lambda x: x)
    prod_conv = {}
    for c in prod_match:
        prod_conv[c] = dff.groupby(c)['Personal Loan'].mean().to_dict()
    if prod_match:
        fig, ax = plt.subplots(figsize=(8,4))
        x = np.arange(len(prod_match))
        rates_no = [prod_conv[c].get(0,0) for c in prod_match]
        rates_yes = [prod_conv[c].get(1,0) for c in prod_match]
        ax.bar(x-0.15, rates_no, width=0.3, label='No', alpha=0.7)
        ax.bar(x+0.15, rates_yes, width=0.3, label='Yes', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(prod_match, rotation=45, ha='right')
        ax.set_ylabel("Conversion rate")
        ax.set_title("Conversion rates by product ownership (0 vs 1)")
        ax.legend()
        st.pyplot(fig)
        plt.clf()

    st.markdown("### 5) Mortgage exposure vs conversion (bubble size = Income) — identify bundle opportunities")
    fig, ax = plt.subplots(figsize=(8,5))
    sizes = (dff['Income'] - dff['Income'].min() + 1) / (dff['Income'].max() - dff['Income'].min() + 1) * 200
    sc = ax.scatter(dff['Mortgage'], dff['CCAvg'], s=sizes, c=dff['Personal Loan'], cmap='coolwarm', alpha=0.6)
    ax.set_xlabel("Mortgage ($000)")
    ax.set_ylabel("CCAvg ($000/month)")
    ax.set_title("Mortgage vs CCAvg (color=Personal Loan)")
    st.pyplot(fig)
    plt.clf()

elif menu == "Modeling (Train & Metrics)":
    st.header("Model training and performance metrics")
    st.subheader("1) Data selection")
    data_choice = st.radio("Dataset to train on", ("Use default (UniversalBank.csv)", "Upload CSV"))
    if data_choice == "Use default (UniversalBank.csv)":
        df = df_default.copy()
    else:
        uploaded = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
        else:
            st.info("Upload a CSV to proceed.")
            st.stop()

    st.write("Dataset shape:", df.shape)
    if st.button("Run training (this will train all 3 models)"):
        X, y = prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        with st.spinner("Training models..."):
            models = train_models(X_train, y_train)
        st.success("Training complete. Models are ready.")

        st.subheader("Metrics table (Train/Test, Precision, Recall, F1, AUC, CV-accuracy)")
        mt = metrics_table(models, X_train, X_test, y_train, y_test)
        st.dataframe(mt)

        st.subheader("ROC Curve (all models)")
        plot_roc(models, X_test, y_test)

        st.subheader("Confusion Matrices — Training (left) and Testing (right)")
        for name, model in models.items():
            st.markdown(f"**{name}**")
            col1, col2 = st.columns(2)
            with col1:
                cm = confusion_matrix(y_train, model.predict(X_train))
                plot_confusion_matrix(cm, title=f"{name} — Confusion Matrix (TRAIN)", cmap="Blues")
            with col2:
                cm = confusion_matrix(y_test, model.predict(X_test))
                plot_confusion_matrix(cm, title=f"{name} — Confusion Matrix (TEST)", cmap="Oranges")

        st.subheader("Feature importances")
        for name, model in models.items():
            if hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
                fi_series = pd.Series(fi, index=X.columns).sort_values(ascending=True)
                fig, ax = plt.subplots(figsize=(6,4))
                fi_series.plot(kind='barh', ax=ax)
                ax.set_title(f"Feature importances — {name}")
                st.pyplot(fig)
                plt.clf()
            else:
                st.write(f"{name} does not expose feature_importances_.")
        st.session_state['trained_models'] = models
        st.session_state['X_columns'] = X.columns.tolist()

elif menu == "Predict & Download":
    st.header("Upload new data to predict 'Personal Loan' label and download predictions")
    uploaded = st.file_uploader("Upload CSV to predict", type=['csv'])
    if uploaded is None:
        st.info("Please upload a CSV file containing the features.")
    else:
        newdf = pd.read_csv(uploaded)
        st.write("Uploaded shape:", newdf.shape)
        base = df_default.copy()
        X_train, y_train = prepare_features(base)
        models = train_models(X_train, y_train)
        cols_to_drop = [c for c in newdf.columns if c.strip().lower()=='id' or 'zip' in c.strip().lower()]
        X_pred = newdf.drop(columns=cols_to_drop+['Personal Loan'], errors='ignore')
        for c in X_train.columns:
            if c not in X_pred.columns:
                X_pred[c] = 0
        X_pred = X_pred[X_train.columns]
        model_choice = st.selectbox("Choose model for prediction", list(models.keys()))
        model = models[model_choice]
        proba = model.predict_proba(X_pred)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_pred)
        label = (proba >= 0.5).astype(int)
        out = newdf.copy()
        out['Personal Loan (pred_proba)'] = np.round(proba,4)
        out['Personal Loan (pred_label)'] = label
        st.dataframe(out.head(20))
        csv = out.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions as CSV", csv, file_name="predictions_with_label.csv", mime='text/csv')

else:
    st.header("About this app")
    st.markdown("""
    ### Universal Bank — Marketing Dashboard
    This Streamlit dashboard helps marketing teams make data-driven decisions.

    **Capabilities:**
    - 📊 EDA: Explore 5 action-oriented charts for targeting and segmentation.
    - 🤖 Modeling: Train Decision Tree, Random Forest, and Gradient Boosting models.
    - 📈 Metrics: Compare accuracies, ROC curves, confusion matrices, and feature importances.
    - 📂 Predictions: Upload new data, generate predicted labels, and download results.

    **Created for:** Marketing Heads and Analysts at Universal Bank  
    **Developed with:** Streamlit + Scikit-learn + Matplotlib + Seaborn
    """)
