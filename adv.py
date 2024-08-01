import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from statsmodels.formula.api import ols

# Set up the app
st.title("Descriptive Statistics and Advanced Analytics by [SumanEcon]")

# File uploader
uploaded_file = st.file_uploader("Upload your data file (CSV or Excel or xls)", type=['csv', 'xlsx','xls'])

# Function to read file
def read_file(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
        else:
            st.error("Unsupported file format")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# Function to compute CAGR and p-value
def compute_cagr(data, column):
    data = data[[column]].dropna().reset_index(drop=True)
    data['Time'] = np.arange(1, len(data) + 1)
    data['LogColumn'] = np.log(data[column])
    model = ols('LogColumn ~ Time', data=data).fit()
    cagr = (np.exp(model.params['Time']) - 1) * 100
    return cagr, model.pvalues['Time'], model.rsquared_adj

# Function to compute CDVI
def compute_cdvi(cv, adj_r_squared):
    return cv * np.sqrt(1 - adj_r_squared)

# Function to compute correlation p-values
def correlation_p_values(df):
    corr_matrix = df.corr()
    p_values = pd.DataFrame(np.ones_like(corr_matrix), columns=corr_matrix.columns, index=corr_matrix.index)
    for i in range(len(corr_matrix.columns)):
        for j in range(i, len(corr_matrix.columns)):
            if i != j:
                _, p_value = stats.pearsonr(df.iloc[:, i], df.iloc[:, j])
                p_values.iloc[i, j] = p_values.iloc[j, i] = p_value
    return p_values

# Function to plot heatmap
def plot_heatmap(matrix, title):
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

# Function to plot correlation circle
def plot_correlation_circle(pca, columns):
    fig, ax = plt.subplots()
    for i, v in enumerate(columns):
        ax.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], head_width=0.05, head_length=0.05, color='b')
        ax.text(pca.components_[0, i]*1.1, pca.components_[1, i]*1.1, v, color='r')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Correlation Circle")
    st.pyplot(fig)

if uploaded_file:
    df = read_file(uploaded_file)
    if df is not None:
        st.write("### Data Preview")
        st.write(df.head())

        # Descriptive Statistics
        st.subheader("Descriptive Statistics")
        numeric_df = df.select_dtypes(include=np.number).loc[:, ~df.columns.str.contains('Year', case=False)]
        descriptive_stats = numeric_df.describe().T
        descriptive_stats['Mode'] = numeric_df.mode().iloc[0]
        descriptive_stats['Variance'] = numeric_df.var()
        descriptive_stats['Standard Deviation'] = numeric_df.std()
        descriptive_stats['Skewness'] = numeric_df.skew()
        descriptive_stats['Kurtosis'] = numeric_df.kurt()
        cagr_results, p_value_results, adj_r_squared_results = zip(*[compute_cagr(numeric_df, col) for col in numeric_df.columns])
        cv_results = numeric_df.apply(lambda col: (col.std() / col.mean()) * 100)
        cdvi_results = [compute_cdvi(cv, adj_r_squared) for cv, adj_r_squared in zip(cv_results, adj_r_squared_results)]
        descriptive_stats['CAGR (%)'] = cagr_results
        descriptive_stats['P-Value (CAGR)'] = p_value_results
        descriptive_stats['CV (%)'] = cv_results
        descriptive_stats['Adjusted R Squared'] = adj_r_squared_results
        descriptive_stats['CDVI'] = cdvi_results
        descriptive_stats = descriptive_stats.round(3)
        basic_stats = pd.DataFrame({
            'Column': numeric_df.columns,
            'Data Type': numeric_df.dtypes,
            'Missing Values': numeric_df.isnull().sum()
        }).set_index('Column').join(descriptive_stats).reset_index()
        st.write(basic_stats)

        # Correlation Analysis
        st.subheader("Correlation Analysis")
        corr_matrix = numeric_df.corr()
        st.write("**Correlation Matrix:**")
        st.write(corr_matrix)
        plot_heatmap(corr_matrix, "Correlation Heatmap")
        p_values = correlation_p_values(numeric_df)
        st.write("**Correlation P-values:**")
        st.write(p_values)
        plot_heatmap(p_values, "Correlation P-value Heatmap")
        significance_level = st.slider("Select significance level (alpha)", 0.01, 0.1, 0.05)
        significant_corrs = corr_matrix[p_values < significance_level]
        st.write(f"Significant correlations with p-value < {significance_level}:")
        st.write(significant_corrs)

        # Pairwise Scatter Plots
        st.subheader("Pairwise Scatter Plots")
        if len(numeric_df.columns) > 1:
            fig = sns.pairplot(numeric_df)
            st.pyplot(fig)
        else:
            st.write("Not enough quantitative variables to generate pairwise scatter plots.")

        # Principal Component Analysis (PCA)
        st.subheader("Principal Component Analysis (PCA)")
        n_components = st.slider("Select number of PCA components", 1, min(len(numeric_df.columns), 10), 2)
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(numeric_df.fillna(0))
        pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
        st.write(pca_df)
        explained_variance = pca.explained_variance_ratio_
        fig, ax = plt.subplots()
        ax.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-')
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance")
        ax.set_title("Scree Plot")
        st.pyplot(fig)
        plot_correlation_circle(pca, numeric_df.columns)

        # Clustered Heatmap
        st.subheader("Clustered Heatmap")
        sns.clustermap(corr_matrix, annot=True, cmap='coolwarm')
        st.pyplot()
