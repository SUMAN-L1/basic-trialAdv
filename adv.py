import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from statsmodels.formula.api import ols

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
    
    cagr = (np.exp(model.params['Time']) - 1) * 100  # Convert to percentage
    p_value = model.pvalues['Time']
    adj_r_squared = model.rsquared_adj
    
    return cagr, p_value, adj_r_squared

# Function to compute CDVI
def compute_cdvi(cv, adj_r_squared):
    return cv * np.sqrt(1 - adj_r_squared)

# Function to compute outliers
def compute_outliers(column):
    z_scores = np.abs(stats.zscore(column.dropna()))
    return np.sum(z_scores > 3)

# Function to calculate correlation p-values
def correlation_p_values(df):
    corr_matrix = df.corr()
    p_values = pd.DataFrame(np.ones_like(corr_matrix), columns=corr_matrix.columns, index=corr_matrix.index)
    for i in range(len(corr_matrix.columns)):
        for j in range(i, len(corr_matrix.columns)):
            if i != j:
                _, p_value = stats.pearsonr(df.iloc[:, i], df.iloc[:, j])
                p_values.iloc[i, j] = p_values.iloc[j, i] = p_value
    return p_values

# Function to plot bubble matrix
def plot_bubble_matrix(p_values):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=p_values.columns, y=p_values.index, size=p_values.values.flatten(), sizes=(20, 200), hue=p_values.values.flatten(), palette='coolwarm', legend=None, ax=ax)
    plt.title('Bubble Matrix of Correlation P-Values')
    plt.xticks(rotation=90)
    return fig

# Function to plot contour plot
def plot_contour(p_values):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.kdeplot(data=p_values.values.flatten(), cmap='coolwarm', fill=True, ax=ax)
    plt.title('Contour Plot of Correlation P-Values')
    return fig

# Function to plot dot plot
def plot_dot_plot(p_values):
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(p_values.shape[0]):
        ax.plot(p_values.columns, p_values.iloc[i], 'o', label=p_values.index[i])
    plt.xticks(rotation=90)
    plt.title('Dot Plot of Correlation P-Values')
    plt.xlabel('Variables')
    plt.ylabel('P-Values')
    plt.legend()
    return fig

# If a file is uploaded
if uploaded_file:
    df = read_file(uploaded_file)
    if df is not None:
        st.write("### Data Preview")
        st.write(df.head())

        # Descriptive Statistics
        st.subheader("Descriptive Statistics")
        # Filter out qualitative columns and 'Year' column
        numeric_df = df.select_dtypes(include=np.number)
        numeric_df = numeric_df.loc[:, ~numeric_df.columns.str.contains('Year', case=False)]
        
        # Compute basic statistics
        descriptive_stats = numeric_df.describe().T
        descriptive_stats['Mode'] = numeric_df.mode().iloc[0]
        descriptive_stats['Variance'] = numeric_df.var()
        descriptive_stats['Standard Deviation'] = numeric_df.std()
        descriptive_stats['Skewness'] = numeric_df.skew()
        descriptive_stats['Kurtosis'] = numeric_df.kurt()
      
        # Compute CAGR and related metrics
        cagr_results = numeric_df.apply(lambda col: compute_cagr(numeric_df, col.name)[0])
        cv_results = numeric_df.apply(lambda col: (col.std() / col.mean()) * 100)
        p_value_results = numeric_df.apply(lambda col: compute_cagr(numeric_df, col.name)[1])
        adj_r_squared_results = numeric_df.apply(lambda col: compute_cagr(numeric_df, col.name)[2])
        cdvi_results = numeric_df.apply(lambda col: compute_cdvi((numeric_df[col.name].std() / numeric_df[col.name].mean()) * 100, compute_cagr(numeric_df, col.name)[2]))
        
        descriptive_stats['CAGR (%)'] = cagr_results
        descriptive_stats['P-Value (CAGR)'] = p_value_results
        descriptive_stats['CV (%)'] = cv_results
        descriptive_stats['Adjusted R Squared'] = adj_r_squared_results
        descriptive_stats['CDVI'] = cdvi_results
        
        # Round all statistics to three decimal places
        descriptive_stats = descriptive_stats.round(3)
        
        # Final descriptive statistics table
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

        st.write("**Correlation Heatmap:**")
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        # Calculate p-values for correlation matrix
        p_values = correlation_p_values(numeric_df)

        st.write("**Correlation P-values:**")
        st.write(p_values)

        st.write("**Correlation P-value Heatmap:**")
        fig, ax = plt.subplots()
        sns.heatmap(p_values, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        st.write("**Correlation P-value Bubble Matrix:**")
        fig = plot_bubble_matrix(p_values)
        st.pyplot(fig)

        st.write("**Correlation P-value Contour Plot:**")
        fig = plot_contour(p_values)
        st.pyplot(fig)

        st.write("**Correlation P-value Dot Plot:**")
        fig = plot_dot_plot(p_values)
        st.pyplot(fig)

        st.write("**Significant Correlations:**")
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

        st.write("**Scree Plot:**")
        explained_variance = pca.explained_variance_ratio_
        fig, ax = plt.subplots()
        ax.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-')
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance")
        st.pyplot(fig)

        st.write("**Correlation Circle:**")
        pca_components = pca.components_
        fig, ax = plt.subplots()
        for i, v in enumerate(numeric_df.columns):
            ax.arrow(0, 0, pca_components[0, i], pca_components[1, i], head_width=0.05, head_length=0.05, color='b')
            ax.text(pca_components[0, i]*1.1, pca_components[1, i]*1.1, v, color='r')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Correlation Circle")
        st.pyplot(fig)

        # Clustered Heatmap
        st.subheader("Clustered Heatmap")
        fig = sns.clustermap(corr_matrix, annot=True, cmap='coolwarm')
        st.pyplot(fig)
