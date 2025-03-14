import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import statsmodels.api as sm


def filter_genes_by_expression(data, threshold=0.1):
    """Filter genes by expression level."""
    min_expressed_cells = int(threshold * data.shape[0])  # Number of cells required
    # Define expression criterion: a gene is considered "expressed" if its value is greater than 0
    mask = (data.X > 0.0).sum(axis=0) >= min_expressed_cells
    # Filter genes
    filtered_data = data[:, mask]

    return filtered_data


def filter_cells_by_anaotation(data, metadata, cluster_name, annotation_label):
    """Filter cells by label, corresponding to metadata.
    Input:
        data: AnnData object
        metadata: pd.DataFrame
    
    """
    new_metadata = metadata[metadata[cluster_name]==annotation_label]
    cells_label = [l for l in new_metadata.index]
    cells = [cell for cell in cells_label if cell in data.obs_names]
    new_data = data[cells]
    return new_data, new_metadata


def filter_cells_by_dict(adata, metadata, filter_dict):
    """Filter cells by label, corresponding to metadata.
    Input:
        adata: AnnData object
        metadata: pd.DataFrame
    """
    new_adata = adata
    new_metadata = metadata
    for key, value in filter_dict.items():
        new_adata, new_metadata = filter_cells_by_anaotation(new_adata, new_metadata, key, value)
    return new_adata, new_metadata


def print_column_info(df):
    
    for c in df.columns:
        grouped = df[[c]].groupby(c).count()
        members = ''
        if len(grouped) < 30:
            members = str(list(grouped.index))
        print("Number of unique %s = %d %s" % (c, len(grouped), members))


def name_with_index(query_names, name_list):
    query_names_with_index = []
    for query in query_names:
        for name in name_list:
            if query in name:
                query_names_with_index.append(name)

    return query_names_with_index


def assign_binary_regression_labels(adata, metadata, assign_list):
    """Assign regression labels to the metadata.
    Input:
        adata: AnnData object
        metadata: pd.DataFrame
        assign_list: list of strings in metadata column names
    """
    y = np.zeros((adata.shape[0], len(assign_list)))
    for i, assign in enumerate(assign_list):
        if metadata[assign].astype(str).str.isnumeric().all():
            y[:, i] = metadata.reindex(adata.obs.index)['assign']
        else: 
            y[:, i] = pd.factorize(metadata[assign])[0]

    adata.obs['regression_labels'] = y
    return adata


def order_loadings_by_shap(shap_values):
    # Compute mean absolute SHAP value per feature
    shap_importance = np.abs(shap_values).mean(axis=0)

    # Get sorted indices (most important features first)
    sorted_indices = np.argsort(-shap_importance)  # Negative sign for descending order

    return sorted_indices


def extract_top_genes_from_loadings(adata, loadings, n_genes=1000): # todo
    """Extract top genes from PCA or NMF loadings.
    Input:
        adata: AnnData object
        loadings: np.array, shape (n_genes, )
        n_genes: int, number of genes to extract
    """

    # Get gene names
    genes = adata.var['gene_symbol']

    # Create a DataFrame with gene names and their PC loadings
    df = pd.DataFrame({"gene": genes, "loading": loadings})

    # Sort by absolute loading values (descending)
    df_sorted = df.reindex(df["loading"].abs().sort_values(ascending=False).index)
    top_genes = df_sorted['gene'][:n_genes]

    return top_genes


def is_in_degenes(query_genes, degenes):
    """Check if query genes are in DE genes."""
    query_genes = set(query_genes)
    degenes = set(degenes)
    return query_genes.intersection(degenes)


def detect_degenes_accumulated(adata, loading_idx, degenes, n_genes_per_pc=1000):
    
    degenes = set(degenes)
    total_genes = set()
    
    degenes_detected_among_all = []
    degenes_detected_among_extracted = []
    for i in loading_idx:
        top_genes = extract_top_genes_from_loadings(adata, adata.varm["PCs"][:, i], n_genes=n_genes_per_pc)
        total_genes = total_genes.union(set(top_genes))
        degenes_detected = is_in_degenes(total_genes, degenes)
        degenes_detected_among_all.append(len(degenes_detected)/len(degenes))
        degenes_detected_among_extracted.append(len(degenes_detected)/len(total_genes))
    return degenes_detected_among_all, degenes_detected_among_extracted


# define function to plot the degene-detected results
def plot_degene_trend(n_selected_features, 
                      degenes_detected_among_all, 
                      degenes_detected_among_extracted, 
                      acc,
                      subclass_type,
                      subclass_name):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(range(1, n_selected_features+1), degenes_detected_among_all, label='# DE genes / all DE genes')
    ax.plot(range(1, n_selected_features+1), degenes_detected_among_extracted, label='# DE genes / selected genes')
    ax.set_xlabel('Number of features selected')
    ax.set_ylabel('Genes detected ratio')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title(f"{subclass_name}: {subclass_type}, Classification accuracy: {acc:.2f}", fontsize=8)
    plt.tight_layout()
    plt.show()


def shap_importance_analysis(lasso_logreg, adata, max_display=10):
    # Create a SHAP explainer
    X = adata.obsm['X_pca']
    X_with_intercept = sm.add_constant(X)
    explainer = shap.LinearExplainer(lasso_logreg, X_with_intercept)

    feature_names = ['PC{}'.format(i+1) for i in range(X.shape[1])]
    feature_names = ['intercept'] + feature_names

    # Get SHAP values for the training data
    shap_values = explainer.shap_values(X_with_intercept)

    # Plot summary of SHAP values (global feature importance)
    shap.summary_plot(
        shap_values, X_with_intercept, feature_names=feature_names, max_display=max_display, plot_size=(4, 4)
        )
    return shap_values
