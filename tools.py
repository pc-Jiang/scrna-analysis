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