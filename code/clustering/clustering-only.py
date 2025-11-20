import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os
import glob
from pathlib import Path


def load_question_text_mapping(csv_path):
    """
    Load question text mapping from CSV file.
    
    Parameters:
    csv_path (str): Path to CSV file with columns including: question_number, question_text
    
    Returns:
    dict: Dictionary mapping question_number to question_text
    """
    try:
        df = pd.read_csv(csv_path)
        question_mapping = dict(zip(df['question_number'], df['question_text']))
        print(f"Loaded question text for {len(question_mapping)} items")
        return question_mapping
    except Exception as e:
        print(f"Warning: Could not load question text mapping from {csv_path}: {e}")
        return {}


def find_model_files(data_directory, construct_name):
    """
    Find similarity CSV files for a specific construct in the data directory.
    
    Parameters:
    data_directory (str): Path to directory containing construct subdirectories
    construct_name (str): Name of the construct to find files for
    
    Returns:
    dict: Dictionary mapping model names to file paths
    """
    model_files = {}
    data_path = Path(data_directory)
    
    # Look for the construct-specific subdirectory
    construct_subdir = None
    for subdir in data_path.iterdir():
        if subdir.is_dir() and construct_name.lower() in subdir.name.lower():
            construct_subdir = subdir
            break
    
    if construct_subdir is None:
        print(f"No subdirectory found for construct '{construct_name}' in {data_directory}")
        return model_files
    
    print(f"Found construct directory: {construct_subdir}")
    
    # Look for similarity files in the construct subdirectory
    similarity_files = list(construct_subdir.glob("*similarity*.csv"))
    
    for file_path in similarity_files:
        # Extract model name from filename
        filename = file_path.stem  # Remove .csv extension
        
        # Try to extract model name (everything before the construct name)
        parts = filename.split('_')
        if len(parts) >= 2:
            # Find where the construct name appears and take everything before it
            model_name_parts = []
            for part in parts:
                if construct_name.lower() in part.lower():
                    break
                model_name_parts.append(part)
            
            if model_name_parts:
                model_name = '_'.join(model_name_parts)
                model_files[model_name] = str(file_path)
            else:
                # Fallback: use the first part as model name
                model_name = parts[0]
                model_files[model_name] = str(file_path)
    
    return model_files


def similarity_to_distance(similarity_matrix):
    """Convert similarity matrix to a distance matrix"""
    # Convert similarities (0-1) to distances (0-1)
    # Higher similarity = lower distance
    return 1 - similarity_matrix


def find_optimal_clusters(distance_matrix, max_clusters=None):
    """
    Find optimal number of clusters using multiple metrics.
    
    Parameters:
    distance_matrix (numpy.ndarray): Distance matrix
    max_clusters (int): Maximum number of clusters to test
    
    Returns:
    dict: Dictionary with optimal cluster numbers for different metrics
    """
    n_samples = distance_matrix.shape[0]
    
    if max_clusters is None:
        max_clusters = min(10, n_samples - 1)
    
    cluster_range = range(2, min(max_clusters + 1, n_samples))
    
    # Convert distance matrix to embeddings for K-means
    mds = MDS(n_components=min(n_samples-1, 10), dissimilarity='precomputed', random_state=42)
    embeddings = mds.fit_transform(distance_matrix)
    
    silhouette_scores = []
    calinski_scores = []
    davies_bouldin_scores = []
    inertias = []
    
    for n_clusters in cluster_range:
        # K-means clustering on MDS embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate metrics
        sil_score = silhouette_score(embeddings, cluster_labels)
        cal_score = calinski_harabasz_score(embeddings, cluster_labels)
        db_score = davies_bouldin_score(embeddings, cluster_labels)
        
        silhouette_scores.append(sil_score)
        calinski_scores.append(cal_score)
        davies_bouldin_scores.append(db_score)
        inertias.append(kmeans.inertia_)
    
    # Find optimal clusters for each metric
    optimal_silhouette = cluster_range[np.argmax(silhouette_scores)]
    optimal_calinski = cluster_range[np.argmax(calinski_scores)]
    optimal_davies_bouldin = cluster_range[np.argmin(davies_bouldin_scores)]
    
    # Find elbow for inertia (simple approach)
    # Calculate the rate of change
    if len(inertias) > 2:
        diffs = np.diff(inertias)
        diff2 = np.diff(diffs)
        if len(diff2) > 0:
            optimal_elbow = cluster_range[np.argmax(diff2) + 2]
        else:
            optimal_elbow = optimal_silhouette
    else:
        optimal_elbow = optimal_silhouette
    
    return {
        'silhouette': optimal_silhouette,
        'calinski_harabasz': optimal_calinski,
        'davies_bouldin': optimal_davies_bouldin,
        'elbow': optimal_elbow,
        'silhouette_scores': list(zip(cluster_range, silhouette_scores)),
        'calinski_scores': list(zip(cluster_range, calinski_scores)),
        'davies_bouldin_scores': list(zip(cluster_range, davies_bouldin_scores)),
        'inertias': list(zip(cluster_range, inertias))
    }


def export_cluster_assignments(linkage_matrix, labels, optimal_clusters_info, question_text_mapping, output_path, model_name, construct_name):
    """
    Export cluster assignments to a single consolidated CSV file per construct.
    
    Parameters:
    linkage_matrix: Hierarchical clustering linkage matrix
    labels: List of question IDs
    optimal_clusters_info: Dictionary with optimal cluster numbers for different metrics
    question_text_mapping: Dictionary mapping question IDs to question text
    output_path: Path to save CSV files
    model_name: Name of the model
    construct_name: Name of the construct
    """
    
    # Create base DataFrame with question information
    cluster_data = []
    for i, question_id in enumerate(labels):
        question_text = question_text_mapping.get(question_id, "Question text not found")
        
        row_data = {
            'question_id': question_id,
            'question_text': question_text
        }
        
        # Add cluster assignments for each optimal number
        metrics_to_export = ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'elbow']
        
        for metric in metrics_to_export:
            n_clusters = optimal_clusters_info[metric]
            cluster_assignments = hierarchy.fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Add column for this metric's optimal clustering
            column_name = f'{metric}_optimal_{n_clusters}clusters'
            row_data[column_name] = cluster_assignments[i]
        
        cluster_data.append(row_data)
    
    # Convert to DataFrame
    cluster_df = pd.DataFrame(cluster_data)
    
    # Reorder columns to have question_id, question_text first, then cluster columns
    cols = ['question_id', 'question_text'] + [col for col in cluster_df.columns if col.endswith('clusters')]
    cluster_df = cluster_df[cols]
    
    # Save consolidated CSV
    filename = f"{model_name}_{construct_name}_all_cluster_assignments.csv"
    csv_path = output_path / filename
    cluster_df.to_csv(csv_path, index=True)  # Keep index for row numbers
    
    print(f"Saved all cluster assignments: {filename}")
    
    # Print preview of the results
    print(f"\nPreview of cluster assignments for {model_name} - {construct_name}:")
    print(cluster_df.head().to_string())
    print(f"... (showing first 5 of {len(cluster_df)} items)\n")


def create_individual_metric_plots(optimal_clusters_results, construct_name, construct_output_dir, model_colors=None):
    """
    Create individual plots for each clustering evaluation metric.
    
    Parameters:
    optimal_clusters_results: Dictionary containing optimal cluster results for all models
    construct_name: Name of the construct
    construct_output_dir: Output directory path
    model_colors: Dictionary mapping model names to colors for consistency
    """
    
    # Define metrics and their properties
    metrics_info = [
        {
            'metric_key': 'silhouette_scores',
            'optimal_key': 'silhouette',
            'title': 'Silhouette Score',
            'subtitle': '(higher is better)',
            'filename': 'silhouette_score'
        },
        {
            'metric_key': 'calinski_scores',
            'optimal_key': 'calinski_harabasz',
            'title': 'Calinski-Harabasz Score',
            'subtitle': '(higher is better)',
            'filename': 'calinski_harabasz_score'
        },
        {
            'metric_key': 'davies_bouldin_scores',
            'optimal_key': 'davies_bouldin',
            'title': 'Davies-Bouldin Score',
            'subtitle': '(lower is better)',
            'filename': 'davies_bouldin_score'
        },
        {
            'metric_key': 'inertias',
            'optimal_key': 'elbow',
            'title': 'Inertia (Elbow Method)',
            'subtitle': '(look for elbow point)',
            'filename': 'inertia_elbow'
        }
    ]
    
    for metric_info in metrics_info:
        # Create individual plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot each model's scores with consistent colors
        for model_name, optimal_result in optimal_clusters_results.items():
            scores = optimal_result[metric_info['metric_key']]
            x_vals, y_vals = zip(*scores)
            
            # Use consistent color if provided
            color = model_colors.get(model_name) if model_colors else None
            ax.plot(x_vals, y_vals, marker='o', label=model_name, linewidth=2, markersize=6, color=color)
            
            # Mark optimal point
            optimal_x = optimal_result[metric_info['optimal_key']]
            
            # Find corresponding y value
            optimal_y = next(y for x, y in scores if x == optimal_x)
            ax.scatter([optimal_x], [optimal_y], s=120, marker='*', 
                      color='red', alpha=0.9, zorder=5, edgecolors='black', linewidth=0.5)
        
        # Customize plot
        ax.set_xlabel('Number of Clusters', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{metric_info["title"]} - {construct_name}\n{metric_info["subtitle"]}', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Place legend outside the plot area
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Adjust layout to accommodate legend
        plt.tight_layout()
        
        # Save the plot
        filename = f'{construct_name}_{metric_info["filename"]}.png'
        plt.savefig(construct_output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved individual metric plot: {filename}")


def process_construct(construct_name, data_directory, output_dir, question_text_mapping=None, linkage_method='average', global_model_colors=None):
    """
    Process unsupervised clustering analysis for a single construct.
    
    Parameters:
    construct_name (str): Name of the construct
    data_directory (str): Path to directory containing construct subdirectories
    output_dir (str): Directory to save output files
    question_text_mapping (dict): Dictionary mapping question IDs to question text
    linkage_method (str): Linkage method for hierarchical clustering
    global_model_colors (dict): Dictionary mapping model names to colors for consistency across constructs
    """
    print(f"\n{'='*60}")
    print(f"Processing construct: {construct_name}")
    print(f"{'='*60}")
    
    # Find model files for this specific construct
    model_files = find_model_files(data_directory, construct_name)
    print(f"Found {len(model_files)} models: {list(model_files.keys())}")
    
    if not model_files:
        print(f"No model files found for construct {construct_name}")
        return
    
    # Create output directory for this construct
    construct_output_dir = Path(output_dir) / construct_name
    construct_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each model
    clustering_results = {}
    mds_embeddings = {}
    optimal_clusters_results = {}
    
    for model_name, file_path in model_files.items():
        try:
            print(f"\nProcessing {model_name} ({file_path})...")
            
            # Read CSV - the first column will be the index
            df = pd.read_csv(file_path, index_col=0)
            print(f"Loaded similarity matrix: {df.shape}")
            print(f"Questions: {list(df.index)}")
            
            # Process similarity matrix
            similarity_matrix = df.values
            distance_matrix = similarity_to_distance(similarity_matrix)
            
            # Ensure distance matrix is symmetric
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            
            # Create labels from dataframe index
            labels = df.index.tolist()
            
            # Compute linkage for hierarchical clustering
            condensed_distance = squareform(distance_matrix)
            linkage_matrix = hierarchy.linkage(condensed_distance, method=linkage_method)
            
            # Find optimal number of clusters
            optimal_clusters = find_optimal_clusters(distance_matrix)
            optimal_clusters_results[model_name] = optimal_clusters
            
            # Export cluster assignments to CSV
            if question_text_mapping:
                export_cluster_assignments(
                    linkage_matrix, labels, optimal_clusters, 
                    question_text_mapping, construct_output_dir, 
                    model_name, construct_name
                )
            
            # Save results
            clustering_results[model_name] = {
                'linkage': linkage_matrix,
                'labels': labels,
                'df': df,
                'distance_matrix': distance_matrix,
                'similarity_matrix': similarity_matrix
            }
            
            # MDS for 2D visualization
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            mds_result = mds.fit_transform(distance_matrix)
            mds_embeddings[model_name] = {
                'embeddings': mds_result,
                'labels': labels
            }
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if not clustering_results:
        print(f"No valid results for construct {construct_name}")
        return
    
    # Create comprehensive visualization
    n_models = len(clustering_results)
    fig = plt.figure(figsize=(20, 8 * n_models))
    
    for i, model_name in enumerate(clustering_results.keys()):
        # Get clustering data
        result = clustering_results[model_name]
        linkage = result['linkage']
        labels = result['labels']
        distance_matrix = result['distance_matrix']
        
        # Plot 1: Dendrogram
        plt.subplot(n_models, 3, i*3 + 1)
        dendrogram = hierarchy.dendrogram(
            linkage,
            labels=labels,
            orientation='right',
            leaf_font_size=8,
            leaf_rotation=0,
        )
        plt.title(f'Hierarchical Clustering - {model_name}\n{construct_name}', fontsize=12)
        
        # Plot 2: MDS 2D embedding
        plt.subplot(n_models, 3, i*3 + 2)
        mds_data = mds_embeddings[model_name]
        points = mds_data['embeddings']
        point_labels = mds_data['labels']
        
        # Get optimal number of clusters for coloring
        optimal_n = optimal_clusters_results[model_name]['silhouette']
        clusters = hierarchy.fcluster(linkage, optimal_n, criterion='maxclust')
        
        # Create color map
        colors = sns.color_palette("husl", optimal_n)
        cluster_colors = [colors[cluster-1] for cluster in clusters]
        
        # Plot MDS points colored by cluster
        scatter = plt.scatter(points[:, 0], points[:, 1], c=cluster_colors, s=50, alpha=0.7)
        
        for j, question_id in enumerate(point_labels):
            plt.annotate(question_id, (points[j, 0], points[j, 1]), 
                        fontsize=8, alpha=0.8)
        
        plt.title(f'MDS Visualization - {model_name}\n{optimal_n} clusters (Silhouette)', fontsize=12)
        
        # Plot 3: Distance matrix heatmap
        plt.subplot(n_models, 3, i*3 + 3)
        mask = np.triu(np.ones_like(distance_matrix, dtype=bool), k=1)
        sns.heatmap(distance_matrix, 
                   mask=mask,
                   xticklabels=labels, 
                   yticklabels=labels,
                   cmap='viridis', 
                   square=True,
                   linewidths=0.1,
                   cbar_kws={'shrink': 0.8})
        plt.title(f'Distance Matrix - {model_name}\n{construct_name}', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(construct_output_dir / f'{construct_name}_unsupervised_clustering_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual metric plots (NEW FUNCTIONALITY)
    create_individual_metric_plots(optimal_clusters_results, construct_name, construct_output_dir, global_model_colors)
    
    # Create summary table of optimal clusters
    summary_data = []
    for model_name, optimal_result in optimal_clusters_results.items():
        summary_data.append({
            'Model': model_name,
            'Silhouette Optimal': optimal_result['silhouette'],
            'Calinski-Harabasz Optimal': optimal_result['calinski_harabasz'],
            'Davies-Bouldin Optimal': optimal_result['davies_bouldin'],
            'Elbow Optimal': optimal_result['elbow']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(construct_output_dir / f'{construct_name}_optimal_clusters_summary.csv', index=False)
    
    print(f"Analysis complete for {construct_name}!")
    print("Optimal cluster numbers:")
    print(summary_df.to_string(index=False))


def create_global_model_colors(constructs, data_directory):
    """
    Create a consistent color mapping for all models across all constructs.
    
    Parameters:
    constructs: List of construct names
    data_directory: Path to data directory
    
    Returns:
    dict: Dictionary mapping model names to colors
    """
    all_models = set()
    
    # Collect all unique model names across all constructs
    for construct_name in constructs:
        model_files = find_model_files(data_directory, construct_name)
        all_models.update(model_files.keys())
    
    # Create consistent color palette
    all_models = sorted(list(all_models))  # Sort for consistency
    colors = sns.color_palette("husl", len(all_models))
    
    model_colors = {model: colors[i] for i, model in enumerate(all_models)}
    
    print(f"Created consistent color mapping for {len(all_models)} models: {list(all_models)}")
    return model_colors


def main(data_directory, output_directory="output_unsupervised_clustering", constructs=None, 
         linkage_method='average', question_text_csv=None):
    """
    Main function to run the unsupervised clustering analysis.
    
    Parameters:
    data_directory (str): Path to directory containing construct subdirectories with similarity files
    output_directory (str): Directory to save all output files
    constructs (list): List of construct names to process. If None, will find all available constructs
    linkage_method (str): Linkage method for hierarchical clustering ('average', 'complete', 'ward', etc.)
    question_text_csv (str): Path to CSV file containing question text mapping
    """
    print("Starting unsupervised clustering analysis...")
    
    # Load question text mapping if provided
    question_text_mapping = {}
    if question_text_csv:
        question_text_mapping = load_question_text_mapping(question_text_csv)
    
    # Create main output directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    # Find all available constructs if not specified
    if constructs is None:
        data_path = Path(data_directory)
        constructs = []
        for subdir in data_path.iterdir():
            if subdir.is_dir():
                # Check if this directory has similarity files
                similarity_files = list(subdir.glob("*similarity*.csv"))
                if similarity_files:
                    constructs.append(subdir.name)
        
        print(f"Auto-detected constructs: {constructs}")
    
    # Create global color mapping for all models across constructs
    global_model_colors = create_global_model_colors(constructs, data_directory)
    
    # Process each construct
    for construct_name in constructs:
        try:
            process_construct(construct_name, data_directory, output_directory, 
                            question_text_mapping, linkage_method, global_model_colors)
        except Exception as e:
            print(f"Error processing construct {construct_name}: {e}")
    
    print(f"\n{'='*60}")
    print("All constructs processed!")
    print(f"Results saved in: {output_directory}")
    print(f"Color consistency maintained across all constructs")
    print(f"{'='*60}")


# Example usage:
if __name__ == "__main__":
    # Update these paths for your setup
    DATA_DIRECTORY = "output_similarity-analysis"  # Directory containing model subdirectories
    OUTPUT_DIR = "../clustering/output_unsupervised-clustering"  # Output directory
    QUESTION_TEXT_CSV = "questionnaire_model-evaluation-input_aug25.csv"  # CSV with question text
    
    # Optionally specify which constructs to process
    # CONSTRUCTS = ["BDI", "GDS", "PHQ9"]  # Or None to auto-detect all
    CONSTRUCTS = None
    
    # Choose linkage method: 'average', 'complete', 'single', 'ward'
    LINKAGE_METHOD = 'ward'
    
    # Run the analysis
    main(DATA_DIRECTORY, OUTPUT_DIR, CONSTRUCTS, LINKAGE_METHOD, QUESTION_TEXT_CSV)