import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform, pdist
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
import os
import glob
from pathlib import Path


def load_thematic_categories(csv_path):
    """
    Load thematic categories from CSV file.
    
    Parameters:
    csv_path (str): Path to CSV file with columns: construct, questionnaire_name, 
                   question_number, question_text, answer_options, scoring, construct_category
                   Note: question_number should contain full question IDs like "BDI_01", "GDS_02"
    
    Returns:
    dict: Dictionary mapping construct to question-category mappings
    """
    df = pd.read_csv(csv_path)
    
    # Group by construct
    constructs_data = {}
    
    for construct in df['construct'].unique():
        construct_df = df[df['construct'] == construct]
        
        # Create question identifier to category mapping for this construct
        question_id_to_category = {}
        thematic_categories = {}
        
        for _, row in construct_df.iterrows():
            # Use question_number directly as it already contains the full question ID
            question_id = row['question_number']
            
            category = row['construct_category']
            question_text = row['question_text']
            
            # Add to question ID-to-category mapping
            question_id_to_category[question_id] = category
            
            # Add to thematic categories grouping (using question text for display)
            if category not in thematic_categories:
                thematic_categories[category] = []
            thematic_categories[category].append(question_text)
        
        constructs_data[construct] = {
            'thematic_categories': thematic_categories,
            'question_id_to_category': question_id_to_category
        }
    
    return constructs_data


def find_model_files(data_directory, construct_name, file_type='similarity'):
    """
    Find similarity or embedding CSV files for a specific construct in the data directory.
    
    Parameters:
    data_directory (str): Path to directory containing construct subdirectories
    construct_name (str): Name of the construct to find files for
    file_type (str): Type of file to find - 'similarity' or 'embeddings'
    
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
    
    # Look for files in the construct subdirectory
    if file_type == 'similarity':
        files = list(construct_subdir.glob("*similarity*.csv"))
    elif file_type == 'embeddings':
        files = list(construct_subdir.glob("*embeddings.csv"))
    else:
        print(f"Unknown file_type: {file_type}")
        return model_files
    
    for file_path in files:
        # Extract model name from filename
        # Expected format: ModelName_construct_similarity-results.csv or ModelName_construct_embeddings.csv
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


def evaluate_thematic_clustering(linkage_matrix, labels, question_id_to_category):
    """Function to evaluate cluster quality based on thematic categories"""
    # Extract flat clusters at different distance thresholds
    # Use a range that can accommodate the number of thematic categories
    max_categories = len(set(question_id_to_category.values()))
    max_clusters = min(max_categories + 2, len(labels))  # Allow a bit more than categories
    num_clusters_to_try = range(2, max_clusters + 1)
    results = []
    
    for n_clusters in num_clusters_to_try:
        # Get cluster assignments
        clusters = hierarchy.fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Calculate how well clusters align with thematic categories
        # Initialize counters for each category
        category_cluster_counts = {}
        for category in set(question_id_to_category.values()):
            category_cluster_counts[category] = {}
        
        # Count which clusters each category's questions fall into
        for i, label in enumerate(labels):
            if label in question_id_to_category:
                category = question_id_to_category[label]
                cluster = clusters[i]
                if cluster not in category_cluster_counts[category]:
                    category_cluster_counts[category][cluster] = 0
                category_cluster_counts[category][cluster] += 1
        
        # Calculate cohesion score for each category
        # (% of category members in the most common cluster for that category)
        cohesion_scores = {}
        for category, cluster_counts in category_cluster_counts.items():
            if cluster_counts:  # Only calculate if we have questions from this category
                max_count = max(cluster_counts.values()) if cluster_counts else 0
                total_questions = sum(cluster_counts.values())
                cohesion_scores[category] = max_count / total_questions if total_questions > 0 else 0
            else:
                cohesion_scores[category] = np.nan
        
        # Calculate overall cohesion (average of category cohesions)
        valid_scores = [score for score in cohesion_scores.values() if not np.isnan(score)]
        overall_cohesion = np.mean(valid_scores) if valid_scores else 0
        
        results.append({
            'n_clusters': n_clusters,
            'overall_cohesion': overall_cohesion,
            'category_cohesion': cohesion_scores
        })
    
    return results


def process_similarity_matrix(model_name, file_path, question_id_to_category):
    """Process clustering from similarity matrix"""
    print(f"\n  [Similarity Matrix] Processing {model_name}...")
    
    # Read CSV - the first column will be the index
    df = pd.read_csv(file_path, index_col=0)
    
    # Filter questions relevant to this construct
    relevant_questions = []
    relevant_indices = []
    
    for idx, question_id in enumerate(df.index):
        if question_id in question_id_to_category:
            relevant_questions.append(question_id)
            relevant_indices.append(idx)
    
    print(f"    Found {len(relevant_questions)} relevant questions")
    
    if not relevant_questions:
        return None
    
    # Filter the dataframe to only include relevant questions
    df_filtered = df.iloc[relevant_indices, relevant_indices]
    
    # Process similarity matrix
    similarity_matrix = df_filtered.values
    distance_matrix = similarity_to_distance(similarity_matrix)
    
    # Ensure distance matrix is symmetric
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    labels = df_filtered.index.tolist()
    
    # Compute linkage for hierarchical clustering
    condensed_distance = squareform(distance_matrix)
    linkage_matrix = hierarchy.linkage(condensed_distance, method='average')
    
    # MDS for 2D visualization
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    mds_result = mds.fit_transform(distance_matrix)
    
    return {
        'linkage': linkage_matrix,
        'labels': labels,
        'df': df_filtered,
        'distance_matrix': distance_matrix,
        'mds_embeddings': mds_result,
        'method': 'similarity_matrix'
    }


def process_embeddings(model_name, file_path, question_id_to_category):
    """Process clustering from embeddings directly"""
    print(f"\n  [Embeddings] Processing {model_name}...")
    
    # Read embeddings CSV
    df = pd.read_csv(file_path, index_col=0)
    
    # Filter questions relevant to this construct
    relevant_questions = []
    relevant_indices = []
    
    for idx, question_id in enumerate(df.index):
        if question_id in question_id_to_category:
            relevant_questions.append(question_id)
            relevant_indices.append(idx)
    
    print(f"    Found {len(relevant_questions)} relevant questions")
    
    if not relevant_questions:
        return None
    
    # Filter embeddings
    embeddings = df.iloc[relevant_indices].values
    labels = [df.index[i] for i in relevant_indices]
    
    # Compute pairwise distances using cosine distance
    distances = pdist(embeddings, metric='cosine')
    distance_matrix = squareform(distances)
    
    # Compute linkage for hierarchical clustering
    linkage_matrix = hierarchy.linkage(distances, method='average')
    
    # MDS for 2D visualization
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    mds_result = mds.fit_transform(distance_matrix)
    
    return {
        'linkage': linkage_matrix,
        'labels': labels,
        'embeddings': embeddings,
        'distance_matrix': distance_matrix,
        'mds_embeddings': mds_result,
        'method': 'embeddings'
    }


def process_construct(construct_name, construct_data, data_directory, output_dir, 
                     use_embeddings=True, model_order=None):
    """
    Process clustering analysis for a single construct using both methods if available.
    
    Parameters:
    construct_name (str): Name of the construct
    construct_data (dict): Dictionary containing thematic_categories and question_id_to_category
    data_directory (str): Path to directory containing construct subdirectories
    output_dir (str): Directory to save output files
    use_embeddings (bool): Whether to also process embeddings if available
    model_order (list): Optional list defining the order of models for consistent visualization
    """
    print(f"\n{'='*60}")
    print(f"Processing construct: {construct_name}")
    print(f"{'='*60}")
    
    thematic_categories = construct_data['thematic_categories']
    question_id_to_category = construct_data['question_id_to_category']
    
    # Create output directory for this construct
    construct_output_dir = Path(output_dir) / construct_name
    construct_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find both similarity and embedding files
    similarity_files = find_model_files(data_directory, construct_name, 'similarity')
    embedding_files = find_model_files(data_directory, construct_name, 'embeddings') if use_embeddings else {}
    
    print(f"Found {len(similarity_files)} similarity files")
    print(f"Found {len(embedding_files)} embedding files")
    
    # Process similarity matrices
    similarity_results = {}
    for model_name, file_path in similarity_files.items():
        try:
            result = process_similarity_matrix(model_name, file_path, question_id_to_category)
            if result:
                similarity_results[model_name] = result
        except Exception as e:
            print(f"    Error: {e}")
    
    # Process embeddings
    embedding_results = {}
    for model_name, file_path in embedding_files.items():
        try:
            result = process_embeddings(model_name, file_path, question_id_to_category)
            if result:
                embedding_results[model_name] = result
        except Exception as e:
            print(f"    Error: {e}")
    
    # Evaluate both methods
    similarity_evaluations = {}
    embedding_evaluations = {}
    
    for model_name, result in similarity_results.items():
        similarity_evaluations[model_name] = evaluate_thematic_clustering(
            result['linkage'], result['labels'], question_id_to_category
        )
    
    for model_name, result in embedding_results.items():
        embedding_evaluations[model_name] = evaluate_thematic_clustering(
            result['linkage'], result['labels'], question_id_to_category
        )
    
    # Create comparison visualization if we have both methods for the same models
    common_models = set(similarity_results.keys()) & set(embedding_results.keys())
    
    if common_models:
        print(f"\nCreating comparison for {len(common_models)} models with both methods")
        
        # Compare overall cohesion across methods
        n_categories = len(set(question_id_to_category.values()))
        
        comparison_data = []
        for model_name in common_models:
            sim_eval = similarity_evaluations[model_name]
            emb_eval = embedding_evaluations[model_name]
            
            # Find evaluation at n_categories clusters
            sim_result = next((r for r in sim_eval if r['n_clusters'] == n_categories), None)
            emb_result = next((r for r in emb_eval if r['n_clusters'] == n_categories), None)
            
            if sim_result and emb_result:
                comparison_data.append({
                    'model': model_name,
                    'similarity_method': sim_result['overall_cohesion'],
                    'embeddings_method': emb_result['overall_cohesion']
                })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            df_comparison = df_comparison.set_index('model')
            
            # Create comparison bar plot
            fig, ax = plt.subplots(figsize=(12, 6))
            df_comparison.plot(kind='bar', ax=ax, width=0.8)
            ax.set_ylabel('Overall Thematic Cohesion Score')
            ax.set_xlabel('Model')
            ax.set_title(f'Comparison: Similarity Matrix vs Embeddings Method\n{construct_name}')
            ax.legend(['Similarity Matrix Method', 'Direct Embeddings Method'])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(construct_output_dir / f'{construct_name}_method_comparison.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save comparison data
            df_comparison.to_csv(construct_output_dir / f'{construct_name}_method_comparison.csv')
    
    # Create heatmaps for both methods
    for method_name, evaluations in [('similarity', similarity_evaluations), 
                                     ('embeddings', embedding_evaluations)]:
        if not evaluations:
            continue
        
        n_categories = len(set(question_id_to_category.values()))
        cohesion_data = {}
        
        for model_name, eval_result in evaluations.items():
            target_result = next((r for r in eval_result if r['n_clusters'] == n_categories), None)
            if target_result:
                cohesion_data[model_name] = target_result['category_cohesion']
        
        if cohesion_data:
            # Convert to DataFrame for heatmap
            if model_order:
                ordered_models = [model for model in model_order if model in cohesion_data]
                cohesion_df = pd.DataFrame({model: cohesion_data[model] for model in ordered_models})
            else:
                cohesion_df = pd.DataFrame(cohesion_data)
            
            # Calculate dynamic figure size
            n_cats = len(cohesion_df.index)
            n_models = len(cohesion_df.columns)
            
            cell_width = 1.2
            cell_height = 0.8
            
            fig_width = max(6, n_models * cell_width + 2)
            fig_height = max(4, n_cats * cell_height + 2)
            
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            colorbar_shrink = max(0.6, min(0.9, fig_height / 8))
            colorbar_aspect = max(15, min(30, fig_height * 3))
            
            sns.heatmap(cohesion_df, 
                        annot=True, 
                        square=True,
                        cmap='YlGnBu', 
                        fmt='.2f', 
                        linewidths=0.5,
                        cbar_kws={
                            'shrink': colorbar_shrink,
                            'aspect': colorbar_aspect,
                            'pad': 0.08
                        },
                        ax=ax)
            
            title_fontsize = min(16, max(12, fig_width))
            label_fontsize = min(14, max(10, fig_width * 0.8))
            
            method_label = "Similarity Matrix" if method_name == 'similarity' else "Direct Embeddings"
            ax.set_title(f'Category Cohesion by Model - {construct_name}\nMethod: {method_label}', 
                        fontweight='bold', fontsize=title_fontsize, pad=20)
            ax.set_xlabel('Different Sentence Transformer Models', 
                         fontweight='bold', fontsize=label_fontsize)
            ax.set_ylabel('Construct Dimensions', 
                         fontweight='bold', fontsize=label_fontsize)
            
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            plt.savefig(construct_output_dir / f'{construct_name}_cohesion_heatmap_{method_name}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"\nAnalysis complete for {construct_name}!")


def main(data_directory, categories_csv_path, output_directory="output", use_embeddings=True):
    """
    Main function to run the analysis.
    
    Parameters:
    data_directory (str): Path to directory containing construct subdirectories
    categories_csv_path (str): Path to CSV file with construct categories
    output_directory (str): Directory to save all output files
    use_embeddings (bool): Whether to also process embeddings if available
    """
    # Load thematic categories from CSV
    print("Loading thematic categories...")
    constructs_data = load_thematic_categories(categories_csv_path)
    print(f"Loaded {len(constructs_data)} constructs: {list(constructs_data.keys())}")
    
    # Create main output directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    # Collect all unique model names
    all_model_names = set()
    for construct_name in constructs_data.keys():
        model_files = find_model_files(data_directory, construct_name, 'similarity')
        all_model_names.update(model_files.keys())
        if use_embeddings:
            embedding_files = find_model_files(data_directory, construct_name, 'embeddings')
            all_model_names.update(embedding_files.keys())
    
    model_order = sorted(list(all_model_names))
    print(f"Model order: {model_order}")
    
    # Process each construct
    for construct_name, construct_data in constructs_data.items():
        try:
            process_construct(construct_name, construct_data, data_directory, 
                            output_directory, use_embeddings, model_order)
        except Exception as e:
            print(f"Error processing construct {construct_name}: {e}")
    
    print(f"\n{'='*60}")
    print("All constructs processed!")
    print(f"Results saved in: {output_directory}")
    print(f"{'='*60}")


# Example usage:
if __name__ == "__main__":
    DATA_DIRECTORY = "output_similarity-analysis"
    CATEGORIES_CSV = "model-evaluation-input_Sep25.csv"
    OUTPUT_DIR = "../clustering/output_clustering-cohesion"
    
    # Set to True to compare both methods, False to only use similarity matrices
    USE_EMBEDDINGS = True
    
    main(DATA_DIRECTORY, CATEGORIES_CSV, OUTPUT_DIR, USE_EMBEDDINGS)