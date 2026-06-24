"""
evaluation.py - Evaluation Pipeline (Baseline Configuration)
Restores strict string-context checking and un-labeled minimalist t-SNE settings.
"""

import os
import sys

def evaluate_ndcg(model, index, test_df, movies_df, sample_size=1000):
    """Computes exact-string matching context metrics on the chronological evaluation partition."""
    import math
    import numpy as np
    from tqdm import tqdm  
    
    print("\n" + "="*60)
    print(f" STEP 1: CALCULATING NDCG@10 METRICS (Sample Size: {sample_size})")
    print("="*60)
    
    sampled_test = test_df.sample(min(sample_size, len(test_df)), random_state=42)
    user_contexts = sampled_test['user_context'].tolist()
    
    # Reverted to exact context-wrapper target evaluation strings
    target_contexts = sampled_test['movie_context'].tolist()
    catalog_contexts = movies_df['movie_context'].tolist()
    
    print("-> Encoding user profile vectors...")
    user_embeddings = model.encode(user_contexts, show_progress_bar=True, convert_to_numpy=True)
    
    import faiss  
    print("-> Indexing vector matrix norms...")
    faiss.normalize_L2(user_embeddings)
    
    print("-> Executing K-NN index query against the movie catalog...")
    distances, indices = index.search(user_embeddings, 10)
    ndcg_scores = []
    
    print("-> Checking string context equality matches across retrieved indices...")
    for idx in tqdm(range(len(sampled_test)), desc="Calculating NDCG"):
        target_movie = target_contexts[idx].strip()
        retrieved_indices = indices[idx]
        
        rank = -1
        for r, catalog_idx in enumerate(retrieved_indices):
            # Strict original context verification check
            if catalog_contexts[catalog_idx] == target_movie:
                rank = r
                break
                
        if rank != -1:
            ndcg = 1.0 / math.log2(rank + 2)
        else:
            ndcg = 0.0
            
        ndcg_scores.append(ndcg)
        
    mean_ndcg = np.mean(ndcg_scores)
    print("\n" + "-"*40)
    print(f" FINAL RELEVANCE SCORE: NDCG@10 = {mean_ndcg:.4f}")
    print("-"*40 + "\n")
    return mean_ndcg

def visualize_tsne(model, movies_df):
    """Generates a clean 2D t-SNE scatter representation of movie context vectors."""
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    
    print("="*60)
    print(" STEP 2: GENERATING HIGH-DIMENSIONAL t-SNE VISUALIZATION")
    print("="*60)
    
    movie_contexts = movies_df['movie_context'].tolist()
    print("-> Encoding movie catalog to latent embeddings...")
    embeddings = model.encode(movie_contexts, show_progress_bar=True, convert_to_numpy=True)
    
    print("-> Computing 2D manifold embeddings layout...")
    # Kept max_iter here to maintain scikit-learn version compliance
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, init='pca', verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    primary_genres = movies_df['movie_genres'].apply(lambda x: str(x).split(",")[0].strip())
    unique_genres = primary_genres.unique()
    
    print("\n-> Rendering visual layer components...")
    plt.figure(figsize=(12, 8), facecolor='#111111')
    ax = plt.axes()
    ax.set_facecolor('#111111')
    
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, len(unique_genres))]
    
    for genre, color in tqdm(zip(unique_genres, colors), total=len(unique_genres), desc="Plotting Genres"):
        mask = primary_genres == genre
        if mask.sum() > 5: 
            plt.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1], 
                label=genre, 
                alpha=0.7, 
                edgecolors='none', 
                s=15,
                color=color
            )
            
    plt.title("Two-Tower Movie Catalog Latent Vector Space (t-SNE)", color='white', fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, color='#222222', linestyle='--', alpha=0.5)
    
    plt.xlabel("")
    plt.ylabel("")
    
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), facecolor='#111111', edgecolor='#333333', title="Primary Genre")
    plt.setp(legend.get_title(), color='white', fontsize=10, fontweight='bold')
    plt.setp(legend.get_texts(), color='#CCCCCC', fontsize=8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.tick_params(colors='#888888', labelsize=8)
    
    output_plot_path = "movie_embeddings_tsne.png"
    plt.savefig(output_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print("\n" + "-"*40)
    print(f" SUCCESS: Minimalist t-SNE plot exported to '{output_plot_path}'")
    print("-"*40 + "\n")

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" Starting evaluation pipeline...")
    print("="*60)
    
    model_path = os.path.join("data", "two_tower_model")
    index_path = os.path.join("data", "movies.index")
    
    if not os.path.exists(model_path) or not os.path.exists(index_path):
        print("\n[ERROR]: Pipeline error: Missing required artifacts in the 'data' directory.")
        print("Please run 'python engine.py' first to save model weights and index artifacts.")
        sys.exit(1)
        
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    from dataset import prepare_and_split_data
    
    nn_model = SentenceTransformer(model_path)
    faiss_index = faiss.read_index(index_path)
    
    _, test_dataset, catalog_df = prepare_and_split_data()
    
    evaluate_ndcg(nn_model, faiss_index, test_dataset, catalog_df, sample_size=1000)
    visualize_tsne(nn_model, catalog_df)