"""
engine.py - Core Training Pipeline (Baseline Configuration)
Restores MultipleNegativesRankingLoss and batch size 64 layout.
"""

import os
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from dataset import prepare_and_split_data

def train_pipeline():
    os.makedirs("data", exist_ok=True)
    
    train_df, _, movies_df = prepare_and_split_data()
    sampled_train = train_df.sample(min(75000, len(train_df)), random_state=42)
    
    print(f"Initializing Transformer Architecture with {len(sampled_train)} training records...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    if torch.cuda.is_available():
        model = model.to('cuda')
        print("-> Hardware state: Training on CUDA GPU accelerated cores.")
    else:
        print("-> Hardware state: No GPU detected. Training on native CPU.")

    # Restored to standard contrastive text matching examples
    train_examples = []
    for _, row in sampled_train.iterrows():
        train_examples.append(InputExample(
            texts=[str(row['user_context']), str(row['movie_context'])]
        ))
        
    # Reverted to peak baseline batch size configuration
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
    
    # Restored contrastive In-Batch Negatives loss setup
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    print("Beginning neural weight updates across 2 epochs...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=5,
        warmup_steps=int(len(train_dataloader) * 0.1),
        show_progress_bar=True
    )
    
    output_model_path = os.path.join("data", "two_tower_model")
    model.save(output_model_path)
    print(f"Model saved locally to '{output_model_path}'")
    
    import faiss
    print("\nBuilding global FAISS vector search index...")
    movie_contexts = movies_df['movie_context'].tolist()
    movie_embeddings = model.encode(movie_contexts, show_progress_bar=True, convert_to_numpy=True)
    
    faiss.normalize_L2(movie_embeddings)
    embedding_dim = movie_embeddings.shape[1]
    
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(movie_embeddings)
    
    output_index_path = os.path.join("data", "movies.index")
    faiss.write_index(index, output_index_path)
    print(f"FAISS index successfully saved to '{output_index_path}'")

if __name__ == "__main__":
    train_pipeline()