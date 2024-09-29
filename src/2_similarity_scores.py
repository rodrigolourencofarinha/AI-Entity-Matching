import pandas as pd
import openai
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from tqdm import tqdm
import numpy as np
import os
import re

# Upload data
tvl_df = pd.read_parquet(
    'C:/Users/rodri/Offline Folder/tvl_with_embeddings.parquet',
    columns=['EMBEDDING_DATA', 'ada_embedding']
)
mintel_df = pd.read_parquet(
    'C:/Users/rodri/Offline Folder/mintel_with_embeddings.parquet',
    columns=['EMBEDDING_DATA', 'ada_embedding']
)

# Convert embeddings to numpy arrays
tvl_embeddings_np = np.array(tvl_df['ada_embedding'].tolist())
mintel_embeddings_np = np.array(mintel_df['ada_embedding'].tolist())

# Compute cosine similarity matrix
# Shape: (len(tvl_df), len(mintel_df))
# Each element [i][j] represents the similarity between TVL[i] and Mintel[j]
similarity_matrix = cosine_similarity(tvl_embeddings_np, mintel_embeddings_np)

print("Cosine Similarity Matrix Shape:", similarity_matrix.shape)

# Define a function to compute fuzzy score
def compute_fuzzy_score(name1, name2):
    return fuzz.token_sort_ratio(name1, name2) / 100  # Normalize to [0,1]

# Define weights
EMBEDDING_WEIGHT = 1.0
FUZZY_WEIGHT = 0.0  # Set to >0 if you want to include fuzzy matching

# Function to find the best match for each TVL company in Mintel
def match_companies(
    base_df, 
    target_df, 
    similarity_matrix, 
    top_n=5, 
    embedding_weight=1.0, 
    fuzzy_weight=0.0, 
    threshold=0.7
):
    """
    For each company in base_df (TVL), find the best matching company in target_df (Mintel).
    """
    matches = []
    for i, base_row in tqdm(
        base_df.iterrows(), 
        total=base_df.shape[0], 
        desc="Matching Companies"
    ):
        # Get similarity scores for the current TVL company against all Mintel companies
        sim_scores = similarity_matrix[i]
        
        # Get top_n indices with highest similarity
        top_indices = sim_scores.argsort()[-top_n:][::-1]
        
        best_match = None
        best_score = 0
        
        for idx in top_indices:
            target_row = target_df.iloc[idx]
            embedding_sim = sim_scores[idx]
            fuzzy_sim = compute_fuzzy_score(base_row['EMBEDDING_DATA'], target_row['EMBEDDING_DATA'])
            
            # Combine scores
            combined_score = (embedding_weight * embedding_sim) + (fuzzy_weight * fuzzy_sim)
            
            if combined_score > best_score:
                best_score = combined_score
                best_match = {
                    'TVL_Company': base_row['EMBEDDING_DATA'],
                    'Mintel_Company': target_row['EMBEDDING_DATA'],
                    'Embedding_Similarity': embedding_sim,
                    'Fuzzy_Similarity': fuzzy_sim,
                    'Combined_Score': combined_score
                }
        
        # Apply threshold
        if best_match and best_score >= threshold:
            matches.append(best_match)
        else:
            matches.append({
                'TVL_Company': base_row['EMBEDDING_DATA'],
                'Mintel_Company': None,
                'Embedding_Similarity': None,
                'Fuzzy_Similarity': None,
                'Combined_Score': None
            })
    
    return pd.DataFrame(matches)

# Perform matching
matched_df = match_companies(
    base_df=tvl_df,
    target_df=mintel_df,
    similarity_matrix=similarity_matrix,
    top_n=5,
    embedding_weight=EMBEDDING_WEIGHT,
    fuzzy_weight=FUZZY_WEIGHT,
    threshold=0.1  # Adjust based on your validation
)

# Display some matched results
print(matched_df.head())

# Save matched results to a CSV for manual review
matched_df.to_csv('C:/Users/rodri/Offline Folder/matched_companies_tvl_based_2.csv', index=False)
