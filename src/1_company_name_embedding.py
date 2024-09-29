import pandas as pd
import openai
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from tqdm import tqdm
import numpy as np
import os
import re

# Set your OpenAI API key

OPEN_AI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
openai.api_key = OPEN_AI_API_KEY

# Load datasets
mintel_df = pd.read_csv("C:/Users/rodri/Offline Folder/mintel_companies.csv", encoding='ISO-8859-1')  # Replace with your actual file path
tvl_df = pd.read_csv("C:/Users/rodri/Offline Folder/tvl_companies.csv", encoding='ISO-8859-1')        # Replace with your actual file path

# Inspect the first few rows
print("MINTEL Data:")
print(mintel_df.head())

print("/nTVL Data:")
print(tvl_df.head())

import re

def clean_company_name(name):
    # Convert to lowercase
    name = name.lower()
    
    # Remove punctuation and special characters
    name = re.sub(r'[^\w\s]', '', name)
    
    # Remove common corporate suffixes
    suffixes = ['inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'co', 'company', 'sa', 'ltda']
    pattern = r'\b(?:' + '|'.join(suffixes) + r')\b'
    name = re.sub(pattern, '', name)
    
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

# Apply cleaning function
mintel_df['Clean_Name'] = mintel_df['COMPANY'].apply(clean_company_name)
tvl_df['Clean_Name'] = tvl_df['COMPANY'].apply(clean_company_name)

mintel_df.dropna(subset=['Clean_Name'], inplace=True)
tvl_df.dropna(subset=['Clean_Name'], inplace=True)

# Inspect cleaned names
print(mintel_df[['COMPANY', 'Clean_Name']].head())
print(tvl_df[['COMPANY', 'Clean_Name']].head())

def get_embeddings_batch(texts, model="text-embedding-3-large", batch_size=200):
    """
    Generate embeddings for a list of texts using OpenAI's Embedding API with batching.
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = openai.embeddings.create(input=batch, model=model)
        embeddings.extend([data.embedding for data in response.data])
    return embeddings

def process_dataframe(df, text_column, embedding_column, model="text-embedding-3-large", batch_size=200):
    """
    Process a dataframe to add embeddings for a specified text column.
    """
    texts = df[text_column].tolist()
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc=f"Generating Embeddings for {text_column}"):
        batch = texts[i:i + batch_size]
        batch_embeddings = get_embeddings_batch(batch, model=model, batch_size=batch_size)
        embeddings.extend(batch_embeddings)
    
    df[embedding_column] = embeddings
    return df

# Process dataframe
tvl_df = process_dataframe(tvl_df, 'EMBEDDING_DATA', 'ada_embedding', model='text-embedding-3-large')
tvl_df.to_parquet('C:/Users/rodri/Offline Folder/tvl_with_embeddings.parquet')

#mintel_df = process_dataframe(mintel_df, 'EMBEDDING_DATA', 'ada_embedding', model='text-embedding-3-large')
#mintel_df.to_parquet('C:/Users/rodri/Offline Folder/mintel_with_embeddings.parquet')