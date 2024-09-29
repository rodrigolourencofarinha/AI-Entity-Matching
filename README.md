# Entity Matching Methodology using AI

This repository contains a Python-based approach for matching companies across two datasets using a combination of OpenAI embeddings (or any other API) and fuzzy string matching techniques. The goal is to accurately match companies based on their names and additional embedding data, leveraging cosine similarity and optional fuzzy matching.

# How I Started

I developed this methodology as part of a research project where I needed to match two datasets, each containing over 20,000 companies. One dataset included the CUSIP and ISIN codes for companies, while the other did not. To overcome the lack of standard identifiers in one of the datasets, I created this matching process based on multiple fields such as company name, address, country, and ticker symbols.

This script utilizes a large language model (LLM) to generate embeddings, which are then used to compute cosine similarity for matching. This is the first version of the script, and I plan to refine and improve it over time.

## Key Components

### 1. **Data Loading and Preprocessing**
   - **Data Sources**: Two datasets are loaded:
   - **Cleaning Function**: Company information are cleaned using the `clean_company_name` function to standardize them by:
     - Lowercasing.
     - Removing punctuation and special characters.
     - Stripping common corporate suffixes (e.g., "Inc", "LLC", etc.).
   - The cleaned names are stored in new columns for further processing.

### 2. **Embedding Generation**
   - The OpenAI API is used to generate text embeddings for company names, which allows for more robust similarity comparisons.
   - A custom function `get_embeddings_batch` is provided for generating embeddings in batches to efficiently process large datasets.
   - Embeddings are saved as `.parquet` files for later use.

### 3. **Cosine Similarity Calculation**
   - After generating embeddings for both datasets, a cosine similarity matrix is computed. This matrix provides a pairwise similarity score between all TVL and Mintel companies based on their embeddings.

### 4. **Fuzzy Matching**
   - Optionally, fuzzy string matching (using RapidFuzz) can be incorporated to improve the matching by accounting for variations in spelling and formatting.
   - The `compute_fuzzy_score` function computes a normalized fuzzy match score between company names.

### 5. **Company Matching**
   - The `match_companies` function combines embedding-based cosine similarity and fuzzy matching scores (if used) to find the best matching companies.
   - The top N matches are considered for each company, and a threshold is applied to filter out weak matches.
   - Matches are saved for further review.

## Installation

To use this code, you'll need to install the following dependencies:

```bash
pip install pandas numpy scikit-learn openai rapidfuzz tqdm
```

You also need an OpenAI API key. Ensure it is set up in your environment variables:
```bash
OPENAI_API_KEY='your-api-key'
```
## Other Resources

Check out other amazing resources:
- [Dedupe](https://github.com/dedupeio/dedupe)
- [RecordLinkage](https://recordlinkage.readthedocs.io/en/latest/)
