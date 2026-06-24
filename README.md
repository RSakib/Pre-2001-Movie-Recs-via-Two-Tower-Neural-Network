# Pre-2001 Movie Recommendation System using a Two-Tower Neural Network

![Application Frontend](front-end.jpg)

Built with a Two-Tower neural network architecture to recommend movies from the MovieLens 1M dataset, which are all before the year 2001. My neural network system maps users (who are defined by their gender, age, and their 3 last liked movies) and movies into a shared vector space. Retrieval is accelerated using an FAISS index

A live demo of this application is available on my Hugging Face at: [[link](https://huggingface.co/spaces/RSakib/Two_Towers_MovieLens_Recommendation)]

## System Architecture

* dataset.py: Restores raw database transactions, constructs textual profile strings for both user and movie towers, maps chronological watch histories, and performs an 80/20 chronological split for validation.

* engine.py: Instantiates the SentenceTransformer, fine-tunes the embeddings with in-batch contrastive loss, and exports the pre-computed catalog FAISS index.

* evaluation.py: Computes quantitative validation metrics (NDCG@10) on the test partition and produces a 2D t-SNE scatter plot of the catalog semantic space.

* app.py: Hosts the web application backend and provides a user-friendly front-end.

## Setup and Installation

Follow these steps to configure your local environment, train the neural representations, and launch the user interface.

### 1. Initialize a Virtual Environment

Set up an isolated python environment to ensure dependency isolation:

```
python -m venv venv
source venv/bin/activate
```

*(On Windows systems, use `venv\Scripts\activate` instead)*

### 2. Install Project Dependencies

Install all modules, frameworks, and system-level acceleration packages:

```
pip install -r requirements.txt
```

### 3. Run the Training Engine

Execute the training script to process the local datasets, fine-tune the Two-Tower weights, and compile the FAISS database index:

```
python engine.py
```

### 4. Run Evaluation (Optional)

To verify retrieval performance on the split test set and render the vector visualization scatter plot (saved as `movie_embeddings_tsne.png`), execute the evaluation module:

```
python evaluation.py
```

### 5. Launch the Web Application

Launch the local web server to host the custom dark-themed application interface:

```
python app.py
```

Open the local network URL displayed in your console to query movie recommendations.

## Credits and Citations

The metadata and image posters used in this project are derived from the MovieLens 1M Dataset with Posters on Kaggle:
https://www.kaggle.com/datasets/mohamedelmallah1/movielens-1m-with-posters-and-metadata/data

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Intelligent Systems and Technology (TIST) 5, 4, Article 19 (December 2015).
