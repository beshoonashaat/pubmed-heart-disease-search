# pubmed-heart-disease-search
# PubMed Heart Disease Search Engine

An end-to-end Information Retrieval (IR) system that scrapes medical research from PubMed, processes the text, and provides search functionality using Vector Space Modeling.

## 🚀 Features
* **Automated Scraping:** Fetches titles and abstracts via NCBI Entrez API.
* **Text Normalization:** Custom NLP pipeline (Tokenization, Stopword removal, Porter Stemming).
* **Search Models:** Implements both Inverted Indexing and TF-IDF Cosine Similarity.
* **Performance Metrics:** Evaluates accuracy using Precision, Recall, F1-Score, and MAP (Mean Average Precision).
* **Interactive UI:** Search results rendered directly in Jupyter via ipywidgets.
