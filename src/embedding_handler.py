import faiss
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from logging_config import LoggingConfig

logger = LoggingConfig().get_logger()

class EmbeddingHandler:
    """
    Handles FAISS-based embedding storage and retrieval.
    """
    def __init__(self, embedding_dim=768, model_name="bert-base-uncased"):
        """
        Initialize the FAISS index with the specified embedding dimension.

        Args:
            embedding_dim (int): Dimensionality of embeddings.
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(self.embedding_dim)  # Initialize FAISS with correct dimension
        self.metadata = []  # Metadata to track chunks associated with embeddings
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def add_embeddings(self, embeddings, metadata):
        """
        Add embeddings and associated metadata to the FAISS index.

        Args:
            embeddings (list or np.ndarray): List of embeddings to add.
            metadata (list): Metadata corresponding to each embedding.
        """
        try:
            # Convert embeddings to float32 numpy array
            
            embeddings_array = np.array(embeddings).astype('float32')

            # Validate embedding dimensions
            if embeddings_array.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {embeddings_array.shape[1]}"
                )

            # Add embeddings to FAISS index
            self.index.add(embeddings_array)

            # Extend metadata list
            self.metadata.extend(metadata)

            # Log FAISS index state
            logger.info(f"FAISS index updated. Total entries: {self.index.ntotal}")
            if len(self.metadata) != self.index.ntotal:
                raise ValueError(
                    f"Metadata count ({len(self.metadata)}) does not match FAISS index count ({self.index.ntotal})."
                )
        except Exception as e:
            logger.error(f"{str(e)}", exc_info=True)
            logger.error(f"Error adding embeddings to FAISS index: {str(e)}", exc_info=True)
            raise

    def search(self, query_embedding, top_k=10):
        """
        Search the FAISS index for the most similar embeddings.

        Args:
            query_embedding (np.ndarray): Query embedding for similarity search.
            top_k (int): Number of top results to retrieve.

        Returns:
            list: List of results with distance and associated metadata.
        """
        try:
            # Check if FAISS index is populated
            if self.index.ntotal == 0:
                raise ValueError("FAISS index is empty. Add embeddings before searching.")

            # Ensure query_embedding is properly formatted
            if not isinstance(query_embedding, (np.ndarray, list)):
                raise ValueError("Query embedding must be a numerical vector, not a string.")

            # Convert query embedding to a numpy array
            query_array = np.array([query_embedding]).astype('float32')

            # Validate query embedding dimensions
            if query_array.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"Query embedding dimension mismatch. Expected {self.embedding_dim}, got {query_array.shape[1]}"
                )

            # Perform FAISS search
            logger.info("Performing FAISS search...")
            distances, indices = self.index.search(query_array, top_k)

            # Compile results with metadata
            results = [
                {"distance": distances[0][i], "metadata": self.metadata[idx]}
                for i, idx in enumerate(indices[0]) if idx < len(self.metadata)
            ]

            # Log results
            logger.info(f"Search returned {len(results)} results.")
            return results
        except ValueError as ve:
            # Handle specific validation errors
            logger.error(f"Validation error during FAISS search: {str(ve)}", exc_info=True)
            raise ve
        except Exception as e:
            logger.error(f"{str(e)}", exc_info=True)
            logger.error(f"Error during FAISS search: {str(e)}", exc_info=True)
            raise

    
    def generate_embedding(self, text: str):
        """
        Generate embeddings for the given text using a Hugging Face BERT model.

        Args:
            text (str): Input text.

        Returns:
            np.ndarray: Generated embedding.
        """
        try:
            # Tokenize the text
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

            # Get the outputs from the BERT model
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use the last hidden state to create a mean pooled embedding
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()
            
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
            raise