import pickle
import hashlib
import os

class EmbeddingCache:
    """
    Caches embeddings in memory and provides persistence to disk.
    """
    def __init__(self, cache_dir="cache"):
        """
        Initializes the cache with a directory for saving embeddings.
        
        Args:
            cache_dir (str): Directory to store cached embeddings.
        """
        self.cache = {}
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_key(self, file_path):
        """
        Generate a unique cache key based on the file content.
        
        Args:
            file_path (str): Path to the PDF file.
        
        Returns:
            str: MD5 hash of the file content.
        """
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash

    def get_embeddings(self, file_key):
        """
        Retrieve embeddings from the cache or disk.
        
        Args:
            file_key (str): The cache key for the file.
        
        Returns:
            dict or None: Cached embeddings and metadata if available.
        """
        # Check in-memory cache
        if file_key in self.cache:
            return self.cache[file_key]
        
        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{file_key}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.cache[file_key] = pickle.load(f)
            return self.cache[file_key]
        
        return None

    def save_embeddings(self, file_key, embeddings, metadata):
        """
        Save embeddings and metadata to memory and disk.
        
        Args:
            file_key (str): The cache key for the file.
            embeddings (list): Embedding vectors.
            metadata (list): Metadata for the embeddings.
        """
        # Save to memory
        self.cache[file_key] = {"embeddings": embeddings, "metadata": metadata}

        # Save to disk
        cache_file = os.path.join(self.cache_dir, f"{file_key}.pkl")
        with open(cache_file, "wb") as f:
            pickle.dump(self.cache[file_key], f)
