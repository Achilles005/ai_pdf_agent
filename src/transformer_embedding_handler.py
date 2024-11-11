import torch
from transformers import BertTokenizer, BertModel
from logging_config import LoggingConfig

logger = LoggingConfig().get_logger()
class TransformerEmbeddingHandler:
    """
    Handles embedding generation using Hugging Face Transformers.
    """
    def __init__(self, model_name="bert-base-uncased"):
        """
        Initialize the embedding handler with the chosen transformer model.

        Args:
            model_name (str): Name of the Hugging Face model.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def generate_embedding(self, text: str):
        """
        Generate embeddings for the given text using the specified model.

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

