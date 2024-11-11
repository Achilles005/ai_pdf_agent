from transformer_embedding_handler import TransformerEmbeddingHandler
from openai import OpenAI
from logging_config import LoggingConfig

logger = LoggingConfig().get_logger()

class LLMHandler:
    """
    Handles interactions with the LLM, including embedding generation and answering questions.
    """
    def __init__(self,api_key, model_name="bert-base-uncased",model="gpt-4o-mini"):
        """
        Initialize the LLM handler with a transformer-based embedding handler.

        Args:
            model_name (str): Hugging Face model for embedding generation.
        """
        self.model=model
        self.client = OpenAI(api_key=api_key)
        self.embedding_handler = TransformerEmbeddingHandler(model_name)

    def generate_embedding(self, text: str):
        """
        Generate an embedding for the given text.

        Args:
            text (str): Input text.

        Returns:
            numpy.ndarray: Generated embedding.
        """
        try:
            logger.info("Generating embedding for text.")
            embedding = self.embedding_handler.generate_embedding(text)
            logger.info("Embedding generation successful.")
            return embedding
        except Exception as e:
            logger.error(f"{str(e)}", exc_info=True)
            logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
            raise

    def generate_answer(self, question: str, context_chunks: list) -> dict:
        """
        Generate an answer to the user's question using OpenAI's GPT model.

        Args:
            question (str): The user's question.
            context_chunks (list): Relevant context chunks.

        Returns:
            dict: Contains the answer.
        """
        try:
            logger.info("Generating answer with OpenAI.")
            context = "\n".join(context_chunks) if context_chunks else "No context available."
            prompt = (
                'You are an intelligent assistant answering questions based on the provided context.Please provide concise and accurate answers based on the context only.\n'
                f"Context:\n{context}\n\n"
                f"Question: {question}\n"
                f"Answer:"
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            answer = response.choices[0].message.content.strip()
            # Confidence estimation (can be adjusted with a better heuristic)
            confidence = 1.0 if question.strip().lower() in context.strip().lower() else 0.8

            logger.info("Answer generation successful.")
            return {"answer": answer, "confidence": confidence}
        except Exception as e:
            logger.error(f"{str(e)}", exc_info=True)
            logger.error(f"Error generating answer with OpenAI: {str(e)}", exc_info=True)
            raise
