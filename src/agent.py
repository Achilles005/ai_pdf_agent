import time
from logging_config import LoggingConfig
from query_understanding import QueryUnderstanding
from transformers import AutoTokenizer, AutoModel
import torch
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity

logger = LoggingConfig().get_logger()

class AIAgent:
    def __init__(self, pdf_processor, embedding_handler, llm_handler, embedding_cache):
        """
        Initialize the AI agent with shared components.
        """
        self.pdf_processor = pdf_processor
        self.embedding_handler = embedding_handler
        self.llm_handler = llm_handler
        self.embedding_cache = embedding_cache
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.query_understanding = QueryUnderstanding()

    def prioritize_query(self, question: str) -> dict:
        """
        Prioritize the query based on extracted entities.

        Args:
            question (str): User's question.

        Returns:
            dict: Prioritized query information including entities and types.
        """
        entities = self.query_understanding.extract_entities(question)
        logger.info(f"Extracted entities from query: {entities}")

        prioritized = {
            "query": question,
            "entities": [ent for ent in entities if ent[1] in {"PERSON", "ORG", "GPE"}]
        }
        return prioritized

    def process_with_chain_parallel(self, question: str) -> dict:
        """
        Processes the question using advanced context retrieval and parallelized document chaining.

        Args:
            question (str): User's question.

        Returns:
            dict: Final answer and confidence score.
        """
        try:
            logger.info(f"Processing question with advanced retrieval and parallel chaining: {question}")

            # Check if FAISS index is populated
            if self.embedding_handler.index is None or self.embedding_handler.index.ntotal == 0:
                raise ValueError("Cannot search. The FAISS index is empty.")

            # Step 1: Generate query embedding
            logger.info(f"Generating embedding for the query: {question}")
            query_embedding = self.embedding_handler.generate_embedding(question)

            # Step 2: Perform FAISS search to retrieve relevant chunks
            logger.info("Performing FAISS search for relevant chunks.")
            faiss_results = self.embedding_handler.search(query_embedding, top_k=10)

            if not faiss_results:
                logger.warning("No relevant chunks found during FAISS search.")
                return {"question": question, "answer": "Data Not Available", "confidence": 0.0}

            # Extract chunks from FAISS results
            chunks = [result["metadata"]["chunk"] for result in faiss_results]

            # Step 3: Process chunks in parallel to refine the answer
            def refine_chunk(chunk):
                """
                Processes a single chunk to refine the answer.
                """
                context = f"Chunk context:\n{chunk}\n\n"
                response = self.llm_handler.generate_answer(question, [context])
                return response["answer"]

            logger.info(f"Processing {len(chunks)} chunks in parallel.")
            with ThreadPoolExecutor() as executor:
                refined_answers = list(executor.map(refine_chunk, chunks))

            # Step 4: Deduplicate answers
            logger.info("Deduplicating answers.")
            unique_answers = self._deduplicate_answers(refined_answers)

            # Step 5: Aggregate results into a final answer
            final_answer = " ".join(unique_answers).strip()
            logger.info(f"Final answer after advanced retrieval and parallel chaining: {final_answer}")

            # Return the result
            return {"question": question, "answer": final_answer, "confidence": 0.9}
        except Exception as e:
            logger.error(f"{str(e)}", exc_info=True)
            logger.error(f"Error during advanced retrieval and parallel chaining: {str(e)}", exc_info=True)
            return {"question": question, "answer": "Data Not Available", "confidence": 0.0}


    def _deduplicate_answers(self, answers):
        """
        Removes redundant answers using similarity-based filtering.

        Args:
            answers (list): List of answers.

        Returns:
            list: List of unique answers.
        """
        unique_answers = []
        embeddings = self._generate_embeddings(answers)

        for i, answer in enumerate(answers):
            is_duplicate = False
            for j in range(len(unique_answers)):
                similarity = cosine_similarity(
                    embeddings[i].unsqueeze(0), self._generate_embeddings([unique_answers[j]])
                )[0][0]
                if similarity > 0.85:  # Threshold for deduplication
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_answers.append(answer)

        return unique_answers

    def _generate_embeddings(self, texts):
        """
        Generate embeddings using a Hugging Face Transformers model.

        Args:
            texts (list of str): List of text inputs.

        Returns:
            torch.Tensor: Embeddings for each text.
        """
        with torch.no_grad():
            # Tokenize inputs
            encoded_input = self.tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt", max_length=512
            )

            # Pass inputs through the model
            model_output = self.model(**encoded_input)

            # Use the mean pooling of token embeddings (excluding [PAD] tokens)
            input_mask_expanded = encoded_input["attention_mask"].unsqueeze(-1).expand(model_output.last_hidden_state.size())
            sum_embeddings = torch.sum(model_output.last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            return sum_embeddings / sum_mask

    
    def process_pdf(self, file_path):
        """
        Processes the PDF and populates the FAISS index.
        """
        try:

            logger.info(f"Processing PDF: {file_path}")
            file_key = self.embedding_cache.get_cache_key(file_path)

            # Check cache
            cached_data = self.embedding_cache.get_embeddings(file_key)
            if cached_data:
                logger.info("Embeddings found in cache. Adding to FAISS index.")
                self.embedding_handler.add_embeddings(
                    cached_data["embeddings"], cached_data["metadata"]
                )
                logger.info(f"Cached embeddings added. FAISS index total: {self.embedding_handler.index.ntotal}")
                return

            # Normal processing
            start_time = time.time()
            text = self.pdf_processor.extract_text()
            chunks = self.pdf_processor.preprocess_text()

            logger.info(f"Generating embeddings for {len(chunks)} chunks in parallel...")

            # Parallel embedding generation
            with ThreadPoolExecutor() as executor:
                embeddings = list(executor.map(self.llm_handler.generate_embedding, chunks))

            metadata = [{"chunk": chunk} for chunk in chunks]

            self.embedding_handler.add_embeddings(embeddings, metadata)
            self.embedding_cache.save_embeddings(file_key, embeddings, metadata)
            total_time = time.time() - start_time
            logger.info(f"Added {len(embeddings)} embeddings to FAISS index in {total_time}")
        except Exception as e:
            logger.error(f"{str(e)}", exc_info=True)
            logger.error(f"Error during PDF processing: {str(e)}", exc_info=True)
            raise

