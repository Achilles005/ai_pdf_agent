AI-Powered PDF Question-Answering System
This project is an AI-powered system that allows users to upload PDFs and ask questions about the content. The application leverages advanced embedding models, semantic search, and a streamlined architecture to retrieve relevant information and provide accurate answers.

Features
PDF Processing:
Extracts text from PDFs and splits it into manageable chunks for processing.
Question Answering:
Uses AI to answer user queries based on the content of the uploaded PDF.
Contextual Search:
Combines embeddings and semantic search to retrieve the most relevant chunks.
Advanced Deduplication:
Handles redundant and contradictory answers to ensure coherent responses.
Slack Integration:
Posts structured results to a Slack channel.
Streamlit Interface:
User-friendly interface to upload PDFs, ask questions, and view answers.
Directory Structure
Here’s an overview of the project files and their functionalities:

Core Files
agent.py:

Orchestrates the process of answering questions, from chunk retrieval to final response generation.
Handles advanced deduplication and parallel processing for efficiency.
build_app.py:

Initializes and configures the components (e.g., embedding handler, LLM handler, Slack integration).
embedding_cache.py:

Implements caching for embeddings to optimize performance and reduce redundant computations.
embedding_handler.py:

Manages embedding generation and FAISS-based vector search for retrieving relevant chunks.
hybrid_search.py:

(Optional) Implements hybrid search logic for combining traditional keyword search with embeddings.
llm_handler.py:

Handles interaction with the language model for generating answers and embeddings.
logging_config.py:

Configures project-wide logging for better traceability and debugging.
pdf_processor.py:

Extracts text from PDFs and processes it into manageable chunks using a recursive text splitter.
query_understanding.py:

Includes utilities for query understanding (e.g., named entity recognition, advanced parsing).
RecursiveCharacterTextSplitter.py:

Splits text into chunks while maintaining semantic coherence.
slack_integration.py:

Posts results to Slack channels for real-time notifications and sharing.
streamlit_app.py:

Provides a web interface using Streamlit for users to upload PDFs and ask questions.
transformer_embedding_handler.py:

Uses transformer-based models (e.g., BERT) for embedding generation and similarity computation.
Setup Instructions
Requirements
Ensure you have the following installed:

Python 3.9+
Required libraries (see requirements.txt)
Installation
Clone the repository:

```bash
git clone https://github.com/your-repo-name/ai-pdf-qa.git
cd ai-pdf-qa
Install dependencies:
```
```bash
pip install -r requirements.txt
Configure API keys:
```
Set up OpenAI and Slack API keys in a config.json file:
json
{
    "openai_api_key": "your-openai-key",
    "slack_webhook_url": "your-slack-webhook-url"
}
Usage
Streamlit App
Run the Streamlit application:

```bash
streamlit run src/streamlit_app.py
Open your browser and navigate to the provided URL to:
```
Upload a PDF file.
Enter questions.
View the AI-generated answers.
Slack Integration
The system automatically posts results to Slack if configured.
Batch Processing
Use the agent.py or build_app.py modules for batch question processing.
Key Workflows
PDF Upload and Text Extraction:

Text is extracted and processed into chunks using pdf_processor.py and RecursiveCharacterTextSplitter.py.
Question Embedding:

Questions are converted into embeddings using transformer models in transformer_embedding_handler.py.
Chunk Retrieval:

Relevant chunks are retrieved using FAISS in embedding_handler.py.
Answer Generation:

The language model (llm_handler.py) generates answers based on retrieved chunks.
Deduplication:


Evaluation
The project includes evaluation metrics to measure model performance and system accuracy.
Extend llm_handler.py and agent.py to add custom evaluation logic.

Future Improvements
Contradictory or redundant answers are filtered in agent.py.
Optimize further to handle larger PDFs and complex queries.
Enhanced Deduplication:
Use advanced semantic techniques to refine responses.
Hybrid Search:
Incorporate both embedding-based and keyword-based retrieval.
Additional Integrations:
Support more communication platforms (e.g., Microsoft Teams).