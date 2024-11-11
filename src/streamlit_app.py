import streamlit as st
import json
from pdf_processor import PDFProcessor
from embedding_handler import EmbeddingHandler
from llm_handler import LLMHandler
from agent import AIAgent
from embedding_cache import EmbeddingCache
from logging_config import LoggingConfig
from concurrent.futures import ThreadPoolExecutor
from slack_integration import SlackIntegration
from build_app import build_app
logger = LoggingConfig().get_logger()
open_ai_key , slack_webhook_url = build_app()
slack_webhook_url = slack_webhook_url
def run_app():
    # Streamlit UI
    st.title("AI PDF Question-Answering Agent")
    st.markdown(
        "Upload a PDF document, and ask questions about its content. "
        "The AI will provide context-specific answers."
    )

    # PDF Upload Section
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        try:
            # Save the uploaded file locally
            pdf_file_path = "uploaded_file.pdf"
            with open(pdf_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("PDF uploaded successfully!")

            # Initialize components
            logger.info("Initializing components...")
            pdf_processor = PDFProcessor(file_path=pdf_file_path)
            embedding_handler = EmbeddingHandler(embedding_dim=768)
            llm_handler = LLMHandler(api_key=open_ai_key)
            embedding_cache = EmbeddingCache()
            slack_integration = SlackIntegration(webhook_url=slack_webhook_url)
            agent = AIAgent(pdf_processor, embedding_handler, llm_handler, embedding_cache)
            logger.info("Processing the uploaded PDF...")
            agent.process_pdf(pdf_file_path)
            st.success("PDF processed and indexed for question answering.")
        except Exception as e:
            logger.error(f"{str(e)}", exc_info=True)
            logger.error(f"Error during PDF processing: {str(e)}", exc_info=True)
            st.error(f"An error occurred while processing the PDF: {e}")

    # Question Input Section
    if uploaded_file is not None:
        st.markdown("### Add Questions")
        questions_input_mode = st.radio("Choose how to provide questions:", ["Enter questions", "Upload a text file"])

        if questions_input_mode == "Enter questions":
            questions = st.text_area("Enter your questions, one per line:")
            questions_list = [q.strip() for q in questions.split("\n") if q.strip()]
        elif questions_input_mode == "Upload a text file":
            question_file = st.file_uploader("Upload a text file with questions (one question per line)", type="txt")
            if question_file is not None:
                questions_list = [line.strip() for line in question_file.readlines() if line.strip()]
            else:
                questions_list = []

        if st.button("Get Answers"):
            try:
                if not questions_list:
                    st.error("Please provide at least one question.")
                else:
                    # Parallel question processing
                    output = []
                    for question in questions_list:
                        logger.info(f"Answering question with parallel document chaining: {question}")
                        result = agent.process_with_chain_parallel(question)
                        output.append(result)
                        st.markdown(f"**Q:** {result['question']}\n\n**A:** {result['answer']}\n\n---")

                    # Display JSON output
                    st.json(output)
                    # Post results to Slack
                    structured_message = json.dumps(output, indent=2)
                    slack_integration.post_message(f"Question-Answer Results:\n```{structured_message}```")
                    st.success("Results posted to Slack.")
            except Exception as e:
                logger.error(f"{str(e)}", exc_info=True)
                logger.error(f"Error during question processing: {str(e)}", exc_info=True)
                st.error(f"An error occurred while answering the questions: {e}")


if __name__ == "__main__":
    run_app()
