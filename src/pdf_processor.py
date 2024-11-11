from RecursiveCharacterTextSplitter import RecursiveCharacterTextSplitter
import PyPDF2
from logging_config import LoggingConfig

logger = LoggingConfig().get_logger()

class PDFProcessor:
    """
    Handles PDF text extraction and preprocessing.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.text = None
        self.text_chunks = None  # To store preprocessed text chunks

    def extract_text(self) -> str:
        """
        Extracts text from the PDF file.

        Returns:
            str: Extracted text from the PDF.
        """
        logger.info(f"Extracting text from PDF: {self.file_path}")
        try:
            with open(self.file_path, "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                self.text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            logger.info("Text extraction successful.")
            return self.text
        except Exception as e:
            logger.error(f"{str(e)}", exc_info=True)
            logger.error(f"Error extracting text from PDF: {str(e)}", exc_info=True)
            raise ValueError(f"Error extracting text: {str(e)}")

    def preprocess_text(self):
        """
        Preprocesses the extracted text into chunks and stores them in `text_chunks`.
        """
        if not self.text:
            raise ValueError("No text available to preprocess. Ensure text extraction is performed first.")

        try:
            logger.info("Preprocessing extracted text into chunks.")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=60, separator="\n\n")
            self.text_chunks = splitter.split_text(self.text)
            logger.info(f"Generated {len(self.text_chunks)} chunks from the extracted text.")
            return self.text_chunks
        except Exception as e:
            logger.error(f"{str(e)}", exc_info=True)
            logger.error(f"Error during text preprocessing: {str(e)}", exc_info=True)
            raise ValueError(f"Error during text preprocessing: {str(e)}")

    def get_chunks(self) -> list:
        """
        Returns preprocessed text chunks.

        Returns:
            list: List of text chunks.
        """
        if self.text_chunks is None:
            raise ValueError("Text chunks not available. Ensure preprocessing is performed.")
        return self.text_chunks
