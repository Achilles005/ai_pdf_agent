class RecursiveCharacterTextSplitter:
    """
    Splits text into chunks of a specified size with optional overlap.
    """
    def __init__(self, chunk_size=1000, chunk_overlap=60, separator="\n\n"):
        """
        Initializes the text splitter.
        
        Args:
            chunk_size (int): The maximum size of each chunk in characters.
            chunk_overlap (int): The overlap size between consecutive chunks.
            separator (str): The primary separator for splitting text.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def split_text(self, text):
        """
        Splits the text into chunks using recursive splitting.
        
        Args:
            text (str): The input text to split.
        
        Returns:
            list: A list of text chunks.
        """
        chunks = []
        current_position = 0

        while current_position < len(text):
            # Find the chunk end position
            end_position = current_position + self.chunk_size

            if end_position >= len(text):
                # Last chunk: take the remaining text
                chunks.append(text[current_position:])
                break

            # Try splitting at the separator closest to the chunk_size
            split_position = text.rfind(self.separator, current_position, end_position)

            if split_position == -1:
                # If no separator is found, split at chunk_size
                split_position = end_position

            # Add the chunk and adjust the current position
            chunks.append(text[current_position:split_position].strip())
            current_position = split_position - self.chunk_overlap  # Overlap with the previous chunk

        return chunks
