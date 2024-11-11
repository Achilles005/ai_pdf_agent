# src/query_understanding.py
import spacy
from typing import List, Tuple

class QueryUnderstanding:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")

    def extract_entities(self, question: str) -> List[Tuple[str, str]]:
        """
        Extract named entities from the query.

        Args:
            question (str): User's question.

        Returns:
            list: List of tuples containing entities and their labels.
        """
        doc = self.nlp(question)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
