import re
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from typing import List, Dict, Any
import argparse
from pathlib import Path
from loguru import logger

class ErrorClassifier:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the error classifier with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.error_messages = []
        self.error_counts = defaultdict(int)
        self.similarity_threshold = 0.96  # the similarity threshold for grouping errors

    def load_errors_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load error messages from a JSONL file (one JSON object per line)."""
        error_data = []
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                error_data.append(json.loads(line))
        
        logger.info(f"Loaded {len(error_data)} error records from {file_path}")
        return error_data

    def preprocess_error(self, error_message: str) -> str:
        """Preprocess the error message to normalize timestamps, device IDs, and file paths."""
        # Normalize timestamps (various common formats)
        error_message = re.sub(
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?',
            '<TIMESTAMP>',
            error_message
        )
        
        # Skip normalizing paths that appear to be in stack traces (code files)
        def is_stack_trace_path(path):
            return re.search(r'\.(?:py|js|java|cpp|h|cs|rb|go|rs|tsx?|jsx?)[:)]', path) is not None
        
        # Split the message into lines to process each separately
        lines = error_message.split('\n')
        processed_lines = []
        
        for line in lines:
            # Skip normalization for lines containing stack trace paths
            if any(is_stack_trace_path(word) for word in line.split()):
                processed_lines.append(line)
                continue
                
            # Normalize file paths
            # S3 paths
            line = re.sub(
                r's3://[\w.-]+/[\w./%-]+',
                '<S3_PATH>',
                line
            )
            # GCS paths
            line = re.sub(
                r'gs://[\w.-]+/[\w./%-]+',
                '<GCS_PATH>',
                line
            )
                        # Image files
            line = re.sub(
                r'(?:[A-Za-z]:)?[\\/](?:[\w.-]+[\\/])*[\w.-]+\.(?:jpg|jpeg|png|gif|bmp|tiff)',
                '<IMAGE_FILE>',
                line
            )
            
            # Config files
            line = re.sub(
                r'(?:[A-Za-z]:)?[\\/](?:[\w.-]+[\\/])*[\w.-]+\.(?:conf|cfg|ini|env)',
                '<CONFIG_FILE>',
                line
            )
            
            processed_lines.append(line)
        
        return '\n'.join(processed_lines).strip()

    def build_index(self, error_data: List[Dict[str, Any]]):
        """Build the FAISS index from error messages."""
        # Extract and preprocess error messages
        self.error_messages = [
            self.preprocess_error(item['errorMessage'])
            for item in error_data
            if 'errorMessage' in item
        ]

        logger.info(f"Error messages size: {len(self.error_messages)}")
        if not self.error_messages:
            return

        # Generate embeddings
        embeddings = self.model.encode(self.error_messages)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Initialize the index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(embeddings)

    def classify_errors(self):
        """Classify errors and return detailed grouping information."""
        if not self.index or not self.error_messages:
            return {}

        processed_indices = set()
        error_groups = {}

        for i, error_msg in enumerate(self.error_messages):
            if i in processed_indices:
                continue

            # Search for similar errors
            embedding = self.model.encode([error_msg])
            faiss.normalize_L2(embedding)
            D, I = self.index.search(embedding, len(self.error_messages))
            
            # Group similar errors with their similarity scores
            similar_errors = [
                (self.error_messages[idx], sim) 
                for idx, sim in zip(I[0], D[0]) 
                if sim >= self.similarity_threshold
            ]
            
            # Add to processed indices
            processed_indices.update(I[0][D[0] >= self.similarity_threshold])
            
            # Store the group with detailed information
            error_groups[error_msg] = similar_errors

        return error_groups

def main():
    parser = argparse.ArgumentParser(description='Classify similar error messages from a JSON file.')
    parser.add_argument('file_path', type=str, help='Path to the JSON file containing error messages')
    args = parser.parse_args()

    classifier = ErrorClassifier()
    
    # Load and process errors
    error_data = classifier.load_errors_from_file(args.file_path)
    classifier.build_index(error_data) # first 100 errors
    
    # Classify errors and print results
    error_groups = classifier.classify_errors()
    logger.info(f"Error groups number: {len(error_groups)}")    
    
    print("\nError Classification Results:")
    print("-" * 80)
    for representative_error, similar_errors in sorted(error_groups.items(), 
                                                     key=lambda x: len(x[1]), 
                                                     reverse=True):
        print(f"\nGroup Size: {len(similar_errors)}")
        print(f"Representative Error: {representative_error}")
        print(f"Representative Error word count: {len(representative_error.split())}")
        print("\nFirst 10 Similar Errors:")
        for error, similarity in similar_errors[:10]:
            print(f"[{similarity:.3f}] {error}")
        print("-" * 80)

if __name__ == "__main__":
    main()
