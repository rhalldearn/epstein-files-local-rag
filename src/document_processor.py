"""Document processing module for extracting and chunking PDF text."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
from tqdm import tqdm


class DocumentProcessor:
    """Handles PDF text extraction and chunking."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize the document processor.

        Args:
            chunk_size: Number of tokens per chunk
            overlap: Number of overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract text and metadata from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing text, metadata, and page information
        """
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)  # Get page count before closing
            pages = []

            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():  # Only include pages with text
                    pages.append({
                        "page_num": page_num + 1,
                        "text": text
                    })

            doc.close()

            return {
                "file_path": str(pdf_path),
                "file_name": pdf_path.name,
                "dataset": pdf_path.parent.name,
                "pages": pages,
                "total_pages": total_pages
            }
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return None

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk

        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Simple word-based chunking (approximating tokens)
        words = text.split()
        chunks = []

        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "metadata": metadata.copy()
                })

            start += (self.chunk_size - self.overlap)

        return chunks

    def process_all_documents(self, base_dir: Path, output_file: Path) -> List[Dict[str, Any]]:
        """
        Process all PDFs in the directory structure.

        Args:
            base_dir: Base directory containing PDF files
            output_file: Path to save processed chunks

        Returns:
            List of all chunks with metadata
        """
        pdf_files = list(base_dir.rglob("*.pdf"))
        all_chunks = []

        print(f"Found {len(pdf_files)} PDF files to process")

        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            doc_data = self.extract_text_from_pdf(pdf_path)

            if doc_data is None:
                continue

            # Process each page separately to maintain page-level metadata
            for page in doc_data["pages"]:
                page_metadata = {
                    "file_path": doc_data["file_path"],
                    "file_name": doc_data["file_name"],
                    "dataset": doc_data["dataset"],
                    "page_num": page["page_num"],
                    "total_pages": doc_data["total_pages"]
                }

                page_chunks = self.chunk_text(page["text"], page_metadata)
                all_chunks.extend(page_chunks)

        # Save to disk
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        print(f"Processed {len(all_chunks)} chunks from {len(pdf_files)} PDFs")
        print(f"Saved to {output_file}")

        return all_chunks

    @staticmethod
    def load_processed_chunks(input_file: Path) -> List[Dict[str, Any]]:
        """
        Load previously processed chunks from disk.

        Args:
            input_file: Path to the processed chunks JSON file

        Returns:
            List of chunks
        """
        with open(input_file, "r", encoding="utf-8") as f:
            return json.load(f)
