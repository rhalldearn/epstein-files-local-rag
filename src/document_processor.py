"""Document processing module for extracting and chunking PDF text."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Set
from datetime import datetime
import fitz  # PyMuPDF
from tqdm import tqdm


class DocumentProcessor:
    """Handles PDF text extraction and chunking."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50, checkpoint_interval: int = 100):
        """
        Initialize the document processor.

        Args:
            chunk_size: Number of tokens per chunk
            overlap: Number of overlapping tokens between chunks
            checkpoint_interval: Save progress every N files
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.checkpoint_interval = checkpoint_interval

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

    def _get_checkpoint_path(self, output_file: Path) -> Path:
        """Get the checkpoint file path for a given output file."""
        return output_file.parent / f"{output_file.stem}_checkpoint.json"

    def _get_temp_chunks_path(self, output_file: Path) -> Path:
        """Get the temporary JSONL chunks file path."""
        return output_file.parent / f"{output_file.stem}_temp.jsonl"

    def _load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load checkpoint data if it exists."""
        if checkpoint_path.exists():
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "processed_files": [],
            "total_chunks": 0,
            "last_saved": None,
            "processing_params": {
                "chunk_size": self.chunk_size,
                "overlap": self.overlap
            }
        }

    def _save_checkpoint(self, checkpoint_path: Path, checkpoint_data: Dict[str, Any]):
        """Save checkpoint data to disk."""
        checkpoint_data["last_saved"] = datetime.now().isoformat()
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

    def _append_chunks_to_temp(self, temp_file: Path, chunks: List[Dict[str, Any]]):
        """Append chunks to temporary JSONL file."""
        with open(temp_file, "a", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    def _finalize_chunks_file(self, temp_file: Path, output_file: Path) -> List[Dict[str, Any]]:
        """Convert temporary JSONL file to final JSON array format."""
        all_chunks = []
        if temp_file.exists():
            with open(temp_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_chunks.append(json.loads(line))

        # Write final JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        # Clean up temp file
        if temp_file.exists():
            temp_file.unlink()

        return all_chunks

    def process_all_documents(self, base_dir: Path, output_file: Path, resume: bool = True) -> List[Dict[str, Any]]:
        """
        Process all PDFs in the directory structure with checkpoint support.

        Args:
            base_dir: Base directory containing PDF files
            output_file: Path to save processed chunks
            resume: Whether to resume from checkpoint if available

        Returns:
            List of all chunks with metadata
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_path = self._get_checkpoint_path(output_file)
        temp_file = self._get_temp_chunks_path(output_file)

        # Load checkpoint
        checkpoint = self._load_checkpoint(checkpoint_path)
        processed_files_set = set(checkpoint["processed_files"]) if resume else set()
        total_chunks = checkpoint["total_chunks"] if resume else 0

        # Validate processing parameters match
        if resume and processed_files_set:
            stored_params = checkpoint.get("processing_params", {})
            if (stored_params.get("chunk_size") != self.chunk_size or
                stored_params.get("overlap") != self.overlap):
                print("[Warning] Processing parameters changed. Starting fresh.")
                processed_files_set.clear()
                total_chunks = 0
                if temp_file.exists():
                    temp_file.unlink()

        # Get all PDF files
        pdf_files = list(base_dir.rglob("*.pdf"))

        # Filter out already processed files
        remaining_files = [f for f in pdf_files if str(f) not in processed_files_set]

        print(f"Found {len(pdf_files)} total PDF files")
        if processed_files_set:
            print(f"Resuming: {len(processed_files_set)} already processed, {len(remaining_files)} remaining")
            print(f"Current chunk count: {total_chunks}")

        if not remaining_files:
            print("All files already processed!")
            return self._finalize_chunks_file(temp_file, output_file)

        # Process remaining files
        files_since_checkpoint = 0

        for pdf_path in tqdm(remaining_files, desc="Processing PDFs", initial=len(processed_files_set), total=len(pdf_files)):
            doc_data = self.extract_text_from_pdf(pdf_path)

            if doc_data is None:
                # Still mark as processed to avoid retrying
                processed_files_set.add(str(pdf_path))
                continue

            # Process each page separately to maintain page-level metadata
            file_chunks = []
            for page in doc_data["pages"]:
                page_metadata = {
                    "file_path": doc_data["file_path"],
                    "file_name": doc_data["file_name"],
                    "dataset": doc_data["dataset"],
                    "page_num": page["page_num"],
                    "total_pages": doc_data["total_pages"]
                }

                page_chunks = self.chunk_text(page["text"], page_metadata)
                file_chunks.extend(page_chunks)

            # Append to temporary file
            self._append_chunks_to_temp(temp_file, file_chunks)
            total_chunks += len(file_chunks)

            # Mark file as processed
            processed_files_set.add(str(pdf_path))
            files_since_checkpoint += 1

            # Save checkpoint periodically
            if files_since_checkpoint >= self.checkpoint_interval:
                checkpoint["processed_files"] = list(processed_files_set)
                checkpoint["total_chunks"] = total_chunks
                self._save_checkpoint(checkpoint_path, checkpoint)
                files_since_checkpoint = 0

        # Final checkpoint save
        checkpoint["processed_files"] = list(processed_files_set)
        checkpoint["total_chunks"] = total_chunks
        self._save_checkpoint(checkpoint_path, checkpoint)

        # Finalize: convert JSONL to JSON array
        print(f"\nFinalizing: Converting {total_chunks} chunks to final format...")
        all_chunks = self._finalize_chunks_file(temp_file, output_file)

        # Clean up checkpoint after successful completion
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        print(f"Processed {total_chunks} chunks from {len(pdf_files)} PDFs")
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
