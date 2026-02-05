"""Enhanced document processor with OCR support for images."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
from tqdm import tqdm
from PIL import Image
import io

# OCR support (optional dependency)
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: pytesseract not installed. Install with: pip install pytesseract")
    print("Also install Tesseract: sudo apt-get install tesseract-ocr (Linux)")


class DocumentProcessorOCR:
    """Enhanced PDF processor with OCR for extracting text from images."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50, use_ocr: bool = True):
        """
        Initialize the document processor with OCR support.

        Args:
            chunk_size: Number of tokens per chunk
            overlap: Number of overlapping tokens between chunks
            use_ocr: Whether to use OCR on images (requires pytesseract)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.use_ocr = use_ocr and OCR_AVAILABLE

        if use_ocr and not OCR_AVAILABLE:
            print("OCR requested but pytesseract not available. Falling back to text-only extraction.")

    def extract_text_from_image(self, image_data: bytes) -> str:
        """
        Extract text from image using OCR.

        Args:
            image_data: Raw image bytes

        Returns:
            Extracted text from image
        """
        if not self.use_ocr:
            return ""

        try:
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data))

            # Perform OCR
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            print(f"OCR error: {e}")
            return ""

    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract text and metadata from a PDF file, including OCR on images.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing text, metadata, and page information
        """
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            pages = []

            for page_num in range(total_pages):
                page = doc[page_num]

                # Extract native text
                text = page.get_text()

                # If OCR is enabled and text is minimal, try OCR on images
                if self.use_ocr and len(text.strip()) < 100:  # Threshold for "minimal text"
                    image_list = page.get_images()

                    if image_list:
                        ocr_texts = []
                        for img_index in image_list:
                            xref = img_index[0]

                            try:
                                # Extract image
                                base_image = doc.extract_image(xref)
                                image_bytes = base_image["image"]

                                # Perform OCR
                                ocr_text = self.extract_text_from_image(image_bytes)
                                if ocr_text:
                                    ocr_texts.append(ocr_text)
                            except Exception as e:
                                print(f"Error extracting image from {pdf_path} page {page_num + 1}: {e}")
                                continue

                        # Combine OCR text with any native text
                        if ocr_texts:
                            text = text + "\n\n[OCR Extracted Content]\n" + "\n\n".join(ocr_texts)

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
        if self.use_ocr:
            print("OCR enabled - this will take longer but extract more text")

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
