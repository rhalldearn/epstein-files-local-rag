"""LLM module for Llama 3.2 3B integration."""

import os
from pathlib import Path
from typing import List, Dict, Any
import requests
from tqdm import tqdm


class LLM:
    """Wrapper for Llama 3.2 3B model using llama-cpp-python."""

    MODEL_URL = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    MODEL_FILENAME = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"

    def __init__(self, model_dir: Path, n_ctx: int = 4096, n_gpu_layers: int = -1):
        """
        Initialize the LLM.

        Args:
            model_dir: Directory to store/load the model
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
        """
        self.model_dir = model_dir
        self.model_path = model_dir / self.MODEL_FILENAME
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.llm = None

    def download_model(self):
        """Download the GGUF model if not present."""
        if self.model_path.exists():
            print(f"Model already exists at {self.model_path}")
            return

        print(f"Downloading model from HuggingFace...")
        print(f"URL: {self.MODEL_URL}")
        print(f"Destination: {self.model_path}")

        self.model_dir.mkdir(parents=True, exist_ok=True)

        response = requests.get(self.MODEL_URL, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(self.model_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        print(f"Model downloaded successfully to {self.model_path}")

    def initialize(self):
        """Initialize the Llama model."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Run download_model() first."
            )

        print("Loading Llama 3.2 3B model...")
        print(f"GPU layers: {self.n_gpu_layers}")
        print(f"Context size: {self.n_ctx}")

        try:
            from llama_cpp import Llama

            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )

            print("Model loaded successfully")

        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. "
                "Install with: pip install llama-cpp-python"
            )

    def format_rag_prompt(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Format the RAG prompt with context and question.

        Args:
            question: User's question
            context_chunks: List of relevant document chunks

        Returns:
            Formatted prompt string
        """
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            metadata = chunk.get("metadata", {})
            source = f"{metadata.get('file_name', 'Unknown')} (page {metadata.get('page_num', '?')})"
            text = chunk["text"][:500]  # Limit chunk length
            context_parts.append(f"[Source {i}: {source}]\n{text}")

        context = "\n\n".join(context_parts)

        # Llama 3.2 prompt format
        system_message = """You are a helpful assistant that answers questions about the Epstein Files based on provided documents.

Instructions:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information, say so
- Cite sources by their number [Source 1], [Source 2], etc.
- Be factual and objective
- Do not speculate beyond what's in the documents"""

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

Context from documents:

{context}

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        return prompt

    def generate(self, question: str, context_chunks: List[Dict[str, Any]],
                 max_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Generate an answer to the question using RAG context.

        Args:
            question: User's question
            context_chunks: List of relevant document chunks
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated answer text
        """
        if self.llm is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        prompt = self.format_rag_prompt(question, context_chunks)

        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|eot_id|>", "<|end_of_text|>"],
            echo=False
        )

        answer = response["choices"][0]["text"].strip()
        return answer

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": "Llama-3.2-3B-Instruct",
            "model_path": str(self.model_path),
            "quantization": "Q4_K_M",
            "context_size": self.n_ctx,
            "gpu_layers": self.n_gpu_layers,
            "model_size": f"{self.model_path.stat().st_size / (1024**3):.2f} GB" if self.model_path.exists() else "N/A"
        }
