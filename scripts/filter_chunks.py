"""Filter out very short chunks that are likely poor quality."""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

MIN_CHUNK_LENGTH = 100  # Minimum characters to keep a chunk


def main():
    """Filter processed chunks to remove short, low-quality chunks."""
    project_root = Path(__file__).parent.parent
    input_file = project_root / "processed_chunks.json"
    output_file = project_root / "processed_chunks_filtered.json"

    # Load chunks
    print(f"Loading chunks from {input_file}...")
    with open(input_file) as f:
        chunks = json.load(f)

    original_count = len(chunks)

    # Filter chunks
    filtered_chunks = [c for c in chunks if len(c['text']) >= MIN_CHUNK_LENGTH]

    filtered_count = len(filtered_chunks)
    removed_count = original_count - filtered_count

    # Save filtered chunks
    with open(output_file, 'w') as f:
        json.dump(filtered_chunks, f)

    print(f"\nResults:")
    print(f"  Original chunks: {original_count}")
    print(f"  Filtered chunks: {filtered_count}")
    print(f"  Removed: {removed_count} ({removed_count/original_count*100:.1f}%)")
    print(f"\nSaved to: {output_file}")
    print(f"\nTo use filtered chunks, update scripts/initialize.py to load from:")
    print(f"  processed_chunks_filtered.json")


if __name__ == "__main__":
    main()
