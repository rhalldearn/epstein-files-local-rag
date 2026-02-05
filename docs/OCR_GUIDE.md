# Improving Text Extraction with OCR

## The Problem

Your Epstein Files PDFs contain a mixture of text and images. Analysis shows:

- **Every page contains at least one image** (often scanned documents)
- **20% of chunks have < 50 characters** - indicating poor extraction
- **Text per page varies from 21 to 2688 characters** - inconsistent quality
- **Important information is trapped in images** that the current system ignores

### Example Problem PDF

```
EFTA00005783.pdf:
  - 35 pages
  - 35 images
  - Only 51 characters per page extracted
  - ~1,800 total characters from 35 pages (most content is in images!)
```

The current `document_processor.py` uses:
```python
text = page.get_text()  # Only extracts native text, ignores images
```

## Why Training the Model Won't Help

**You don't need to train/fine-tune Llama 3.2 3B.** Here's why:

1. ✅ **The model works fine** - it can read and understand text perfectly
2. ❌ **The data never reaches the model** - text in images is never extracted
3. ❌ **Training is expensive** - requires GPUs, time, and expertise
4. ❌ **Training won't fix the root cause** - you need better data extraction, not a better model

### RAG Systems Don't Need Domain Training

In Retrieval Augmented Generation (RAG):
- The model's job is to **read provided context** and answer questions
- The model doesn't need **domain knowledge** - it gets that from retrieved documents
- If the model can't answer, it's because **relevant text wasn't retrieved** (extraction problem)

## The Solution: OCR (Optical Character Recognition)

OCR extracts text **from images** in your PDFs. This will dramatically improve results.

### Quick Test

See the difference OCR makes:

```bash
python tools/test_ocr_extraction.py
```

This will show you how much more text can be extracted from sample PDFs.

## Implementation Options

### Option 1: Use Built-in OCR Enhancement (Recommended)

I've created an enhanced document processor with OCR support.

#### Step 1: Install OCR Dependencies

```bash
# Install Python packages
pip install pytesseract pillow

# Install Tesseract OCR engine
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract

# Windows:
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

#### Step 2: Re-initialize with OCR

```bash
python -m scripts.initialize_with_ocr
```

This will:
- Process all PDFs with OCR enabled
- Extract text from images in addition to native text
- Rebuild the vector index with the enhanced data
- Take **20-30 minutes** (vs 5-10 without OCR) but extract significantly more text

#### Step 3: Use the Chatbot Normally

```bash
./run_chatbot.sh
```

The chatbot now has access to text extracted from images!

### Option 2: Alternative OCR Solutions

If Tesseract doesn't work well, consider:

**EasyOCR** - Better for handwritten text:
```bash
pip install easyocr
```

**Cloud OCR APIs** - Higher accuracy but require internet:
- Google Cloud Vision API
- AWS Textract
- Azure Computer Vision

**PaddleOCR** - Fast and accurate:
```bash
pip install paddleocr
```

### Option 3: Use a Multimodal Model

If the PDFs contain **visual information** (photos, diagrams) that text alone can't capture, you need a vision-language model:

**LLaVA** - Open source vision-language model:
- Can "see" images and answer questions about them
- Requires more GPU memory (8-16GB VRAM)
- Slower than text-only models

**GPT-4 Vision (API)** - Highest quality but requires API key:
- Can process PDF pages as images
- Best for understanding complex visual layouts
- Pay per API call

## Comparing Results

After enabling OCR, you should see:

```
Chunks without OCR: 6,782
Chunks with OCR:    ~15,000-20,000  (estimated 2-3x improvement)

Short chunks (< 50 chars):
  Without OCR: 20.4%
  With OCR:    5-10% (estimated)
```

## Technical Details

### How the Enhanced Processor Works

```python
# 1. Extract native text first
text = page.get_text()

# 2. If text is minimal (< 100 chars), check for images
if len(text.strip()) < 100:
    images = page.get_images()

    # 3. For each image, perform OCR
    for image in images:
        image_bytes = doc.extract_image(image_xref)
        ocr_text = pytesseract.image_to_string(image_bytes)

        # 4. Combine native text + OCR text
        text = text + "\n\n[OCR Extracted]\n" + ocr_text
```

### OCR Threshold

The system performs OCR when:
- A page has < 100 characters of native text
- AND the page contains images

This balances thoroughness with processing time.

### Adjusting OCR Behavior

Edit `src/document_processor_ocr.py` to tune:

```python
# Always OCR all images (slower but more thorough)
if len(text.strip()) < 1000:  # Higher threshold

# Only OCR when almost no text
if len(text.strip()) < 20:  # Lower threshold

# OCR every page regardless
if True:  # Force OCR on everything
```

## Performance Considerations

### Processing Time

- **Without OCR**: 5-10 minutes for 525 PDFs
- **With OCR**: 20-30 minutes (or more)

OCR is CPU-intensive. On a 8-core CPU:
- ~2-4 seconds per page
- 525 PDFs × average 5 pages = ~2,600 pages
- Estimated: 2 hours worst case

### Optimization Tips

1. **Process in parallel**: Modify the script to use multiprocessing
2. **Use GPU OCR**: PaddleOCR supports CUDA acceleration
3. **Cache results**: The processed chunks are saved, so you only do this once
4. **Start with a subset**: Test on one dataset first

## Troubleshooting

### Tesseract Not Found

```
Error: pytesseract.pytesseract.TesseractNotFoundError
```

**Solution**: Install Tesseract OCR engine (not just the Python package)
```bash
sudo apt-get install tesseract-ocr  # Linux
brew install tesseract              # macOS
```

### Poor OCR Quality

If OCR results are garbled:

1. **Check image quality**: Low-resolution scans won't OCR well
2. **Try preprocessing**: Increase contrast, remove noise
3. **Use better OCR**: Try EasyOCR or cloud APIs
4. **Adjust Tesseract config**: Use different PSM modes

### Out of Memory

If processing crashes:

1. **Process in smaller batches**: Modify batch_size in script
2. **Close images after OCR**: Ensure proper cleanup
3. **Reduce concurrent processing**: Lower multiprocessing workers

## Alternative: Vision-Language Model

If OCR still doesn't work well (e.g., handwritten notes, complex layouts), consider a multimodal approach:

### Using LLaVA (Locally)

```python
# Process PDFs as images
for page in doc:
    # Convert page to image
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Ask vision model about the image
    response = llava_model.query(img, "What text is in this image?")
```

**Pros**: Can handle handwriting, complex layouts, visual context
**Cons**: Requires 12GB+ VRAM, slower, more complex setup

## Recommended Approach

1. ✅ **Start with Tesseract OCR** (Option 1) - simple, free, good for typed text
2. If results are still poor, **try EasyOCR or PaddleOCR** - better for varied fonts
3. If visual understanding is needed, **consider vision-language models** - handles layout/diagrams
4. **Never fine-tune the base model** - it won't solve extraction problems

## Summary

| Issue | Solution | Don't Do This |
|-------|----------|---------------|
| Text in images not extracted | Add OCR (Tesseract, EasyOCR) | Fine-tune the model |
| Poor OCR quality | Try better OCR engine, preprocess images | Train the model on PDFs |
| Visual information lost | Use vision-language model (LLaVA) | Hope the text model learns to "see" |
| Slow processing | Optimize OCR, use GPU acceleration | Skip OCR entirely |

**Bottom line**: Fix the data pipeline (add OCR), not the model. The model is fine - it just needs better input data.
