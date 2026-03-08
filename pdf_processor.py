"""
PDF Processor Module
Handles PDF text extraction and chunking for the RAG system
Supports both text-based and image-based (scanned) PDFs
"""

import io
import pdfplumber
from PyPDF2 import PdfReader
from typing import List, Dict, Any
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional OCR dependencies
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available. Install with: pip install easyocr")

try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not available. Install with: pip install pdf2image")


class PDFProcessor:
    """Handles PDF document processing including text extraction and chunking"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ocr_reader = None
        
    def _get_ocr_reader(self):
        """Lazy initialization of EasyOCR reader"""
        if self.ocr_reader is None and EASYOCR_AVAILABLE:
            logger.info("Initializing EasyOCR reader...")
            # Initialize with English, use GPU if available
            self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("EasyOCR reader initialized")
        return self.ocr_reader
    
    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """Extract text content from PDF file"""
        text = ""
        
        # First try pdfplumber (works for text-based PDFs)
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if len(text.strip()) > 100:  # If substantial text found
                logger.info(f"Extracted text using pdfplumber: {len(text)} characters")
                return text
            else:
                logger.info("pdfplumber extracted minimal text, trying OCR...")
                
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
        
        # Try PyPDF2 as fallback
        try:
            pdf_reader = PdfReader(io.BytesIO(file_bytes))
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            if len(text.strip()) > 100:
                logger.info(f"Extracted text using PyPDF2: {len(text)} characters")
                return text
                
        except Exception as e:
            logger.warning(f"PyPDF2 also failed: {e}")
        
        # Try OCR for image-based/scanned PDFs
        if EASYOCR_AVAILABLE and PDF2IMAGE_AVAILABLE:
            try:
                logger.info("Attempting OCR for scanned PDF...")
                text = self._extract_text_with_ocr(file_bytes)
                if text and len(text.strip()) > 100:
                    logger.info(f"Extracted text using OCR: {len(text)} characters")
                    return text
            except Exception as e:
                logger.error(f"OCR extraction failed: {e}")
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF. The PDF may be empty or encrypted.")
        
        return text
    
    def _extract_text_with_ocr(self, file_bytes: bytes) -> str:
        """Extract text using OCR from PDF images"""
        try:
            # Convert PDF to images
            images = convert_from_bytes(file_bytes, dpi=300)
            
            ocr_reader = self._get_ocr_reader()
            if not ocr_reader:
                return ""
            
            all_text = []
            total_pages = len(images)
            
            logger.info(f"Processing {total_pages} pages with OCR...")
            
            for i, image in enumerate(images):
                logger.info(f"OCR processing page {i+1}/{total_pages}")
                
                # Convert PIL image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                # Run OCR
                results = ocr_reader.readtext(img_byte_arr.getvalue())
                
                # Extract text from results
                page_text = ' '.join([result[1] for result in results])
                all_text.append(page_text)
            
            return '\n\n'.join(all_text)
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks with metadata"""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        "id": chunk_id,
                        "text": current_chunk.strip(),
                        "char_count": len(current_chunk)
                    })
                    chunk_id += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + " " + para
                else:
                    current_chunk = para
            else:
                current_chunk += " " + para if current_chunk else para
        
        # Add remaining chunk
        if current_chunk:
            chunks.append({
                "id": chunk_id,
                "text": current_chunk.strip(),
                "char_count": len(current_chunk)
            })
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def process_pdf(self, file_bytes: bytes) -> Dict[str, Any]:
        """Complete PDF processing pipeline"""
        # Extract text
        text = self.extract_text_from_pdf(file_bytes)
        
        # Chunk text
        chunks = self.chunk_text(text)
        
        return {
            "full_text": text,
            "chunks": chunks,
            "num_chunks": len(chunks),
            "total_characters": len(text)
        }

