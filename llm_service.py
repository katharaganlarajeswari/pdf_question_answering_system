"""
LLM Service Module
Handles Large Language Model for answer generation
Uses Ollama with llama2 model or fallback to rule-based extraction
"""

import requests
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMService:
    """Handles answer generation using LLM"""
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_base_url = ollama_base_url
        self.model = "llama2"
        self.use_ollama = True
    
    def check_ollama_available(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                logger.info(f"Ollama available with {len(models)} models")
                return True
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
        return False
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer from query and context using LLM"""
        
        if not context_chunks:
            return {
                "answer": "No relevant information found in the document to answer your question.",
                "sources": [],
                "method": "no_context"
            }
        
        # Combine context chunks into a single context
        context = "\n\n".join([chunk['text'] for chunk in context_chunks])
        
        # Check if Ollama is available
        if not self.check_ollama_available():
            logger.info("Ollama not available, using fallback extraction method")
            return self._fallback_answer(query, context_chunks)
        
        # Try to generate with Ollama
        try:
            return self._generate_with_ollama(query, context, context_chunks)
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return self._fallback_answer(query, context_chunks)
    
    def _generate_with_ollama(self, query: str, context: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer using Ollama API"""
        
        prompt = f"""Based on the following context from the document, please answer the user's question.

Context:
{context}

Question: {query}

Instructions:
- Answer the question based ONLY on the provided context
- If the answer is not in the context, say "I couldn't find the answer in the document"
- Provide a clear and concise answer
- If there are multiple relevant sections, combine them logically

Answer:"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 500
            }
        }
        
        response = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API returned status {response.status_code}")
        
        result = response.json()
        answer = result.get("response", "").strip()
        
        # Extract source information
        sources = []
        for chunk in context_chunks:
            source_text = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
            sources.append({
                "text": source_text,
                "chunk_id": chunk.get('metadata', {}).get('chunk_id', 'unknown')
            })
        
        return {
            "answer": answer if answer else "I couldn't generate a proper answer from the context.",
            "sources": sources,
            "method": "ollama"
        }
    
    def _fallback_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback answer generation using keyword extraction and context matching"""
        
        # Simple keyword-based extraction
        query_keywords = set(query.lower().split())
        
        best_chunk = None
        best_score = 0
        
        for chunk in context_chunks:
            chunk_words = set(chunk['text'].lower().split())
            # Calculate keyword overlap
            overlap = len(query_keywords & chunk_words)
            if overlap > best_score:
                best_score = overlap
                best_chunk = chunk
        
        if best_chunk and best_score > 0:
            # Extract relevant sentences
            sentences = best_chunk['text'].split('. ')
            relevant_sentences = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in query_keywords):
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                answer = '. '.join(relevant_sentences[:3])
            else:
                # Use first part of best chunk
                answer = best_chunk['text'][:500] + "..." if len(best_chunk['text']) > 500 else best_chunk['text']
        else:
            answer = "I couldn't find relevant information in the document to answer your question."
        
        sources = []
        for chunk in context_chunks[:3]:
            source_text = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
            sources.append({
                "text": source_text,
                "chunk_id": chunk.get('metadata', {}).get('chunk_id', 'unknown')
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "method": "keyword_extraction"
        }

