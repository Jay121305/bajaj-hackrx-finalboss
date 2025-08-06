import re
import logging
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class TokenOptimizer:
    """Optimize token usage for LLM queries and context"""
    
    def __init__(self):
        self.optimization_stats = {
            "queries_optimized": 0,
            "tokens_saved": 0,
            "avg_compression_ratio": 0
        }
        
        # Estimated token counts for common words
        self.approx_tokens_per_word = 0.75
        
        # Domain-specific templates for optimal prompting
        self.domain_templates = {
            "insurance": """
            Based on the following insurance document excerpts:
            {context}
            
            Please answer the question accurately and cite the specific policy sections:
            {query}
            """,
            
            "legal": """
            Based on the following legal document excerpts:
            {context}
            
            Please answer the question accurately and cite the specific clauses:
            {query}
            """,
            
            "hr": """
            Based on the following HR document excerpts:
            {context}
            
            Please answer the question accurately and cite the specific policies:
            {query}
            """,
            
            "general": """
            Based on the following document excerpts:
            {context}
            
            Please answer the question accurately and cite your sources:
            {query}
            """
        }
    
    async def optimize_query(self, query: str, domain: Optional[str] = None) -> str:
        """Optimize query to use fewer tokens while preserving intent"""
        try:
            original_token_count = self._estimate_token_count(query)
            
            # Remove unnecessary phrases
            optimized = query
            unnecessary_phrases = [
                "I was wondering if", "I want to know", "Can you tell me", 
                "I would like to know", "Please tell me", "I'm curious about",
                "Could you explain", "I need information on", "I need to know"
            ]
            
            for phrase in unnecessary_phrases:
                optimized = re.sub(f"{phrase}\\s+", "", optimized, flags=re.IGNORECASE)
            
            # Remove redundant question starters after optimization
            if not optimized.endswith("?"):
                optimized = optimized + "?"
                
            optimized = re.sub(r'^(?:what|who|where|when|why|how|is|are|do|does|can|could|would|will|has|have)\s+', "", 
                             optimized, flags=re.IGNORECASE)
            
            # If query becomes too short, revert to original
            if len(optimized) < 10:
                optimized = query
                
            # Update stats if optimization was effective
            optimized_token_count = self._estimate_token_count(optimized)
            tokens_saved = original_token_count - optimized_token_count
            
            if tokens_saved > 0:
                self.optimization_stats["queries_optimized"] += 1
                self.optimization_stats["tokens_saved"] += tokens_saved
                
                # Update average compression ratio
                total_optimized = self.optimization_stats["queries_optimized"]
                current_ratio = tokens_saved / original_token_count
                self.optimization_stats["avg_compression_ratio"] = (
                    (self.optimization_stats["avg_compression_ratio"] * (total_optimized - 1) + current_ratio) / 
                    total_optimized
                )
                
                logger.info(f"Optimized query from {original_token_count} to {optimized_token_count} tokens")
            else:
                # If no tokens saved, use original query
                optimized = query
            
            return optimized
            
        except Exception as e:
            logger.error(f"Error optimizing query: {str(e)}")
            return query  # Return original query if optimization fails
    
    async def generate_optimized_prompt(
        self,
        query: str, 
        context_sections: List[Dict[str, Any]], 
        domain: Optional[str] = None
    ) -> Tuple[str, int]:
        """Generate optimized prompt with context for LLM analysis"""
        try:
            # Select template based on domain
            template = self.domain_templates.get(domain or "general", self.domain_templates["general"])
            
            # Format context based on relevance scores
            formatted_context = self._format_context(context_sections)
            
            # Format the prompt
            prompt = template.strip().format(
                context=formatted_context,
                query=query
            )
            
            # Estimate token count
            token_count = self._estimate_token_count(prompt)
            
            logger.info(f"Generated prompt with approximately {token_count} tokens")
            
            return prompt, token_count
            
        except Exception as e:
            logger.error(f"Error generating optimized prompt: {str(e)}")
            
            # Fallback to simple prompt
            fallback_prompt = f"Based on the document excerpts, answer: {query}"
            return fallback_prompt, self._estimate_token_count(fallback_prompt)
    
    def _format_context(self, context_sections: List[Dict[str, Any]]) -> str:
        """Format context sections into an optimized string"""
        formatted_sections = []
        
        for i, section in enumerate(context_sections):
            # Extract text and metadata
            text = section.get("text", "")
            metadata = section.get("metadata", {})
            
            # Include position/location information if available
            location = ""
            if "section_header" in metadata:
                location = f" [{metadata['section_header']}]"
            elif "position" in metadata:
                location = f" [Section {metadata['position'] + 1}]"
                
            # Format section with excerpt number and location
            formatted_section = f"Excerpt {i+1}{location}:\n{text}\n"
            
            formatted_sections.append(formatted_section)
        
        return "\n".join(formatted_sections)
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for a string"""
        # Simple estimation - actual tokenization would be more accurate
        words = len(text.split())
        return int(words * self.approx_tokens_per_word)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Return current optimization statistics"""
        return {
            "queries_optimized": self.optimization_stats["queries_optimized"],
            "tokens_saved": self.optimization_stats["tokens_saved"],
            "avg_compression_ratio": round(self.optimization_stats["avg_compression_ratio"], 2)
        }