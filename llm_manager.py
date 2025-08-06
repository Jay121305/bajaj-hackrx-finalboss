import os
import logging
import time
import json
from typing import Dict, List, Any, Optional
import requests

logger = logging.getLogger(__name__)

class LLMManager:
    """Manage interactions with large language models"""
    
    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY")
        self.default_model = os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo")
        self.premium_model = os.getenv("PREMIUM_LLM_MODEL", "gpt-4")
        self.api_base = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
        
        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "avg_response_time": 0,
            "requests_by_domain": {},
            "error_count": 0,
            "cache_hits": 0
        }
        
        # Simple response cache
        self.response_cache = {}
        self.max_cache_size = 1000
        
        # Domain-specific models
        self.domain_models = {
            "legal": self.premium_model,
            "insurance": self.premium_model,
            "finance": self.premium_model,
            "compliance": self.premium_model,
            "medical": self.premium_model,
            "general": self.default_model,
            "hr": self.default_model
        }
        
        # Analysis output formatters
        self.output_formatters = {
            "json": self._format_json_output,
            "markdown": self._format_markdown_output,
            "html": self._format_html_output
        }
    
    async def initialize(self):
        """Initialize LLM connections and resources"""
        logger.info(f"Initializing LLM manager with default model: {self.default_model}")
        
        # Validate API key
        if not self.api_key:
            logger.warning("LLM_API_KEY environment variable not set. Limited functionality.")
            
        # Test connection if API key available
        if self.api_key:
            try:
                # Simple test call to verify connectivity
                test_result = await self._send_test_request()
                logger.info("LLM API connection verified successfully")
            except Exception as e:
                logger.error(f"Error connecting to LLM API: {str(e)}")
    
    async def analyze_query(
        self, 
        prompt: str, 
        domain: Optional[str] = None,
        model: Optional[str] = None,
        output_format: str = "json",
        optimize_for_throughput: bool = False
    ) -> Dict[str, Any]:
        """Send query to LLM for analysis and return structured response"""
        start_time = time.time()
        
        try:
            # Check cache first for identical prompts
            cache_key = hash(prompt)
            if cache_key in self.response_cache and not optimize_for_throughput:
                self.stats["cache_hits"] += 1
                logger.info(f"Cache hit for prompt")
                return self.response_cache[cache_key]
            
            # Determine which model to use based on domain and override
            selected_model = model or self.domain_models.get(domain or "general", self.default_model)
            
            # If optimizing for throughput, use faster model
            if optimize_for_throughput:
                selected_model = self.default_model
            
            # Send request to LLM API
            response = await self._call_llm_api(prompt, selected_model)
            
            # Process and structure the response
            structured_response = self._process_llm_response(response, domain, output_format)
            
            # Update statistics
            elapsed_time = time.time() - start_time
            self.stats["total_requests"] += 1
            tokens_used = response.get("usage", {}).get("total_tokens", 0)
            self.stats["total_tokens"] += tokens_used
            
            # Update average response time
            self.stats["avg_response_time"] = (
                (self.stats["avg_response_time"] * (self.stats["total_requests"] - 1) + elapsed_time) / 
                self.stats["total_requests"]
            )
            
            # Update domain stats
            if domain:
                self.stats["requests_by_domain"][domain] = self.stats["requests_by_domain"].get(domain, 0) + 1
            
            # Add to cache if not optimizing for throughput
            if not optimize_for_throughput:
                self.response_cache[cache_key] = structured_response
                
                # Prune cache if it gets too large
                if len(self.response_cache) > self.max_cache_size:
                    # Remove a random key - more sophisticated LRU could be implemented
                    self.response_cache.pop(next(iter(self.response_cache)))
            
            # Add performance metadata
            structured_response["performance"] = {
                "response_time": elapsed_time,
                "tokens_used": tokens_used,
                "model": selected_model
            }
            
            logger.info(f"Query analyzed in {elapsed_time:.2f}s using {tokens_used} tokens")
            
            return structured_response
            
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            self.stats["error_count"] += 1
            
            # Return error response
            return {
                "answer": f"Analysis error: {str(e)}",
                "confidence": 0,
                "supporting_clauses": [],
                "error": True,
                "tokens_used": 0
            }
    
    async def _call_llm_api(self, prompt: str, model: str) -> Dict[str, Any]:
        """Make actual API call to LLM provider"""
        # This implementation uses OpenAI's API format
        # For other LLM providers, this method would need to be modified
        try:
            if not self.api_key:
                return self._simulate_llm_response(prompt)  # Fallback for testing
                
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert document analyzer. Provide accurate information based on the document excerpts."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,  # Low temperature for more consistent, factual responses
                "max_tokens": 1000
            }
            
            response = requests.post(
                f"{self.api_base}/chat/completions", 
                headers=headers, 
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"LLM API error: {response.status_code} - {response.text}")
                raise Exception(f"API error: {response.status_code}")
                
            return response.json()
            
        except Exception as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            raise
    
    def _process_llm_response(
        self, 
        response: Dict[str, Any], 
        domain: Optional[str] = None,
        output_format: str = "json"
    ) -> Dict[str, Any]:
        """Process and structure the raw LLM response"""
        try:
            # Extract the main content from the response
            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"]
            else:
                content = "No response generated"
                
            # Parse structured output if response is in JSON format
            structured_data = self._try_parse_json(content)
            
            if structured_data and isinstance(structured_data, dict):
                # Use the parsed JSON directly
                result = structured_data
                # Ensure all required fields are present
                if "answer" not in result:
                    result["answer"] = content
                if "supporting_clauses" not in result:
                    result["supporting_clauses"] = []
                if "confidence" not in result:
                    result["confidence"] = 0.7
            else:
                # If not JSON, create structured output from the raw text
                clauses = self._extract_supporting_clauses(content)
                confidence = self._estimate_confidence(content, domain)
                
                result = {
                    "answer": content,
                    "supporting_clauses": clauses,
                    "confidence": confidence
                }
            
            # Add domain if provided
            if domain:
                result["domain"] = domain
                
            # Add token usage
            if "usage" in response:
                result["tokens_used"] = response["usage"].get("total_tokens", 0)
                
            # Format output if requested
            formatter = self.output_formatters.get(output_format)
            if formatter:
                result["formatted_output"] = formatter(result)
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing LLM response: {str(e)}")
            return {
                "answer": "Error processing response",
                "supporting_clauses": [],
                "confidence": 0,
                "error": True
            }
    
    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Try to parse JSON from text, handling various formats"""
        try:
            # Check if the entire response is valid JSON
            return json.loads(text)
        except:
            pass
            
        try:
            # Check for JSON block in markdown format
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if json_match:
                return json.loads(json_match.group(1))
        except:
            pass
            
        return None
    
    def _extract_supporting_clauses(self, content: str) -> List[Dict[str, Any]]:
        """Extract supporting clauses/references from response text"""
        clauses = []
        
        # Look for quoted text which often indicates document excerpts
        quote_matches = re.findall(r'"([^"]+)"', content)
        for i, quote in enumerate(quote_matches):
            if len(quote) > 10:  # Ignore very short quotes
                clauses.append({
                    "text": quote,
                    "confidence": 0.8,
                    "type": "quote"
                })
        
        # Look for paragraph references
        para_matches = re.findall(r'paragraph\s+(\d+)', content, re.IGNORECASE)
        section_matches = re.findall(r'section\s+([a-z0-9\.]+)', content, re.IGNORECASE)
        
        for para in para_matches:
            clauses.append({
                "text": f"Paragraph {para}",
                "confidence": 0.7,
                "type": "reference"
            })
            
        for section in section_matches:
            clauses.append({
                "text": f"Section {section}",
                "confidence": 0.7,
                "type": "reference"
            })
        
        return clauses
    
    def _estimate_confidence(self, content: str, domain: Optional[str] = None) -> float:
        """Estimate confidence level based on response content"""
        # Base confidence
        confidence = 0.7
        
        # Check for uncertainty markers
        uncertainty_phrases = [
            "not clear", "unclear", "ambiguous", "cannot determine",
            "not specified", "not mentioned", "no information", "unknown",
            "might", "maybe", "perhaps", "possible", "likely", "unlikely"
        ]
        
        # Count uncertainty markers
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in content.lower())
        
        # Adjust confidence based on uncertainty
        if uncertainty_count > 3:
            confidence -= 0.3
        elif uncertainty_count > 0:
            confidence -= 0.1 * uncertainty_count
            
        # Check for certainty markers
        certainty_phrases = [
            "clearly states", "explicitly", "specifically", "directly",
            "definitively", "certainly", "without doubt", "unambiguous"
        ]
        
        # Count certainty markers
        certainty_count = sum(1 for phrase in certainty_phrases if phrase in content.lower())
        
        # Adjust confidence based on certainty
        confidence += 0.05 * certainty_count
        
        # Cap confidence between 0.1 and 0.95
        confidence = max(0.1, min(0.95, confidence))
        
        return confidence
    
    def _simulate_llm_response(self, prompt: str) -> Dict[str, Any]:
        """Simulate LLM response for testing without API key"""
        return {
            "choices": [
                {
                    "message": {
                        "content": "This is a simulated response since no API key was provided. In a real implementation, the LLM would analyze the document content and provide relevant answers based on the query."
                    }
                }
            ],
            "usage": {
                "total_tokens": 50
            }
        }
    
    async def _send_test_request(self) -> Dict[str, Any]:
        """Send a minimal test request to verify API connectivity"""
        try:
            if not self.api_key:
                return {"status": "simulated", "connectivity": "unknown"}
                
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.default_model,
                "messages": [
                    {"role": "user", "content": "Hello, this is a test message."}
                ],
                "max_tokens": 5
            }
            
            response = requests.post(
                f"{self.api_base}/chat/completions", 
                headers=headers, 
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                return {"status": "connected", "connectivity": "verified"}
            else:
                return {"status": "error", "code": response.status_code, "detail": response.text}
                
        except Exception as e:
            logger.error(f"API test connection error: {str(e)}")
            return {"status": "error", "detail": str(e)}
    
    def _format_json_output(self, result: Dict[str, Any]) -> str:
        """Format result as JSON string"""
        return json.dumps(result, indent=2)
    
    def _format_markdown_output(self, result: Dict[str, Any]) -> str:
        """Format result as Markdown"""
        answer = result.get("answer", "")
        confidence = result.get("confidence", 0)
        clauses = result.get("supporting_clauses", [])
        
        md = [f"# Analysis Result\n\n## Answer\n\n{answer}\n"]
        
        if clauses:
            md.append("\n## Supporting Evidence\n")
            for i, clause in enumerate(clauses):
                md.append(f"{i+1}. {clause.get('text', '')}\n")
        
        md.append(f"\n*Confidence: {confidence:.2%}*")
        
        return "\n".join(md)
    
    def _format_html_output(self, result: Dict[str, Any]) -> str:
        """Format result as HTML"""
        answer = result.get("answer", "")
        confidence = result.get("confidence", 0)
        clauses = result.get("supporting_clauses", [])
        
        html = [
            "<div class='analysis-result'>",
            f"<div class='answer'><h3>Answer</h3><p>{answer}</p></div>"
        ]
        
        if clauses:
            html.append("<div class='evidence'><h3>Supporting Evidence</h3><ul>")
            for clause in clauses:
                html.append(f"<li>{clause.get('text', '')}</li>")
            html.append("</ul></div>")
        
        html.append(f"<div class='confidence'>Confidence: {confidence:.2%}</div>")
        html.append("</div>")
        
        return "".join(html)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get LLM service status and statistics"""
        return {
            "default_model": self.default_model,
            "premium_model": self.premium_model,
            "total_requests": self.stats["total_requests"],
            "total_tokens": self.stats["total_tokens"],
            "avg_response_time": round(self.stats["avg_response_time"], 3),
            "error_rate": self.stats["error_count"] / max(1, self.stats["total_requests"]),
            "cache_hits": self.stats["cache_hits"],
            "domain_requests": self.stats["requests_by_domain"],
            "api_status": "active" if self.api_key else "simulated"
        }