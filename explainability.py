import logging
from typing import Dict, List, Any, Optional, Tuple
import json
import re
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class ExplanationEngine:
    """Engine for generating human-readable explanations of query responses"""
    
    def __init__(self):
        self.explanation_templates = {
            "none": "",
            "basic": "Based on the document analysis, {answer}",
            "standard": "Based on the document analysis, {answer} This conclusion is drawn from {sources} {reasoning}",
            "detailed": "Query: \"{query}\"\n\nAnswer: {answer}\n\nAnalysis:\n{detailed_reasoning}\n\nSupporting evidence:\n{evidence_points}\n\nReferences:\n{references}"
        }
        
        self.domain_specific_templates = {
            "insurance": {
                "standard": "Based on the insurance policy document, {answer} According to the {policy_section} section, {key_evidence}. {additional_context}"
            },
            "legal": {
                "standard": "Based on the legal document analysis, {answer} This interpretation is supported by {clause_reference} which states that {key_evidence}. {legal_implications}"
            }
        }
    
    async def generate_explanation(
        self, 
        query: str,
        llm_response: Dict[str, Any],
        context_sections: List[Dict[str, Any]], 
        level: str = "standard"
    ) -> str:
        """Generate an explanation for the answer at the requested detail level"""
        try:
            # Default to standard if invalid level provided
            if level not in self.explanation_templates:
                level = "standard"
                
            # If level is none, return empty explanation
            if level == "none":
                return ""
                
            # Extract required information
            answer = llm_response.get("answer", "")
            confidence = llm_response.get("confidence", 0)
            supporting_clauses = llm_response.get("supporting_clauses", [])
            domain = llm_response.get("domain", "general")
            
            # Generate basic explanation
            if level == "basic":
                return self.explanation_templates["basic"].format(answer=answer)
            
            # Generate standard explanation with template based on domain
            if level == "standard":
                # Try to use domain-specific template if available
                if domain in self.domain_specific_templates and "standard" in self.domain_specific_templates[domain]:
                    template = self.domain_specific_templates[domain]["standard"]
                else:
                    template = self.explanation_templates["standard"]
                
                # Prepare explanation components
                sources = self._format_sources(supporting_clauses)
                reasoning = self._generate_reasoning(query, answer, supporting_clauses, confidence)
                
                # For insurance domain
                policy_section = self._extract_policy_section(supporting_clauses, context_sections)
                key_evidence = self._extract_key_evidence(supporting_clauses)
                additional_context = self._generate_additional_context(context_sections, domain)
                
                # For legal domain
                clause_reference = self._extract_clause_reference(supporting_clauses)
                legal_implications = self._generate_legal_implications(answer, domain)
                
                # Format the template with all possible variables
                explanation = template.format(
                    answer=answer,
                    sources=sources,
                    reasoning=reasoning,
                    policy_section=policy_section,
                    key_evidence=key_evidence,
                    additional_context=additional_context,
                    clause_reference=clause_reference,
                    legal_implications=legal_implications
                )
                
                return explanation
                
            # Generate detailed explanation
            if level == "detailed":
                detailed_reasoning = self._generate_detailed_reasoning(query, answer, supporting_clauses, confidence)
                evidence_points = self._format_detailed_evidence(supporting_clauses, context_sections)
                references = self._format_references(supporting_clauses, context_sections)
                
                explanation = self.explanation_templates["detailed"].format(
                    query=query,
                    answer=answer,
                    detailed_reasoning=detailed_reasoning,
                    evidence_points=evidence_points,
                    references=references
                )
                
                return explanation
                
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            # Fallback to simple explanation
            return f"Based on the document analysis, {llm_response.get('answer', '')}"
    
    def _format_sources(self, supporting_clauses: List[Dict[str, Any]]) -> str:
        """Format sources for explanation"""
        if not supporting_clauses:
            return "the document contents"
            
        if len(supporting_clauses) == 1:
            return f"the section about '{supporting_clauses[0].get('title', 'relevant content')}'"
            
        titles = [clause.get('title', 'relevant section') for clause in supporting_clauses]
        return f"sections including {', '.join(titles[:-1])} and {titles[-1]}"
    
    def _generate_reasoning(
        self, 
        query: str, 
        answer: str, 
        supporting_clauses: List[Dict[str, Any]],
        confidence: float
    ) -> str:
        """Generate reasoning explanation"""
        # Simple reasoning based on confidence
        if confidence > 0.9:
            return "The document explicitly addresses this question."
        elif confidence > 0.7:
            return "This is clearly supported by the document content."
        elif confidence > 0.5:
            return "This interpretation is reasonably supported by the document."
        else:
            return "This is the best interpretation based on limited relevant content in the document."
    
    def _generate_detailed_reasoning(
        self, 
        query: str, 
        answer: str, 
        supporting_clauses: List[Dict[str, Any]],
        confidence: float
    ) -> str:
        """Generate detailed reasoning explanation"""
        reasoning_parts = []
        
        # Add query analysis
        query_keywords = self._extract_keywords(query)
        reasoning_parts.append(f"The query is asking about {', '.join(query_keywords)}.")
        
        # Add evidence-based reasoning
        if supporting_clauses:
            key_clauses = [c.get("text", "")[:100] + "..." for c in supporting_clauses[:2]]
            reasoning_parts.append("Key evidence found in the document includes:")
            for i, clause in enumerate(key_clauses):
                reasoning_parts.append(f"  {i+1}. {clause}")
                
            # Add interpretation
            reasoning_parts.append("\nInterpreting this evidence:")
            if confidence > 0.8:
                reasoning_parts.append("- The document provides clear and direct information addressing the query")
            elif confidence > 0.6:
                reasoning_parts.append("- The document contains relevant information that supports the answer")
            else:
                reasoning_parts.append("- The document contains partial information that suggests this answer")
        else:
            reasoning_parts.append("Limited direct evidence was found in the document. The answer is based on general interpretation.")
        
        return "\n".join(reasoning_parts)
    
    def _format_detailed_evidence(
        self, 
        supporting_clauses: List[Dict[str, Any]],
        context_sections: List[Dict[str, Any]]
    ) -> str:
        """Format detailed evidence points"""
        if not supporting_clauses:
            return "No direct evidence quoted from document."
            
        evidence = []
        for i, clause in enumerate(supporting_clauses):
            text = clause.get("text", "")
            location = clause.get("location", "")
            evidence.append(f"{i+1}. {text} [{location}]")
            
        return "\n".join(evidence)
    
    def _format_references(
        self, 
        supporting_clauses: List[Dict[str, Any]],
        context_sections: List[Dict[str, Any]]
    ) -> str:
        """Format document references"""
        references = []
        
        # Get unique locations
        locations = {}
        for clause in supporting_clauses:
            location = clause.get("location", "")
            if location and location not in locations:
                locations[location] = True
                references.append(f"- {location}")
                
        # If no specific locations, use section metadata
        if not references:
            for i, section in enumerate(context_sections[:3]):
                meta = section.get("metadata", {})
                if "position" in meta:
                    references.append(f"- Document section {meta['position'] + 1}")
        
        if not references:
            return "No specific references available."
            
        return "\n".join(references)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from text"""
        # Simple keyword extraction based on noun phrases and important terms
        # In a real system this would use NLP techniques
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        # Remove common stop words
        stop_words = {"the", "and", "for", "with", "what", "where", "when", "who", "how", "does", "is", "are"}
        keywords = [word for word in words if word.lower() not in stop_words]
        return keywords[:5]  # Return top 5 keywords
    
    # Domain-specific helper methods
    def _extract_policy_section(
        self, 
        supporting_clauses: List[Dict[str, Any]],
        context_sections: List[Dict[str, Any]]
    ) -> str:
        """Extract policy section name for insurance documents"""
        # Check supporting clauses first
        for clause in supporting_clauses:
            if "section" in clause:
                return clause["section"]
                
        # Try to extract from context
        for section in context_sections:
            text = section.get("text", "").lower()
            section_types = ["coverage", "exclusion", "limitation", "benefit", "definition"]
            for s_type in section_types:
                if s_type in text:
                    return s_type
        
        return "relevant"
    
    def _extract_key_evidence(self, supporting_clauses: List[Dict[str, Any]]) -> str:
        """Extract key evidence from supporting clauses"""
        if not supporting_clauses:
            return "the document states important conditions"
            
        # Use the highest confidence clause
        best_clause = sorted(supporting_clauses, key=lambda c: c.get("confidence", 0), reverse=True)[0]
        text = best_clause.get("text", "")
        
        # Trim to a reasonable length
        if len(text) > 100:
            return text[:97] + "..."
            
        return text
    
    def _generate_additional_context(
        self,
        context_sections: List[Dict[str, Any]],
        domain: str
    ) -> str:
        """Generate additional context based on domain"""
        if domain == "insurance":
            for section in context_sections:
                # Look for exclusions or limitations
                text = section.get("text", "").lower()
                if "exclusion" in text or "not cover" in text:
                    return "Note that certain exclusions may apply."
                if "limit" in text or "maximum" in text:
                    return "Coverage limits and conditions may apply."
            return "Review the full policy for all terms and conditions."
            
        elif domain == "legal":
            return "This interpretation may vary based on jurisdiction and specific circumstances."
            
        return ""
    
    def _extract_clause_reference(self, supporting_clauses: List[Dict[str, Any]]) -> str:
        """Extract clause reference for legal documents"""
        if not supporting_clauses:
            return "the document"
            
        for clause in supporting_clauses:
            title = clause.get("title", "")
            section = clause.get("section", "")
            location = clause.get("location", "")
            
            if section:
                return f"Section {section}"
            if title:
                return title
            if location:
                return location
                
        return "relevant clauses in the document"
    
    def _generate_legal_implications(self, answer: str, domain: str) -> str:
        """Generate statement about legal implications"""
        if domain != "legal":
            return ""
            
        # Simple template based on keywords in the answer
        if any(word in answer.lower() for word in ["require", "must", "obligation", "shall"]):
            return "This creates a binding obligation under the terms specified."
        elif any(word in answer.lower() for word in ["permitted", "may", "option", "discretion"]):
            return "This provides conditional permissions rather than obligations."
        elif any(word in answer.lower() for word in ["prohibit", "restrict", "not allowed", "cannot"]):
            return "This represents a limitation or restriction within the agreement."
            
        return "Consider consulting legal counsel for specific advice regarding this provision."