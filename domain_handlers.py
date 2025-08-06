from typing import Dict, List, Any, Optional, Protocol
import logging
import re
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DomainHandler(ABC):
    """Base class for domain-specific document handling"""
    
    def __init__(self):
        self.domain_name = "base"
        self.entity_types = []
        self.field_extractors = {}
    
    @abstractmethod
    async def process_document(self, document_id: str, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process document content with domain-specific logic"""
        pass
        
    @abstractmethod
    async def process_query_context(self, query: str, context_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process query context with domain-specific enhancements"""
        pass
        
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract domain-specific entities from text"""
        entities = {}
        for entity_type in self.entity_types:
            extractor = self.field_extractors.get(entity_type)
            if extractor and callable(extractor):
                entities[entity_type] = extractor(text)
        return entities


class InsuranceHandler(DomainHandler):
    """Handler for insurance documents"""
    
    def __init__(self):
        super().__init__()
        self.domain_name = "insurance"
        self.entity_types = ["policy_number", "premium_amount", "coverage_limit", 
                            "waiting_period", "exclusions", "benefits"]
                            
        # Setup field extractors
        self.field_extractors = {
            "policy_number": self._extract_policy_numbers,
            "premium_amount": self._extract_premium_amounts,
            "coverage_limit": self._extract_coverage_limits,
            "waiting_period": self._extract_waiting_periods,
            "exclusions": self._extract_exclusions,
        }
        
        # Domain-specific patterns
        self.policy_sections = {
            "definitions": ["definition", "means", "refers to", "defined as"],
            "coverage": ["covers", "coverage", "benefit", "sum insured", "limit"],
            "exclusions": ["exclusion", "not covered", "excluded", "does not cover"],
            "conditions": ["condition", "subject to", "provided that"],
            "waiting_period": ["waiting period", "days", "months", "years"],
            "claims": ["claim", "procedure", "documentation", "submission"]
        }
    
    async def process_document(self, document_id: str, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process insurance document with domain-specific logic"""
        try:
            # Extract policy structure
            structure = self._analyze_policy_structure(content)
            
            # Extract key entities
            entities = self.extract_entities(content)
            
            # Process document-specific metadata
            result = {
                "document_id": document_id,
                "domain": "insurance",
                "structure": structure,
                "entities": entities,
                "document_type": self._determine_document_type(content, metadata)
            }
            
            logger.info(f"Processed insurance document {document_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing insurance document: {str(e)}")
            return {
                "document_id": document_id,
                "domain": "insurance",
                "error": str(e)
            }
    
    async def process_query_context(self, query: str, context_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process insurance query context with domain-specific enhancements"""
        enhanced_sections = []
        
        # Determine query intent
        intent = self._determine_query_intent(query)
        
        for section in context_sections:
            # Clone the section
            enhanced = dict(section)
            
            # Add section classification
            enhanced["section_type"] = self._classify_section(section["text"])
            
            # Extract relevant entities based on intent
            if intent == "coverage":
                enhanced["extracted_coverage"] = self._extract_coverage_details(section["text"])
            elif intent == "exclusion":
                enhanced["extracted_exclusions"] = self._extract_exclusions(section["text"])
            elif intent == "waiting_period":
                enhanced["extracted_periods"] = self._extract_waiting_periods(section["text"])
            elif intent == "benefit":
                enhanced["extracted_benefits"] = self._extract_benefits(section["text"])
            
            enhanced_sections.append(enhanced)
        
        # Sort sections by relevance to query intent
        enhanced_sections.sort(
            key=lambda s: self._calculate_section_relevance(s, intent),
            reverse=True
        )
        
        return enhanced_sections
    
    def _analyze_policy_structure(self, content: str) -> Dict[str, Any]:
        """Analyze the structure of an insurance policy document"""
        structure = {}
        
        # Identify major sections
        for section_type, keywords in self.policy_sections.items():
            pattern = "|".join(keywords)
            matches = re.finditer(pattern, content.lower())
            positions = [m.start() for m in matches]
            
            if positions:
                structure[section_type] = {
                    "found": True,
                    "count": len(positions),
                    "first_position": min(positions) if positions else -1
                }
            else:
                structure[section_type] = {"found": False}
        
        # Check for structure indicators
        structure["has_toc"] = "table of contents" in content.lower() or "index" in content.lower()
        structure["has_numbered_sections"] = bool(re.search(r'\n\d+\.\s+[A-Z]', content))
        
        return structure
    
    def _determine_document_type(self, content: str, metadata: Dict[str, Any]) -> str:
        """Determine insurance document type"""
        content_lower = content.lower()
        
        if "health" in content_lower and "insurance" in content_lower:
            return "health_insurance_policy"
        elif "life" in content_lower and "insurance" in content_lower:
            return "life_insurance_policy"
        elif "motor" in content_lower or "vehicle" in content_lower:
            return "motor_insurance_policy"
        elif "home" in content_lower or "property" in content_lower:
            return "property_insurance_policy"
        elif "travel" in content_lower:
            return "travel_insurance_policy"
        elif "marine" in content_lower:
            return "marine_insurance_policy"
        elif "claim" in content_lower and "form" in content_lower:
            return "claim_form"
        else:
            return "generic_insurance_document"
    
    def _determine_query_intent(self, query: str) -> str:
        """Determine the intent of an insurance query"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["cover", "coverage", "covered", "benefit"]):
            return "coverage"
        elif any(term in query_lower for term in ["exclude", "exclusion", "not covered"]):
            return "exclusion"
        elif any(term in query_lower for term in ["waiting period", "wait", "days", "months"]):
            return "waiting_period"
        elif any(term in query_lower for term in ["premium", "pay", "cost", "price"]):
            return "premium"
        elif any(term in query_lower for term in ["claim", "process", "procedure"]):
            return "claims"
        elif any(term in query_lower for term in ["define", "definition", "mean"]):
            return "definition"
        else:
            return "general"
    
    def _classify_section(self, text: str) -> str:
        """Classify insurance document section by type"""
        text_lower = text.lower()
        
        for section_type, keywords in self.policy_sections.items():
            if any(keyword in text_lower for keyword in keywords):
                return section_type
                
        return "general"
    
    def _calculate_section_relevance(self, section: Dict[str, Any], intent: str) -> float:
        """Calculate section relevance score based on query intent"""
        text_lower = section["text"].lower()
        section_type = section.get("section_type", "general")
        
        base_score = 0.5  # Default score
        
        # Type match bonus
        if section_type == intent:
            base_score += 1.0
        
        # Content relevance bonus
        if intent == "coverage":
            if any(term in text_lower for term in ["cover", "coverage", "benefit", "sum insured"]):
                base_score += 0.5
        elif intent == "exclusion":
            if any(term in text_lower for term in ["exclude", "exclusion", "not cover"]):
                base_score += 0.5
        elif intent == "waiting_period":
            if any(term in text_lower for term in ["waiting", "period", "days", "months"]):
                base_score += 0.5
        
        # Number bonus (insurance is often about specific amounts and periods)
        if re.search(r'\d+', text_lower):
            base_score += 0.3
        
        # Structural bonus
        if section.get("metadata", {}).get("contains_section_header", False):
            base_score += 0.2
            
        return base_score
    
    # Entity extraction methods
    def _extract_policy_numbers(self, text: str) -> List[str]:
        """Extract policy numbers from text"""
        patterns = [r'[A-Z]{2,}-\d{6,}', r'POLICY\s*(?:#|NO|NUMBER)[:\s]*([A-Z0-9-]+)']
        results = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            results.extend(matches)
        return list(set(results))  # Deduplicate
    
    def _extract_premium_amounts(self, text: str) -> List[str]:
        """Extract premium amounts from text"""
        patterns = [r'premium[:\s]*(?:Rs\.?|₹|INR)?[\s]*(\d+(?:,\d+)*(?:\.\d{1,2})?)']
        results = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            results.extend(matches)
        return list(set(results))
    
    def _extract_coverage_limits(self, text: str) -> List[str]:
        """Extract coverage limits from text"""
        patterns = [r'sum\s+insured[:\s]*(?:Rs\.?|₹|INR)?[\s]*(\d+(?:,\d+)*(?:\.\d{1,2})?)']
        results = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            results.extend(matches)
        return list(set(results))
    
    def _extract_waiting_periods(self, text: str) -> List[str]:
        """Extract waiting periods from text"""
        periods = []
        patterns = [
            r'waiting\s+period\s+of\s+(\d+)\s*(?:days?|months?|years?)',
            r'(\d+)\s*(?:days?|months?|years?)\s+waiting\s+period'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get surrounding context
                context = re.search(f"{match}[^.]*\.", text, re.IGNORECASE)
                if context:
                    periods.append(context.group(0))
                else:
                    periods.append(f"{match} months/days")
        return periods
    
    def _extract_exclusions(self, text: str) -> List[str]:
        """Extract exclusions from text"""
        exclusions = []
        # Look for "not covered" or "excluded" phrases
        not_covered = re.findall(r'(?:is|are)\s+not\s+covered[:\s]*(.*?)(?:\.|;|\n)', text, re.IGNORECASE)
        exclusions.extend(not_covered)
        # Look for exclusions section with bullet points
        exclusion_section = re.search(r'exclusions[:\s]*(.*?)(?:inclusions|coverage|conditions|$)', 
                                   text, re.IGNORECASE | re.DOTALL)
        if exclusion_section:
            section_text = exclusion_section.group(1)
            bullets = re.findall(r'(?:•|-|\*|\d+\.)\s*(.*?)(?:\n|$)', section_text)
            exclusions.extend(bullets)
        return [e.strip() for e in exclusions if len(e.strip()) > 3]
    
    def _extract_coverage_details(self, text: str) -> Dict[str, Any]:
        """Extract detailed coverage information"""
        coverage = {}
        # Extract coverage amounts
        amounts = re.findall(r'(?:coverage|sum insured|limit)[:\s]*(?:Rs\.?|₹|INR)?[\s]*(\d+(?:,\d+)*(?:\.\d{1,2})?)', 
                           text, re.IGNORECASE)
        if amounts:
            coverage["amounts"] = amounts
        # Extract coverage conditions
        conditions = re.findall(r'(?:subject to|provided that|conditional upon)[:\s]*(.*?)(?:\.|;|\n)', 
                              text, re.IGNORECASE)
        if conditions:
            coverage["conditions"] = [c.strip() for c in conditions]
        return coverage
    
    def _extract_benefits(self, text: str) -> List[str]:
        """Extract benefits from text"""
        benefits = []
        # Look for bullet points after "benefits" heading
        benefit_section = re.search(r'benefits[:\s]*(.*?)(?:exclusions|coverage|conditions|$)', 
                                  text, re.IGNORECASE | re.DOTALL)
        if benefit_section:
            section_text = benefit_section.group(1)
            bullets = re.findall(r'(?:•|-|\*|\d+\.)\s*(.*?)(?:\n|$)', section_text)
            benefits.extend(bullets)
        return [b.strip() for b in benefits if len(b.strip()) > 3]


class LegalHandler(DomainHandler):
    """Handler for legal documents"""
    
    def __init__(self):
        super().__init__()
        self.domain_name = "legal"
        self.entity_types = ["parties", "effective_date", "termination_date", "jurisdiction", "obligations"]
        
        # Setup field extractors
        self.field_extractors = {
            "parties": self._extract_parties,
            "effective_date": self._extract_dates,
            "jurisdiction": self._extract_jurisdiction,
        }
    
    async def process_document(self, document_id: str, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process legal document with domain-specific logic"""
        try:
            # Extract document structure
            structure = self._analyze_legal_structure(content)
            
            # Extract key entities
            entities = self.extract_entities(content)
            
            # Process document-specific metadata
            result = {
                "document_id": document_id,
                "domain": "legal",
                "structure": structure,
                "entities": entities,
                "document_type": self._determine_document_type(content, metadata)
            }
            
            logger.info(f"Processed legal document {document_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing legal document: {str(e)}")
            return {
                "document_id": document_id,
                "domain": "legal",
                "error": str(e)
            }
    
    async def process_query_context(self, query: str, context_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process legal query context with domain-specific enhancements"""
        enhanced_sections = []
        
        # Determine query intent
        intent = self._determine_legal_query_intent(query)
        
        for section in context_sections:
            # Clone the section
            enhanced = dict(section)
            
            # Add section classification
            enhanced["section_type"] = self._classify_legal_section(section["text"])
            
            # Extract relevant entities based on intent
            if intent == "obligations":
                enhanced["extracted_obligations"] = self._extract_obligations(section["text"])
            elif intent == "termination":
                enhanced["extracted_termination"] = self._extract_termination_conditions(section["text"])
            elif intent == "jurisdiction":
                enhanced["extracted_jurisdiction"] = self._extract_jurisdiction(section["text"])
            
            enhanced_sections.append(enhanced)
        
        # Sort sections by relevance to query intent
        enhanced_sections.sort(
            key=lambda s: self._calculate_legal_section_relevance(s, intent),
            reverse=True
        )
        
        return enhanced_sections
    
    def _analyze_legal_structure(self, content: str) -> Dict[str, Any]:
        """Analyze the structure of a legal document"""
        structure = {
            "sections": 0,
            "clauses": 0,
            "has_definitions": False,
            "has_termination": False,
            "has_governing_law": False,
            "has_signatures": False
        }
        
        # Count sections and clauses
        section_matches = re.findall(r'\n(?:Section|SECTION)\s+\d+', content)
        clause_matches = re.findall(r'\n(?:Clause|CLAUSE)\s+\d+', content)
        
        structure["sections"] = len(section_matches)
        structure["clauses"] = len(clause_matches)
        
        # Check for common document parts
        structure["has_definitions"] = bool(re.search(r'\bDefinitions\b|\bTerms Defined\b', content, re.IGNORECASE))
        structure["has_termination"] = bool(re.search(r'\bTermination\b|\bCancellation\b', content, re.IGNORECASE))
        structure["has_governing_law"] = bool(re.search(r'\bGoverning Law\b|\bJurisdiction\b', content, re.IGNORECASE))
        structure["has_signatures"] = bool(re.search(r'\bSignature\b|\bExecuted by\b|\bIN WITNESS WHEREOF\b', content, re.IGNORECASE))
        
        return structure
    
    def _determine_document_type(self, content: str, metadata: Dict[str, Any]) -> str:
        """Determine legal document type"""
        content_