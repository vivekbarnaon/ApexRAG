"""
LangChain Integration Module
Provides enhanced document processing, query routing, and tool use capabilities.
"""

import re
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class EnhancedDocumentProcessor:
    """
    Enhanced document processor using LangChain for better chunking and processing.
    """
    
    def __init__(self):
        self.text_splitter = None
        self._initialize_splitters()
    
    def _initialize_splitters(self):
        """Initialize different text splitters for various document types."""
        # Insurance-specific separators for better boundary detection
        insurance_separators = [
            # Document structure markers (highest priority)
            "\n\nARTICLE", "\n\nSECTION", "\n\nCLAUSE", "\n\nPART",
            "\n\nCOVERAGE", "\n\nBENEFIT", "\n\nEXCLUSION", "\n\nLIMIT",
            "\n\nArticle", "\n\nSection", "\n\nClause", "\n\nPart",
            "\n\nCoverage", "\n\nBenefit", "\n\nExclusion", "\n\nLimit",
            
            # Numbered sections (common in legal documents)
            r"\n\d+\.\d+", r"\n\d+\.", r"\n[A-Z]\.", r"\n[a-z]\.", r"\n[ivxIVX]+\.",
            
            # Definitions and key terms
            "\n\nDefinitions", "\n\nTerms", "\n\nGlossary",
            
            # General document separators
            "\n\n", "\n", ". ", "; ", ", ", " "
        ]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=insurance_separators,
            length_function=len,
            is_separator_regex=False
        )
    
    def process_documents(self, text_by_page: List[str], chunk_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Process documents using LangChain with enhanced metadata extraction and adaptive chunking.

        Args:
            text_by_page: List of text content by page
            chunk_config: Optional adaptive chunk configuration

        Returns:
            List of processed document chunks with metadata
        """
        try:
            # Use adaptive chunking if configuration is provided
            if chunk_config:
                # Insurance-specific separators for better boundary detection
                insurance_separators = [
                    # Document structure markers (highest priority)
                    "\n\nARTICLE", "\n\nSECTION", "\n\nCLAUSE", "\n\nPART",
                    "\n\nCOVERAGE", "\n\nBENEFIT", "\n\nEXCLUSION", "\n\nLIMIT",
                    "\n\nArticle", "\n\nSection", "\n\nClause", "\n\nPart",
                    "\n\nCoverage", "\n\nBenefit", "\n\nExclusion", "\n\nLimit",

                    # Numbered sections (common in legal documents)
                    r"\n\d+\.\d+", r"\n\d+\.", r"\n[A-Z]\.", r"\n[a-z]\.", r"\n[ivxIVX]+\.",

                    # Definitions and key terms
                    "\n\nDefinitions", "\n\nTerms", "\n\nGlossary",

                    # General document separators
                    "\n\n", "\n", ". ", "; ", ", ", " "
                ]

                # Create adaptive text splitter
                adaptive_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_config["chunk_size"],
                    chunk_overlap=chunk_config["overlap"],
                    separators=insurance_separators,
                    length_function=len,
                    is_separator_regex=False
                )
                splitter_to_use = adaptive_splitter
            else:
                splitter_to_use = self.text_splitter

            # Create LangChain documents
            documents = []
            for page_num, page_text in enumerate(text_by_page, 1):
                if not page_text.strip():
                    continue

                doc = Document(
                    page_content=page_text,
                    metadata={
                        "page": page_num,
                        "source": f"page_{page_num}",
                        "document_type": self._detect_document_type(page_text),
                        "chunk_config": chunk_config
                    }
                )
                documents.append(doc)

            # Split documents into chunks using adaptive splitter
            chunks = splitter_to_use.split_documents(documents)
            
            # Convert to enhanced format with metadata
            enhanced_chunks = []
            for i, chunk in enumerate(chunks):
                enhanced_chunk = self._enhance_chunk_metadata(chunk, i)
                enhanced_chunks.append(enhanced_chunk)
            
            return enhanced_chunks
            
        except Exception as e:
            # Fallback to original processing
            return self._fallback_processing(text_by_page)
    
    def _detect_document_type(self, text: str) -> str:
        """
        Detect the type of document content.
        
        Args:
            text: Document text
            
        Returns:
            Document type string
        """
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['policy', 'coverage', 'premium', 'deductible']):
            return "insurance_policy"
        elif any(term in text_lower for term in ['claim', 'benefit', 'reimbursement']):
            return "claims_document"
        elif any(term in text_lower for term in ['terms', 'conditions', 'agreement']):
            return "terms_conditions"
        elif any(term in text_lower for term in ['definition', 'means', 'glossary']):
            return "definitions"
        else:
            return "general"
    
    def _enhance_chunk_metadata(self, chunk: Document, chunk_index: int) -> Dict[str, Any]:
        """
        Enhance chunk with additional metadata.
        
        Args:
            chunk: LangChain document chunk
            chunk_index: Index of the chunk
            
        Returns:
            Enhanced chunk dictionary
        """
        content = chunk.page_content
        metadata = chunk.metadata.copy()
        
        # Extract content characteristics
        content_analysis = self._analyze_content(content)
        
        # Build enhanced chunk
        enhanced_chunk = {
            "text": f"Page {metadata.get('page', 1)}, Section {chunk_index + 1}: {content[:100]}... {content}",
            "raw_text": content,
            "page": metadata.get("page", 1),
            "section": chunk_index + 1,
            "document_type": metadata.get("document_type", "general"),
            "metadata": content_analysis["categories"],
            "content_features": content_analysis["features"],
            "chunk_index": chunk_index,
            "chunk_type": "enhanced",
            "chunk_config": metadata.get("chunk_config"),
            "actual_size": len(content)
        }
        
        return enhanced_chunk
    
    def _analyze_content(self, content: str) -> Dict[str, Any]:
        """
        Analyze content for categories and features.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Analysis results
        """
        content_lower = content.lower()
        categories = []
        features = {}
        
        # Check for content categories
        if 'means' in content_lower and bool(re.search(r'\b\w+\s+means\b', content_lower)):
            categories.append("definition")
        
        if bool(re.search(r'\bexclusion|\bexcluded|\bnot covered|\bnot eligible', content_lower)):
            categories.append("exclusion")
        
        if bool(re.search(r'\bcoverage|\bcovered|\beligible|\bincluded', content_lower)):
            categories.append("coverage")
        
        if bool(re.search(r'\blimit|\bcap|\bmaximum|\bupto|\bup to', content_lower)):
            categories.append("limit")
        
        if bool(re.search(r'\bcondition|\bprovided that|\bsubject to|\bif and only if', content_lower)):
            categories.append("condition")
        
        # Extract features
        features["has_numbers"] = bool(re.search(r'\$|%|\d+', content))
        features["has_dates"] = bool(re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', content))
        features["has_percentages"] = bool(re.search(r'\d+(?:\.\d+)?%', content))
        features["has_currency"] = bool(re.search(r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?', content))
        features["word_count"] = len(content.split())
        features["sentence_count"] = len(re.findall(r'[.!?]+', content))
        
        return {
            "categories": categories,
            "features": features
        }
    
    def _fallback_processing(self, text_by_page: List[str]) -> List[Dict[str, Any]]:
        """
        Fallback processing method if LangChain fails.
        
        Args:
            text_by_page: List of text content by page
            
        Returns:
            List of processed chunks
        """
        chunks = []
        for page_num, page_text in enumerate(text_by_page, 1):
            if not page_text.strip():
                continue
            
            # Simple chunking
            chunk_size = 800
            overlap = 100
            
            for i in range(0, len(page_text), chunk_size - overlap):
                chunk_text = page_text[i:i + chunk_size]
                if len(chunk_text.strip()) < 50:
                    continue
                
                chunk = {
                    "text": f"Page {page_num}, Section {i//chunk_size + 1}: {chunk_text[:100]}... {chunk_text}",
                    "raw_text": chunk_text,
                    "page": page_num,
                    "section": i//chunk_size + 1,
                    "metadata": [],
                    "chunk_index": len(chunks)
                }
                chunks.append(chunk)
        
        return chunks


class QueryRouter:
    """
    Routes queries to appropriate processing pipelines based on content type.
    """
    
    def __init__(self):
        self.route_patterns = self._initialize_route_patterns()
    
    def _initialize_route_patterns(self) -> Dict[str, List[str]]:
        """Initialize routing patterns for different query types."""
        return {
            "definition": [
                r"what is", r"define", r"meaning of", r"definition of",
                r"what does.*mean", r"explain.*term"
            ],
            "coverage": [
                r"covered", r"coverage", r"eligible", r"included",
                r"does.*cover", r"is.*covered"
            ],
            "exclusion": [
                r"excluded", r"not covered", r"exclusion", r"denied",
                r"does not cover", r"not eligible"
            ],
            "process": [
                r"how to", r"process", r"procedure", r"steps",
                r"submit", r"file.*claim"
            ],
            "limits": [
                r"limit", r"maximum", r"minimum", r"cap", r"ceiling",
                r"how much", r"amount"
            ],
            "documents": [
                r"document", r"proof", r"evidence", r"form",
                r"paperwork", r"required.*document"
            ],
            "financial": [
                r"cost", r"price", r"fee", r"premium", r"deductible",
                r"copay", r"coinsurance", r"\$", r"dollar"
            ]
        }
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Route query to appropriate processing pipeline.
        
        Args:
            query: User query
            
        Returns:
            Routing information
        """
        query_lower = query.lower()
        detected_types = []
        confidence_scores = {}
        
        # Check each route pattern
        for route_type, patterns in self.route_patterns.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    matches += 1
            
            if matches > 0:
                confidence = matches / len(patterns)
                detected_types.append(route_type)
                confidence_scores[route_type] = confidence
        
        # Determine primary route
        if detected_types:
            primary_route = max(confidence_scores.items(), key=lambda x: x[1])[0]
        else:
            primary_route = "general"
        
        return {
            "primary_route": primary_route,
            "detected_types": detected_types,
            "confidence_scores": confidence_scores,
            "processing_hints": self._get_processing_hints(primary_route)
        }
    
    def _get_processing_hints(self, route_type: str) -> Dict[str, Any]:
        """
        Get processing hints for the route type.
        
        Args:
            route_type: Type of route
            
        Returns:
            Processing hints
        """
        hints = {
            "definition": {
                "prioritize_chunks": ["definition"],
                "search_terms": ["means", "defined as", "refers to"],
                "response_style": "explanatory"
            },
            "coverage": {
                "prioritize_chunks": ["coverage"],
                "search_terms": ["covered", "eligible", "included"],
                "response_style": "confirmatory"
            },
            "exclusion": {
                "prioritize_chunks": ["exclusion"],
                "search_terms": ["excluded", "not covered", "denied"],
                "response_style": "restrictive"
            },
            "process": {
                "prioritize_chunks": ["condition"],
                "search_terms": ["steps", "procedure", "process"],
                "response_style": "instructional"
            },
            "limits": {
                "prioritize_chunks": ["limit"],
                "search_terms": ["maximum", "limit", "cap"],
                "response_style": "specific"
            },
            "financial": {
                "prioritize_chunks": ["limit", "coverage"],
                "search_terms": ["$", "cost", "fee", "premium"],
                "response_style": "numerical"
            }
        }
        
        return hints.get(route_type, {
            "prioritize_chunks": [],
            "search_terms": [],
            "response_style": "general"
        })
