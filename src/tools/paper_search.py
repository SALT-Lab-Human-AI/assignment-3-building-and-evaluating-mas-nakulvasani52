"""
Paper Search Tool
Integrates with Semantic Scholar API for academic paper search.

This tool provides academic paper search functionality using the
Semantic Scholar API, which offers free access to a large corpus
of academic papers.
"""

from typing import List, Dict, Any, Optional
import os
import logging
import asyncio


class PaperSearchTool:
    """
    Tool for searching academic papers via Semantic Scholar API.
    
    Semantic Scholar provides free access to academic papers with
    rich metadata including citations, abstracts, and author information.
    API key is optional but recommended for higher rate limits.
    """

    def __init__(self, max_results: int = 10, provider: str = "semantic_scholar"):
        """
        Initialize paper search tool.

        Args:
            max_results: Maximum number of papers to return
            provider: Search provider ("semantic_scholar" or "serpapi")
        """
        self.max_results = max_results
        self.provider = provider
        self.logger = logging.getLogger("tools.paper_search")

        # API keys
        self.semantic_scholar_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.serpapi_key = os.getenv("SERPAPI_API_KEY")
        
        if not self.semantic_scholar_key and provider == "semantic_scholar":
            self.logger.info("No Semantic Scholar API key found. Using anonymous access (lower rate limits)")
            
        if not self.serpapi_key and provider == "serpapi":
            self.logger.warning("No SerpAPI key found. Search will fail.")

    async def search(
        self,
        query: str,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        min_citations: int = 0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for academic papers.

        Args:
            query: Search query
            year_from: Filter papers from this year onwards
            year_to: Filter papers up to this year
            min_citations: Minimum citation count
            **kwargs: Additional search parameters
                - fields: List of fields to retrieve

        Returns:
            List of papers with metadata
        """
        self.logger.info(f"Searching papers with {self.provider}: {query}")

        if self.provider == "serpapi":
            return await self._search_serpapi(query, year_from, year_to, min_citations)
        else:
            return await self._search_semantic_scholar(query, year_from, year_to, min_citations, **kwargs)

    async def _search_semantic_scholar(
        self,
        query: str,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        min_citations: int = 0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search using Semantic Scholar API."""
        try:
            from semanticscholar import SemanticScholar
            
            # Initialize Semantic Scholar client
            sch = SemanticScholar(api_key=self.semantic_scholar_key)
            
            # Define fields to retrieve
            fields = kwargs.get("fields", [
                "paperId", "title", "authors", "year", "abstract",
                "citationCount", "url", "venue", "openAccessPdf"
            ])
            
            # Perform search
            results = sch.search_paper(
                query, 
                limit=self.max_results,
                fields=fields
            )
            
            # Parse and filter results
            papers = self._parse_results(results, year_from, year_to, min_citations)
            
            self.logger.info(f"Found {len(papers)} papers")
            return papers
            
        except ImportError:
            self.logger.error("semanticscholar library not installed. Run: pip install semanticscholar")
            return []
        except Exception as e:
            self.logger.error(f"Error searching papers: {e}")
            return []

    async def _search_serpapi(
        self,
        query: str,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        min_citations: int = 0
    ) -> List[Dict[str, Any]]:
        """Search using Google Scholar via SerpAPI."""
        if not self.serpapi_key:
            self.logger.error("SerpAPI key not found")
            return []
            
        try:
            import aiohttp
            
            url = "https://serpapi.com/search.json"
            params = {
                "engine": "google_scholar",
                "q": query,
                "api_key": self.serpapi_key,
                "num": self.max_results
            }
            
            if year_from:
                params["as_ylo"] = str(year_from)
            if year_to:
                params["as_yhi"] = str(year_to)
                
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_serpapi_results(data, min_citations)
                    else:
                        self.logger.error(f"SerpAPI error: {response.status}")
                        return []
                        
        except ImportError:
            self.logger.error("aiohttp not installed")
            return []
        except Exception as e:
            self.logger.error(f"Error searching SerpAPI: {e}")
            return []

    def _parse_serpapi_results(
        self,
        data: Dict[str, Any],
        min_citations: int
    ) -> List[Dict[str, Any]]:
        """Parse SerpAPI Google Scholar results."""
        papers = []
        
        for result in data.get("organic_results", []):
            # Extract citation count
            citation_info = result.get("inline_links", {}).get("cited_by", {})
            citation_count = citation_info.get("total", 0)
            
            if citation_count < min_citations:
                continue
                
            # Extract authors and year from publication_info
            pub_info = result.get("publication_info", {})
            summary = pub_info.get("summary", "")
            
            # Simple parsing of summary "Author1, Author2 - Venue, Year - publisher"
            year = None
            import re
            year_match = re.search(r'\b(19|20)\d{2}\b', summary)
            if year_match:
                year = int(year_match.group(0))
                
            # Extract authors (rough approximation)
            authors_str = summary.split("-")[0].strip()
            authors = [{"name": a.strip()} for a in authors_str.split(",")]
            
            papers.append({
                "paper_id": result.get("result_id"),
                "title": result.get("title", "Unknown"),
                "authors": authors,
                "year": year,
                "abstract": result.get("snippet", ""),
                "citation_count": citation_count,
                "url": result.get("link", ""),
                "venue": "",  # Hard to extract reliably
                "pdf_url": result.get("resources", [{}])[0].get("link") if result.get("resources") else None
            })
            
        return papers

    async def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific paper.

        Args:
            paper_id: Semantic Scholar paper ID

        Returns:
            Detailed paper information
        """
        try:
            from semanticscholar import SemanticScholar
            
            sch = SemanticScholar(api_key=self.api_key)
            paper = sch.get_paper(paper_id)
            
            return {
                "paper_id": paper.paperId,
                "title": paper.title,
                "authors": [{"name": a.name} for a in paper.authors] if paper.authors else [],
                "year": paper.year,
                "abstract": paper.abstract,
                "citation_count": paper.citationCount,
                "url": paper.url,
                "venue": paper.venue,
                "pdf_url": paper.openAccessPdf.get("url") if paper.openAccessPdf else None,
            }
        except Exception as e:
            self.logger.error(f"Error getting paper details: {e}")
            return {}

    async def get_citations(self, paper_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get papers that cite this paper.

        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum number of citations to retrieve

        Returns:
            List of citing papers
        """
        try:
            from semanticscholar import SemanticScholar
            
            sch = SemanticScholar(api_key=self.api_key)
            paper = sch.get_paper(paper_id)
            citations = paper.citations[:limit] if paper.citations else []
            
            return [
                {
                    "paper_id": c.paperId,
                    "title": c.title,
                    "year": c.year,
                }
                for c in citations
            ]
        except Exception as e:
            self.logger.error(f"Error getting citations: {e}")
            return []

    async def get_references(self, paper_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get papers referenced by this paper.

        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum number of references to retrieve

        Returns:
            List of referenced papers
        """
        try:
            from semanticscholar import SemanticScholar
            
            sch = SemanticScholar(api_key=self.api_key)
            paper = sch.get_paper(paper_id)
            references = paper.references[:limit] if paper.references else []
            
            return [
                {
                    "paper_id": r.paperId,
                    "title": r.title,
                    "year": r.year,
                }
                for r in references
            ]
        except Exception as e:
            self.logger.error(f"Error getting references: {e}")
            return []

    def _parse_results(
        self,
        results: Any,
        year_from: Optional[int],
        year_to: Optional[int],
        min_citations: int
    ) -> List[Dict[str, Any]]:
        """
        Parse and filter search results from Semantic Scholar.
        
        Args:
            results: Raw results from Semantic Scholar API
            year_from: Minimum year filter
            year_to: Maximum year filter
            min_citations: Minimum citation count filter
            
        Returns:
            Filtered and formatted list of papers
        """
        papers = []
        
        for paper in results:
            # Skip papers without basic metadata
            if not paper or not hasattr(paper, 'title'):
                continue
                
            paper_dict = {
                "paper_id": paper.paperId if hasattr(paper, 'paperId') else None,
                "title": paper.title if hasattr(paper, 'title') else "Unknown",
                "authors": [{"name": a.name} for a in paper.authors] if hasattr(paper, 'authors') and paper.authors else [],
                "year": paper.year if hasattr(paper, 'year') else None,
                "abstract": paper.abstract if hasattr(paper, 'abstract') else "",
                "citation_count": paper.citationCount if hasattr(paper, 'citationCount') else 0,
                "url": paper.url if hasattr(paper, 'url') else "",
                "venue": paper.venue if hasattr(paper, 'venue') else "",
                "pdf_url": paper.openAccessPdf.get("url") if hasattr(paper, 'openAccessPdf') and paper.openAccessPdf else None,
            }
            
            papers.append(paper_dict)
        
        # Apply filters
        papers = self._filter_by_year(papers, year_from, year_to)
        papers = self._filter_by_citations(papers, min_citations)
        
        return papers

    def _filter_by_year(
        self,
        papers: List[Dict[str, Any]],
        year_from: Optional[int],
        year_to: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Filter papers by publication year."""
        filtered = papers
        if year_from:
            filtered = [p for p in filtered if p.get("year") and p.get("year") >= year_from]
        if year_to:
            filtered = [p for p in filtered if p.get("year") and p.get("year") <= year_to]
        return filtered

    def _filter_by_citations(
        self,
        papers: List[Dict[str, Any]],
        min_citations: int
    ) -> List[Dict[str, Any]]:
        """Filter papers by citation count."""
        return [p for p in papers if p.get("citation_count", 0) >= min_citations]


# Synchronous wrapper for use with AutoGen tools
def paper_search(query: str, max_results: int = 10, year_from: Optional[int] = None) -> str:
    """
    Synchronous wrapper for paper search (for AutoGen tool integration).
    
    Args:
        query: Search query
        max_results: Maximum results to return
        year_from: Only return papers from this year onwards
        
    Returns:
        Formatted string with paper results
    """
    tool = PaperSearchTool(max_results=max_results)
    results = asyncio.run(tool.search(query, year_from=year_from))
    
    if not results:
        return "No academic papers found."
    
    # Format results as readable text
    output = f"Found {len(results)} academic papers for '{query}':\n\n"
    
    for i, paper in enumerate(results, 1):
        authors = ", ".join([a["name"] for a in paper["authors"][:3]])
        if len(paper["authors"]) > 3:
            authors += " et al."
            
        output += f"{i}. {paper['title']}\n"
        output += f"   Authors: {authors}\n"
        output += f"   Year: {paper['year']} | Citations: {paper['citation_count']}"
        if paper.get('venue'):
            output += f" | Venue: {paper['venue']}"
        output += "\n"
        
        if paper.get('abstract'):
            abstract = paper['abstract'][:200] + "..." if len(paper['abstract']) > 200 else paper['abstract']
            output += f"   Abstract: {abstract}\n"
            
        output += f"   URL: {paper['url']}\n"
        output += "\n"
    
    return output
