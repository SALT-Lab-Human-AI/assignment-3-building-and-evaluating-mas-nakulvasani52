"""
LangGraph Agents
Implements 4 specialized agents for literature review using LangGraph.

Agents:
1. Planner - Analyzes research topic and creates search strategy
2. Researcher - Searches for papers and gathers information
3. Analyzer - Identifies patterns, methodologies, and state-of-the-art
4. Writer - Synthesizes findings into comprehensive literature review
"""

from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
import os
import logging
from datetime import datetime

logger = logging.getLogger("agents.langgraph")


class BaseLangGraphAgent:
    """Base class for LangGraph agents."""
    
    def __init__(self, name: str, role: str, model_config: Dict[str, Any], system_prompt: str = ""):
        """
        Initialize base agent.
        
        Args:
            name: Agent name
            role: Agent role/description
            model_config: Model configuration (provider, name, temperature, etc.)
            system_prompt: Custom system prompt (optional)
        """
        self.name = name
        self.role = role
        self.model_config = model_config
        self.system_prompt = system_prompt
        self.logger = logging.getLogger(f"agents.{name}")
        
        # Initialize LLM
        self.llm = self._init_llm()
        
    def _init_llm(self):
        """Initialize the LLM based on provider."""
        provider = self.model_config.get("provider", "groq")
        model_name = self.model_config.get("name", "llama-3.1-70b-versatile")
        temperature = self.model_config.get("temperature", 0.7)
        max_tokens = self.model_config.get("max_tokens", 4096)
        
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                self.logger.error("GROQ_API_KEY not found in environment")
                raise ValueError("GROQ_API_KEY is required")
            
            return ChatGroq(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def invoke(self, messages: List) -> str:
        """
        Invoke the agent with messages.
        
        Args:
            messages: List of messages
            
        Returns:
            Agent's response as string
        """
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            self.logger.error(f"Error invoking agent: {e}")
            return f"Error: {str(e)}"


class PlannerAgent(BaseLangGraphAgent):
    """
    Planner Agent - Analyzes research topic and creates search strategy.
    
    Responsibilities:
    - Parse user's research topic and project description
    - Identify key concepts and search terms
    - Create structured search strategy
    - Determine scope and focus areas
    """
    
    def __init__(self, config: Dict[str, Any]):
        system_prompt = config.get("system_prompt", "") or self._default_prompt()
        super().__init__(
            name="Planner",
            role=config.get("role", "Research Planner"),
            model_config=config.get("model_config", {}),
            system_prompt=system_prompt
        )
    
    def _default_prompt(self) -> str:
        return """You are an expert research planner specializing in literature reviews.

Your task is to analyze a research topic and project description, then create a comprehensive search strategy.

Your output should include:
1. Key concepts and search terms (5-10 terms)
2. Recommended search queries (3-5 queries)
3. Suggested paper filters (year range, fields of study)
4. Focus areas for the literature review

Be specific and actionable. Format your response as a structured plan.

Example output format:
**Key Concepts:**
- [concept 1]
- [concept 2]

**Search Queries:**
1. [query 1]
2. [query 2]

**Filters:**
- Year Range: [e.g., 2018-2024]
- Fields: [e.g., Computer Science, HCI]

**Focus Areas:**
- [area 1: e.g., Design patterns]
- [area 2: e.g., Evaluation methods]
"""
    
    def create_plan(self, topic: str, description: str = "") -> Dict[str, Any]:
        """
        Create research plan for the given topic.
        
        Args:
            topic: Main research topic
            description: Detailed project description (optional)
            
        Returns:
            Dictionary with search plan structure
        """
        self.logger.info(f"Creating search plan for topic: {topic}")
        
        user_message = f"""Research Topic: {topic}

Project Description: {description if description else "Not provided"}

Please analyze this research topic and create a comprehensive search strategy."""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_message)
        ]
        
        response = self.invoke(messages)
        
        return {
            "plan_text": response,
            "topic": topic,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }


class ResearcherAgent(BaseLangGraphAgent):
    """
    Researcher Agent - Searches for papers and gathers information.
    
    Responsibilities:
    - Execute searches using Semantic Scholar and web search
    - Retrieve relevant papers
    - Extract key information
    - Rank and filter results
    """
    
    def __init__(self, config: Dict[str, Any], tools: Dict[str, Any]):
        system_prompt = config.get("system_prompt", "") or self._default_prompt()
        super().__init__(
            name="Researcher",
            role=config.get("role", "Paper Researcher"),
            model_config=config.get("model_config", {}),
            system_prompt=system_prompt
        )
        self.max_papers = config.get("max_papers", 10)
        self.max_sources = config.get("max_sources", 15)
        self.tools = tools
    
    def _default_prompt(self) -> str:
        return """You are an expert researcher specializing in finding relevant academic papers.

Your task is to search for papers based on a research plan and gather relevant information.

You have access to:
- paper_search(): Search academic papers via Semantic Scholar
- web_search(): Search for supplementary web sources

Search for 8-12 highly relevant papers. Prioritize:
- Recent publications (last 5-7 years)
- Highly cited papers
- Papers from reputable venues
- Diversity of perspectives

For each paper, note:
- Title, authors, year
- Key findings or contributions
- Relevance to the research topic
- Citation count and venue

Be thorough but selective - quality over quantity.
"""
    
    async def search_papers(self, search_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search for papers based on the search plan.
        
        Args:
            search_plan: Plan created by PlannerAgent
            
        Returns:
            List of papers with metadata
        """
        self.logger.info("Starting paper search")
        
        papers = []
        
        # Extract topic and description from plan
        topic = search_plan.get("topic", "")
        plan_text = search_plan.get("plan_text", "")
        
        # Use Semantic Scholar to search for papers
        if "paper_search" in self.tools:
            paper_tool = self.tools["paper_search"]
            
            # Search with the main topic
            try:
                results = await paper_tool.search(
                    query=topic,
                    year_from=2018,  # Focus on recent papers
                    min_citations=5   # Filter for impactful papers
                )
                papers.extend(results[:self.max_papers])
                self.logger.info(f"Found {len(results)} papers from Semantic Scholar")
            except Exception as e:
                self.logger.error(f"Error searching papers: {e}")
        
        return papers


class AnalyzerAgent(BaseLangGraphAgent):
    """
    Analyzer Agent - Identifies patterns, methodologies, and state-of-the-art.
    
    Responsibilities:
    - Review collected papers
    - Identify common themes and patterns
    - Compare different approaches
    - Extract state-of-the-art technologies
    - Organize findings by categories
    """
    
    def __init__(self, config: Dict[str, Any]):
        system_prompt = config.get("system_prompt", "") or self._default_prompt()
        super().__init__(
            name="Analyzer",
            role=config.get("role", "Pattern Analyzer"),
            model_config=config.get("model_config", {}),
            system_prompt=system_prompt
        )
    
    def _default_prompt(self) -> str:
        return """You are an expert at analyzing research papers and identifying patterns.

Your task is to analyze a collection of papers and extract:

1. **Common Themes**: What are the recurring topics across papers?
2. **Design Patterns**: What methodologies and approaches are commonly used?
3. **Evolution**: How has the field evolved over time?
4. **State-of-the-Art**: What are the most recent/advanced techniques?
5. **Gaps**: What aspects are understudied?
6. **Comparisons**: How do different approaches compare?

Organize your analysis into clear sections. Be specific and cite papers when making claims.

Your analysis should help a researcher understand:
- What has been done in this area
- What technologies/methods are being used
- Where the field is heading
- What opportunities exist for new research
"""
    
    def analyze_papers(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze collected papers to identify patterns and themes.
        
        Args:
            papers: List of papers from ResearcherAgent
            
        Returns:
            Dictionary with analysis structure
        """
        self.logger.info(f"Analyzing {len(papers)} papers")
        
        # Create a summary of papers for analysis
        papers_summary = self._create_papers_summary(papers)
        
        user_message = f"""I have collected the following papers for analysis:

{papers_summary}

Please analyze these papers and identify:
1. Common themes and patterns
2. Design patterns and methodologies
3. State-of-the-art technologies
4. Evolution of the field
5. Research gaps and opportunities
6. Comparison of different approaches

Organize your analysis into clear sections with specific examples from the papers."""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_message)
        ]
        
        response = self.invoke(messages)
        
        return {
            "analysis_text": response,
            "num_papers_analyzed": len(papers),
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_papers_summary(self, papers: List[Dict[str, Any]]) -> str:
        """Create a concise summary of papers for analysis."""
        summary = []
        for i, paper in enumerate(papers, 1):
            authors = ", ".join([a.get("name", "") for a in paper.get("authors", [])[:2]])
            if len(paper.get("authors", [])) > 2:
                authors += " et al."
            
            summary.append(
                f"{i}. {paper.get('title', 'Unknown')} "
                f"({authors}, {paper.get('year', 'n.d.')})\n"
                f"   Citations: {paper.get('citation_count', 0)} | "
                f"Venue: {paper.get('venue', 'N/A')}\n"
                f"   Abstract: {paper.get('abstract', '')[:200]}...\n"
            )
        
        return "\n".join(summary)


class WriterAgent(BaseLangGraphAgent):
    """
    Writer Agent - Synthesizes findings into comprehensive literature review.
    
    Responsibilities:
    - Synthesize analysis into coherent review
    - Organize by themes or chronologically
    - Include proper citations (APA format)
    - Write clear, academic prose
    - Create comprehensive literature review
    """
    
    def __init__(self, config: Dict[str, Any], citation_tool):
        system_prompt = config.get("system_prompt", "") or self._default_prompt()
        super().__init__(
            name="Writer",
            role=config.get("role", "Literature Review Writer"),
            model_config=config.get("model_config", {}),
            system_prompt=system_prompt
        )
        self.citation_format = config.get("citation_format", "APA")
        self.citation_tool = citation_tool
    
    def _default_prompt(self) -> str:
        return """You are an expert academic writer specializing in literature reviews.

Your task is to synthesize research findings into a comprehensive, well-structured literature review.

Your writing should:
- Be clear, concise, and academic in tone
- Use proper APA citations throughout (Author et al., Year)
- Organize findings logically (by theme, chronology, or methodology)
- Include smooth transitions between sections
- Highlight key findings and contributions
- Discuss relationships between different works
- Identify gaps and future directions

Structure your review with:
1. Introduction (context and scope)
2. Main body (organized by themes/categories)
3. Conclusion (synthesis and gaps)

Write in a style appropriate for a research paper's literature review section.
Include proper in-text citations when referencing specific papers.
"""
    
    def write_review(
        self,
        topic: str,
        analysis: Dict[str, Any],
        papers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Write comprehensive literature review.
        
        Args:
            topic: Research topic
            analysis: Analysis from AnalyzerAgent
            papers: List of papers from ResearcherAgent
            
        Returns:
            Dictionary with review text and citations
        """
        self.logger.info("Writing literature review")
        
        # Add papers to citation tool
        for paper in papers:
            paper["type"] = "paper"
            self.citation_tool.add_citation(paper)
        
        # Create message for writer
        analysis_text = analysis.get("analysis_text", "")
        papers_list = self._create_papers_list(papers)
        
        user_message = f"""Research Topic: {topic}

Analysis of Papers:
{analysis_text}

Available Papers:
{papers_list}

Please write a comprehensive literature review that:
1. Introduces the topic and its importance
2. Organizes findings into logical themes
3. Discusses key papers and their contributions
4. Compares different approaches
5. Identifies gaps and future directions
6. Uses proper in-text citations (Author et al., Year)

Write in clear, academic prose suitable for a research paper.
"""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_message)
        ]
        
        response = self.invoke(messages)
        
        # Generate bibliography
        bibliography = self.citation_tool.generate_bibliography()
        
        return {
            "review_text": response,
            "bibliography": bibliography,
            "num_citations": len(bibliography),
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_papers_list(self, papers: List[Dict[str, Any]]) -> str:
        """Create formatted list of papers for reference."""
        papers_list = []
        for i, paper in enumerate(papers, 1):
            authors = ", ".join([a.get("name", "") for a in paper.get("authors", [])[:3]])
            if len(paper.get("authors", [])) > 3:
                authors += " et al."
            
            papers_list.append(
                f"[{i}] {authors} ({paper.get('year', 'n.d.')}). "
                f"{paper.get('title', 'Unknown')}. "
                f"{paper.get('venue', 'N/A')}."
            )
        
        return "\n".join(papers_list)
