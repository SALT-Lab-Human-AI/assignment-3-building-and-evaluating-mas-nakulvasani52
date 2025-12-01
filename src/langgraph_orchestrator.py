"""
LangGraph Orchestrator
Coordinates multiple agents using LangGraph StateGraph for literature review.

Workflow:
START → Planner → Researcher → Analyzer → Writer → Quality Check → END
                                                         ↓
                                                    [Needs Revision]
                                                         ↓
                                                    Analyzer (revise)
"""

from typing import Dict, Any, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END

import logging
from datetime import datetime

from src.agents.langgraph_agents import (
    PlannerAgent,
    ResearcherAgent,
    AnalyzerAgent,
    WriterAgent
)
from src.tools.paper_search import PaperSearchTool
from src.tools.web_search import WebSearchTool
from src.tools.citation_tool import CitationTool
from src.guardrails.safety_manager import SafetyManager


logger = logging.getLogger("orchestrator.langgraph")


# Define the state schema for the workflow
class LitReviewState(TypedDict):
    """State schema for literature review workflow."""
    # Input
    query: str
    project_description: str
    
    # Intermediate results
    search_plan: Dict[str, Any]
    papers: List[Dict[str, Any]]
    analysis: Dict[str, Any]
    draft: Dict[str, Any]
    
    # Control flow
    revision_count: int
    max_revisions: int
    needs_revision: bool
    
    # Safety
    safety_events: List[Dict[str, Any]]
    input_safe: bool
    output_safe: bool
    
    # Final output
    final_response: str
    bibliography: List[str]
    metadata: Dict[str, Any]
    
    # Messages for transparency
    agent_messages: List[Dict[str, Any]]


class LangGraphOrchestrator:
    """
    Orchestrates multiple agents using LangGraph for literature review.
    
    Features:
    - State-based workflow with LangGraph
    - 4 specialized agents (Planner, Researcher, Analyzer, Writer)
    - Conditional revision logic
    - Safety guardrails integration
    - Transparent agent traces
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LangGraph orchestrator.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("orchestrator.langgraph")
        
        # Initialize safety manager
        safety_config = config.get("safety", {})
        self.safety_manager = SafetyManager(safety_config) if safety_config.get("enabled") else None
        
        # Initialize tools
        tools_config = config.get("tools", {})
        self.tools = self._init_tools(tools_config)
        
        # Initialize citation tool
        self.citation_tool = CitationTool(style="apa")
        
        # Initialize agents
        model_config = config.get("models", {}).get("default", {})
        agents_config = config.get("agents", {})
        
        self.agents = {
            "planner": PlannerAgent({
                **agents_config.get("planner", {}),
                "model_config": model_config
            }),
            "researcher": ResearcherAgent({
                **agents_config.get("researcher", {}),
                "model_config": model_config
            }, self.tools),
            "analyzer": AnalyzerAgent({
                **agents_config.get("analyzer", {}),
                "model_config": model_config
            }),
            "writer": WriterAgent({
                **agents_config.get("writer", {}),
                "model_config": model_config
            }, self.citation_tool)
        }
        
        # Build workflow graph
        self.workflow = self._build_workflow()
        
        self.logger.info("LangGraph orchestrator initialized")
    
    def _init_tools(self, tools_config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize research tools."""
        tools = {}
        
        # Paper search tool
        if tools_config.get("paper_search", {}).get("enabled", True):
            max_results = tools_config.get("paper_search", {}).get("max_results", 10)
            provider = tools_config.get("paper_search", {}).get("provider", "semantic_scholar")
            tools["paper_search"] = PaperSearchTool(max_results=max_results, provider=provider)
            self.logger.info(f"Initialized {provider} paper search")
        
        # Web search tool
        if tools_config.get("web_search", {}).get("enabled", True):
            provider = tools_config.get("web_search", {}).get("provider", "tavily")
            max_results = tools_config.get("web_search", {}).get("max_results", 5)
            tools["web_search"] = WebSearchTool(provider=provider, max_results=max_results)
            self.logger.info(f"Initialized {provider} web search")
        
        return tools
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow."""
        workflow = StateGraph(LitReviewState)
        
        # Add nodes for each agent
        workflow.add_node("safety_check_input", self._safety_check_input_node)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("researcher", self._researcher_node)
        workflow.add_node("analyzer", self._analyzer_node)
        workflow.add_node("writer", self._writer_node)
        workflow.add_node("quality_check", self._quality_check_node)
        workflow.add_node("safety_check_output", self._safety_check_output_node)
        
        # Define edges
        workflow.set_entry_point("safety_check_input")
        
        # Conditional edge after input safety check
        workflow.add_conditional_edges(
            "safety_check_input",
            self._should_continue_after_input_check,
            {
                "continue": "planner",
                "stop": END
            }
        )
        
        workflow.add_edge("planner", "researcher")
        workflow.add_edge("researcher", "analyzer")
        workflow.add_edge("analyzer", "writer")
        workflow.add_edge("writer", "quality_check")
        
        # Conditional edge after quality check
        workflow.add_conditional_edges(
            "quality_check",
            self._should_revise,
            {
                "revise": "analyzer",
                "approve": "safety_check_output"
            }
        )
        
        # Conditional edge after output safety check
        workflow.add_conditional_edges(
            "safety_check_output",
            self._should_continue_after_output_check,
            {
                "continue": END,
                "stop": END
            }
        )
        
        return workflow.compile()
    
    # Node functions
    
    def _safety_check_input_node(self, state: LitReviewState) -> LitReviewState:
        """Check input for safety violations."""
        self.logger.info("[Safety Check] Checking input")
        
        if not self.safety_manager:
            state["input_safe"] = True
            state["agent_messages"].append({
                "agent": "SafetyManager",
                "action": "Input safety check skipped (guardrails disabled)",
                "timestamp": datetime.now().isoformat()
            })
            return state
        
        # Check query
        is_safe, violations = self.safety_manager.check_input(state["query"])
        
        state["input_safe"] = is_safe
        
        if not is_safe:
            state["safety_events"].extend(violations)
            state["final_response"] = "This query was blocked due to safety policy violations."
            state["agent_messages"].append({
                "agent": "SafetyManager",
                "action": f"Blocked unsafe input: {violations}",
                "timestamp": datetime.now().isoformat()
            })
        else:
            state["agent_messages"].append({
                "agent": "SafetyManager",
                "action": "Input passed safety check",
                "timestamp": datetime.now().isoformat()
            })
        
        return state
    
    def _should_continue_after_input_check(self, state: LitReviewState) -> str:
        """Decide whether to continue after input safety check."""
        return "continue" if state["input_safe"] else "stop"
    
    def _planner_node(self, state: LitReviewState) -> LitReviewState:
        """Execute Planner agent."""
        self.logger.info("[Planner] Creating search plan")
        
        agent_start = datetime.now()
        
        search_plan = self.agents["planner"].create_plan(
            topic=state["query"],
            description=state.get("project_description", "")
        )
        
        state["search_plan"] = search_plan
        state["agent_messages"].append({
            "agent": "Planner",
            "action": "Created search strategy",
            "details": search_plan.get("plan_text", "")[:200] + "...",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - agent_start).total_seconds()
        })
        
        return state
    
    def _researcher_node(self, state: LitReviewState) -> LitReviewState:
        """Execute Researcher agent."""
        self.logger.info("[Researcher] Searching for papers")
        
        agent_start = datetime.now()
        
        # Search for papers (synchronous version for now)
        import asyncio
        papers = asyncio.run(self.agents["researcher"].search_papers(state["search_plan"]))
        
        state["papers"] = papers
        state["agent_messages"].append({
            "agent": "Researcher",
            "action": f"Found {len(papers)} relevant papers",
            "details": f"Papers from {min([p.get('year', 9999) for p in papers]) if papers else 'N/A'} to {max([p.get('year', 0) for p in papers]) if papers else 'N/A'}",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - agent_start).total_seconds()
        })
        
        return state
    
    def _analyzer_node(self, state: LitReviewState) -> LitReviewState:
        """Execute Analyzer agent."""
        self.logger.info("[Analyzer] Analyzing papers")
        
        agent_start = datetime.now()
        
        analysis = self.agents["analyzer"].analyze_papers(state["papers"])
        
        state["analysis"] = analysis
        state["agent_messages"].append({
            "agent": "Analyzer",
            "action": "Analyzed papers and identified patterns",
            "details": f"Analyzed {len(state['papers'])} papers",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - agent_start).total_seconds()
        })
        
        return state
    
    def _writer_node(self, state: LitReviewState) -> LitReviewState:
        """Execute Writer agent."""
        self.logger.info("[Writer] Writing literature review")
        
        agent_start = datetime.now()
        
        draft = self.agents["writer"].write_review(
            topic=state["query"],
            analysis=state["analysis"],
            papers=state["papers"]
        )
        
        state["draft"] = draft
        state["agent_messages"].append({
            "agent": "Writer",
            "action": "Drafted literature review",
            "details": f"Review length: {len(draft.get('review_text', ''))} characters, {draft.get('num_citations', 0)} citations",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - agent_start).total_seconds()
        })
        
        return state
    
    def _quality_check_node(self, state: LitReviewState) -> LitReviewState:
        """Check quality of draft."""
        self.logger.info("[Quality Check] Evaluating draft")
        
        # Simple quality checks
        draft = state["draft"]
        review_text = draft.get("review_text", "")
        
        # Check 1: Sufficient length
        if len(review_text) < 500:
            state["needs_revision"] = True
            state["revision_count"] = state.get("revision_count", 0) + 1
            state["agent_messages"].append({
                "agent": "QualityCheck",
                "action": f"Draft too short, needs revision (attempt {state['revision_count']})",
                "timestamp": datetime.now().isoformat()
            })
            return state
        
        # Check 2: Has citations
        if draft.get("num_citations", 0) == 0:
            state["needs_revision"] = True
            state["revision_count"] = state.get("revision_count", 0) + 1
            state["agent_messages"].append({
                "agent": "QualityCheck",
                "action": f"No citations found, needs revision (attempt {state['revision_count']})",
                "timestamp": datetime.now().isoformat()
            })
            return state
        
        # Draft is good
        state["needs_revision"] = False
        state["agent_messages"].append({
            "agent": "QualityCheck",
            "action": "Draft approved",
            "timestamp": datetime.now().isoformat()
        })
        
        return state
    
    def _should_revise(self, state: LitReviewState) -> str:
        """Decide whether to revise the draft."""
        # Check if we've hit max revisions
        if state["revision_count"] >= state["max_revisions"]:
            self.logger.warning("Max revisions reached, approving draft")
            return "approve"
        
        if state["needs_revision"]:
            self.logger.info(f"Revision needed (attempt {state.get('revision_count', 0)})")
            return "revise"
        
        return "approve"
    
    def _safety_check_output_node(self, state: LitReviewState) -> LitReviewState:
        """Check output for safety violations."""
        self.logger.info("[Safety Check] Checking output")
        
        if not self.safety_manager:
            state["output_safe"] = True
            state["final_response"] = state["draft"].get("review_text", "")
            state["bibliography"] = state["draft"].get("bibliography", [])
            return state
        
        # Check draft review
        review_text = state["draft"].get("review_text", "")
        is_safe, sanitized_text, violations = self.safety_manager.check_output(review_text)
        
        state["output_safe"] = is_safe
        
        if not is_safe:
            state["safety_events"].extend(violations)
            state["final_response"] = sanitized_text
            state["agent_messages"].append({
                "agent": "SafetyManager",
                "action": f"Output sanitized due to violations: {violations}",
                "timestamp": datetime.now().isoformat()
            })
        else:
            state["final_response"] = review_text
            state["agent_messages"].append({
                "agent": "SafetyManager",
                "action": "Output passed safety check",
                "timestamp": datetime.now().isoformat()
            })
        
        state["bibliography"] = state["draft"].get("bibliography", [])
        
        return state
    
    def _should_continue_after_output_check(self, state: LitReviewState) -> str:
        """Always continue to END after output check."""
        return "continue"
    
    # Public interface
    
    def process_query(self, query: str, project_description: str = "") -> Dict[str, Any]:
        """
        Process a literature review query through the agent workflow.
        
        Args:
            query: Research topic/query
            project_description: Detailed project description (optional)
            
        Returns:
            Dictionary with final review and metadata
        """
        self.logger.info(f"Processing query: {query}")
        
        start_time = datetime.now()
        
        # Initialize state
        initial_state: LitReviewState = {
            "query": query,
            "project_description": project_description,
            "search_plan": {},
            "papers": [],
            "analysis": {},
            "draft": {},
            "revision_count": 0,
            "max_revisions": self.config.get("system", {}).get("max_revisions", 2),
            "needs_revision": False,
            "safety_events": [],
            "input_safe": True,
            "output_safe": True,
            "final_response": "",
            "bibliography": [],
            "metadata": {},
            "agent_messages": []
        }
        
        # Run workflow
        try:
            final_state = self.workflow.invoke(initial_state)
        except Exception as e:
            self.logger.error(f"Error in workflow: {e}")
            return {
                "response": f"Error processing query: {str(e)}",
                "error": str(e),
                "metadata": {
                    "success": False,
                    "error_message": str(e)
                }
            }
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Compile results
        result = {
            "response": final_state.get("final_response", ""),
            "bibliography": final_state.get("bibliography", []),
            "papers": final_state.get("papers", []),
            "agent_traces": final_state.get("agent_messages", []),
            "safety_events": final_state.get("safety_events", []),
            "metadata": {
                "query": query,
                "num_papers": len(final_state.get("papers", [])),
                "num_citations": len(final_state.get("bibliography", [])),
                "num_revisions": final_state.get("revision_count", 0),
                "input_safe": final_state.get("input_safe", True),
                "output_safe": final_state.get("output_safe", True),
                "duration_seconds": duration,
                "timestamp": start_time.isoformat(),
                "success": True
            }
        }
        
        self.logger.info(f"Query processed successfully in {duration:.2f}s")
        
        return result
