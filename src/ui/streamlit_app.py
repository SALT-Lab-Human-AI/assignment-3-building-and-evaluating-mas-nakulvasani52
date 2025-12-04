"""
Streamlit Web Interface
Web UI for the multi-agent literature review system.

Run with: streamlit run src/ui/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import yaml
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Literature Review Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_config():
    """Load configuration file."""
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@st.cache_resource
def initialize_orchestrator():
    """Initialize the LangGraph orchestrator."""
    from src.langgraph_orchestrator import LangGraphOrchestrator
    
    config = load_config()
    return LangGraphOrchestrator(config)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False


def process_query(query: str, description: str = ""):
    """
    Process a query through the orchestrator.
    
    Args:
        query: Research query to process
        description: Project description (optional)
        
    Returns:
        Result dictionary with response, citations, and metadata
    """
    orchestrator = initialize_orchestrator()
    
    with st.spinner("🔍 Processing your literature review request..."):
        result = orchestrator.process_query(query, description)
    
    return result


def display_agent_traces(traces: list):
    """
    Display agent execution traces.
    """
    st.subheader("🤖 Agent Activity")
    
    if not traces:
        st.info("No agent activity recorded.")
        return
    
    # Create expandable sections for each agent
    for trace in traces:
        agent_name = trace.get("agent", "Unknown")
        action = trace.get("action", "")
        timestamp = trace.get("timestamp", "")
        duration = trace.get("duration_seconds", 0)
        details = trace.get("details", "")
        
        # Format timestamp
        try:
            ts = datetime.fromisoformat(timestamp)
            time_str = ts.strftime("%H:%M:%S")
        except:
            time_str = timestamp
        
        # Color-code by agent
        agent_colors = {
            "SafetyManager": "🛡️",
            "Planner": "📋",
            "Researcher": "🔍",
            "Analyzer": "🔬",
            "Writer": "✍️",
            "QualityCheck": "✅",
            "Judge": "⚖️"
        }
        
        icon = agent_colors.get(agent_name, "🤖")
        tool_used = trace.get("tool", "")
        evaluation = trace.get("evaluation", {})
        
        if tool_used == "tavily":
            with st.expander(f"🌐 **Tavily Web Search** - {action} ({time_str})", expanded=True):
                st.success("✅ Tavily API was used for this search")
                if duration:
                    st.caption(f"⏱️ Duration: {duration:.2f}s")
                if details:
                    st.info(details)
        elif agent_name == "Judge" and evaluation:
            # Special display for Judge evaluation
            overall_score = evaluation.get("overall_score", 0)
            with st.expander(f"⚖️ **LLM Judge** - {action} ({time_str})", expanded=True):
                st.success(f"📊 Overall Quality Score: **{overall_score:.1f}/10**")
                
                # Display criteria scores
                criteria_scores = evaluation.get("criteria_scores", {})
                if criteria_scores:
                    st.markdown("#### Evaluation Breakdown:")
                    for criterion, data in criteria_scores.items():
                        score = data.get("score", 0)
                        weight = data.get("weight", 0)
                        
                        # Color code based on score
                        if score >= 8:
                            color = "🟢"
                        elif score >= 6:
                            color = "🟡"
                        else:
                            color = "🔴"
                        
                        st.markdown(f"{color} **{criterion}**: {score:.1f}/10 (weight: {weight:.0%})")
                    
                    # Show feedback
                    feedback = evaluation.get("feedback", {})
                    if feedback:
                        with st.expander("📝 Detailed Feedback"):
                            for criterion, fb in feedback.items():
                                st.markdown(f"**{criterion}:**")
                                st.text(fb)
                                st.divider()
                
                if duration:
                    st.caption(f"⏱️ Evaluation time: {duration:.2f}s")
        else:
            with st.expander(f"{icon} **{agent_name}** - {action} ({time_str})", expanded=False):
                if duration:
                    st.caption(f"⏱️ Duration: {duration:.2f}s")
                if details:
                    st.text(details)


def display_response(result: Dict[str, Any]):
    """
    Display query response with literature review.
    """
    st.subheader("📝 Literature Review")
    
    response = result.get("response", "")
    bibliography = result.get("bibliography", [])
    papers = result.get("papers", [])
    safety_events = result.get("safety_events", [])
    metadata = result.get("metadata", {})
    
    # Display safety warnings if any
    if safety_events:
        st.warning("⚠️ Safety Events Detected")
        for event in safety_events:
            st.error(f"**{event.get('category', 'Unknown')}**: {event.get('reason', '')}")
    
    # Display main review
    if response:
        st.markdown(response)
    else:
        st.info("No response generated.")
    
    # Display bibliography
    if bibliography:
        st.divider()
        st.subheader("📚 References")
        for i, citation in enumerate(bibliography, 1):
            st.markdown(f"{i}. {citation}")
    
    # Display source papers in sidebar
    with st.sidebar:
        if papers:
            st.subheader(f"📄 Source Papers ({len(papers)})")
            for paper in papers:
                with st.expander(f"{paper.get('title', 'Unknown')[:50]}..."):
                    authors = ", ".join([a.get("name", "") for a in paper.get("authors", [])[:2]])
                    if len(paper.get("authors", [])) > 2:
                        authors += " et al."
                    
                    st.write(f"**Authors:** {authors}")
                    st.write(f"**Year:** {paper.get('year', 'N/A')}")
                    st.write(f"**Citations:** {paper.get('citation_count', 0)}")
                    if paper.get('venue'):
                        st.write(f"**Venue:** {paper.get('venue')}")
                    if paper.get('url'):
                        st.markdown(f"[View Paper]({paper.get('url')})")


def display_sidebar():
    """Display sidebar with settings and statistics."""
    config = load_config()
    
    with st.sidebar:
        st.title("📚 Literature Review Assistant")
        st.caption("Powered by LangGraph + Groq")
        
        st.divider()
        
        # System info
        st.subheader("System Configuration")
        st.info(f"**Model:** {config.get('models', {}).get('default', {}).get('name', 'Unknown')}")
        st.info(f"**Max Iterations:** {config.get('system', {}).get('max_iterations', 3)}")
        
        # Session statistics
        if st.session_state.history:
            st.subheader("Session Statistics")
            st.metric("Queries Processed", len(st.session_state.history))
            
            total_papers = sum(
                len(r.get("papers", [])) 
                for r in st.session_state.history
            )
            st.metric("Total Papers Found", total_papers)
        
        st.divider()
        
        # Configuration notes
        with st.expander("ℹ️ About"):
            st.markdown("""
            This system uses 4 specialized agents:
            - 🎯 **Planner**: Creates search strategy
            - 🔍 **Researcher**: Finds relevant papers
            - 🔬 **Analyzer**: Identifies patterns
            - ✍️ **Writer**: Synthesizes review
            
            Papers are sourced from Semantic Scholar and evaluated for relevance.
            """)
        
        # Clear history button
        if st.session_state.history:
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.session_state.current_result = None
                st.rerun()


def display_history():
    """Display query history."""
    if not st.session_state.history:
        return
    
    st.subheader("📜 Query History")
    
    for i, hist_item in enumerate(reversed(st.session_state.history), 1):
        query = hist_item.get("query", "Unknown")
        timestamp = hist_item.get("timestamp", "")
        num_papers = hist_item.get("num_papers", 0)
        
        try:
            ts = datetime.fromisoformat(timestamp)
            time_str = ts.strftime("%Y-%m-%d %H:%M")
        except:
            time_str = timestamp
        
        with st.expander(f"Query {len(st.session_state.history) - i + 1}: {query[:60]}... ({time_str})"):
            st.write(f"**Papers Found:** {num_papers}")
            if st.button(f"Reload", key=f"reload_{i}"):
                st.session_state.current_result = hist_item.get("result")
                st.rerun()


def main():
    """Main Streamlit app."""
    initialize_session_state()
    display_sidebar()
    
    # Main content
    st.title("📚 Literature Review Assistant")
    st.markdown("""
    Get comprehensive literature reviews for your research project. 
    Our multi-agent system searches academic papers, identifies patterns, and synthesizes findings.
    """)
    
    # Input form
    with st.form("query_form"):
        st.subheader("🎯 Your Research Topic")
        
        query = st.text_area(
            "Research Topic",
            placeholder="e.g., Explainable AI in healthcare applications",
            help="Enter the main topic you want to research",
            height=100
        )
        
        description = st.text_area(
            "Project Description (Optional)",
            placeholder="Provide additional context about your research project, specific aspects you're interested in, or constraints...",
            help="Optional: Add more details about your project to get better results",
            height=150
        )
        
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            submit = st.form_submit_button("🚀 Generate Review", use_container_width=True)
        with col2:
            example = st.form_submit_button("💡 Load Example", use_container_width=True)
    
    # Handle example button
    if example:
        st.session_state.example_query = "Design patterns for accessible user interfaces in mobile applications"
        st.session_state.example_description = "I'm researching how to make mobile apps more accessible for users with disabilities. I'm particularly interested in design patterns, best practices, and recent innovations in this area."
        st.rerun()
    
    # Pre-fill with example if requested
    if hasattr(st.session_state, 'example_query'):
        query = st.session_state.example_query
        description = st.session_state.example_description
        delattr(st.session_state, 'example_query')
        delattr(st.session_state, 'example_description')
    
    # Handle form submission
    if submit and query:
        # Check if API key is set
        if not os.getenv("GROQ_API_KEY"):
            st.error("❌ GROQ_API_KEY not found! Please set your API key in the .env file.")
            st.info("Create a .env file in the project root and add: GROQ_API_KEY=your_key_here")
            return
        
        try:
            # Process query
            result = process_query(query, description)
            
            # Store in history
            history_item = {
                "query": query,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "num_papers": len(result.get("papers", [])),
                "result": result
            }
            st.session_state.history.append(history_item)
            st.session_state.current_result = result
            
            st.success("✅ Literature review generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            st.exception(e)
    
    elif submit and not query:
        st.warning("⚠️ Please enter a research topic.")
    
    # Display results
    if st.session_state.current_result:
        st.divider()
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📝 Review", "🤖 Agent Traces", "📊 Metadata"])
        
        with tab1:
            display_response(st.session_state.current_result)
        
        with tab2:
            traces = st.session_state.current_result.get("agent_traces", [])
            display_agent_traces(traces)
        
        with tab3:
            st.subheader("📊 Processing Metadata")
            metadata = st.session_state.current_result.get("metadata", {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Papers Found", metadata.get("num_papers", 0))
            with col2:
                st.metric("Citations", metadata.get("num_citations", 0))
            with col3:
                st.metric("Duration", f"{metadata.get('duration_seconds', 0):.2f}s")
            
            # Show detailed metadata
            with st.expander("Detailed Metadata"):
                st.json(metadata)
    
    # Display history at the bottom
    if st.session_state.history:
        st.divider()
        display_history()


if __name__ == "__main__":
    main()
