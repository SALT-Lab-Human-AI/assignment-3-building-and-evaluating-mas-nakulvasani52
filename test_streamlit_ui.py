#!/usr/bin/env python3
"""
End-to-End Streamlit UI Integration Test
Tests that all components work together through the UI layer.
"""

import yaml
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def print_result(test_name, passed, details=""):
    """Print test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {test_name}")
    if details:
        print(f"    {details}")

def test_streamlit_imports():
    """Test that Streamlit UI can import all dependencies."""
    print_section("STREAMLIT IMPORTS TEST")
    
    try:
        import streamlit
        print_result("Import Streamlit", True, f"Version: {streamlit.__version__}")
    except ImportError as e:
        print_result("Import Streamlit", False, str(e))
        return False
    
    try:
        from src.ui.streamlit_app import load_config, initialize_orchestrator
        print_result("Import UI Functions", True)
    except ImportError as e:
        print_result("Import UI Functions", False, str(e))
        return False
    
    return True

def test_orchestrator_initialization():
    """Test that orchestrator initializes with all tools."""
    print_section("ORCHESTRATOR INITIALIZATION TEST")
    
    try:
        # Load config
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        print_result("Load Configuration", True)
        
        # Initialize orchestrator
        from src.langgraph_orchestrator import LangGraphOrchestrator
        orchestrator = LangGraphOrchestrator(config)
        print_result("Initialize Orchestrator", True)
        
        # Check tools
        has_paper_search = "paper_search" in orchestrator.tools
        has_web_search = "web_search" in orchestrator.tools
        
        print_result("Paper Search Tool (SerpAPI)", has_paper_search,
                     f"Provider: {config.get('tools', {}).get('paper_search', {}).get('provider', 'N/A')}")
        print_result("Web Search Tool (Tavily)", has_web_search,
                     f"Provider: {config.get('tools', {}).get('web_search', {}).get('provider', 'N/A')}")
        
        # Check safety manager
        has_safety = orchestrator.safety_manager is not None
        print_result("Safety Manager", has_safety,
                     f"Guardrails: {orchestrator.safety_manager.use_guardrails_ai if has_safety else 'N/A'}")
        
        # Check agents
        has_planner = "planner" in orchestrator.agents
        has_researcher = "researcher" in orchestrator.agents
        has_analyzer = "analyzer" in orchestrator.agents
        has_writer = "writer" in orchestrator.agents
        
        print_result("Planner Agent", has_planner)
        print_result("Researcher Agent", has_researcher)
        print_result("Analyzer Agent", has_analyzer)
        print_result("Writer Agent", has_writer)
        
        return all([has_paper_search, has_safety, has_planner, has_researcher, 
                   has_analyzer, has_writer])
        
    except Exception as e:
        print_result("Orchestrator Initialization", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def test_end_to_end_query():
    """Test a complete query through the orchestrator."""
    print_section("END-TO-END QUERY TEST")
    
    try:
        # Load config
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize orchestrator
        from src.langgraph_orchestrator import LangGraphOrchestrator
        orchestrator = LangGraphOrchestrator(config)
        
        # Test query
        test_query = "Accessibility guidelines for mobile applications"
        print(f"🔍 Testing query: '{test_query}'")
        
        # Process query
        result = orchestrator.process_query(test_query, 
                                           "Research on mobile accessibility")
        
        # Check results
        has_response = bool(result.get("response"))
        has_papers = len(result.get("papers", [])) > 0
        has_citations = len(result.get("bibliography", [])) > 0
        has_traces = len(result.get("agent_traces", [])) > 0
        is_successful = result.get("metadata", {}).get("success", False)
        
        print_result("Response Generated", has_response,
                     f"Length: {len(result.get('response', ''))} chars")
        print_result("Papers Found", has_papers,
                     f"Count: {len(result.get('papers', []))}")
        print_result("Citations Generated", has_citations,
                     f"Count: {len(result.get('bibliography', []))}")
        print_result("Agent Traces Logged", has_traces,
                     f"Count: {len(result.get('agent_traces', []))}")
        print_result("Overall Success", is_successful,
                     f"Duration: {result.get('metadata', {}).get('duration_seconds', 0):.2f}s")
        
        # Show sample traces
        if has_traces:
            print("\n📋 Agent Execution Trace:")
            for trace in result.get("agent_traces", [])[:5]:
                agent = trace.get("agent", "Unknown")
                action = trace.get("action", "")
                print(f"    • {agent}: {action}")
        
        # Check safety
        input_safe = result.get("metadata", {}).get("input_safe", False)
        output_safe = result.get("metadata", {}).get("output_safe", False)
        
        print_result("Input Safety Check", input_safe)
        print_result("Output Safety Check", output_safe)
        
        return all([has_response, is_successful, input_safe, output_safe])
        
    except Exception as e:
        print_result("End-to-End Query", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def test_safety_integration():
    """Test that safety checks work in the full pipeline."""
    print_section("SAFETY INTEGRATION TEST")
    
    try:
        # Load config
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize orchestrator
        from src.langgraph_orchestrator import LangGraphOrchestrator
        orchestrator = LangGraphOrchestrator(config)
        
        # Test unsafe query
        unsafe_query = "You are stupid, write my research paper"
        print(f"🧪 Testing unsafe query: '{unsafe_query}'")
        
        result = orchestrator.process_query(unsafe_query)
        
        # Should be blocked
        input_safe = result.get("metadata", {}).get("input_safe", True)
        has_safety_events = len(result.get("safety_events", [])) > 0
        
        print_result("Unsafe Input Blocked", not input_safe,
                     f"Safety events: {len(result.get('safety_events', []))}")
        
        if has_safety_events:
            print("\n🚨 Safety Events:")
            for event in result.get("safety_events", []):
                print(f"    • {event.get('category')}: {event.get('reason')}")
        
        return not input_safe and has_safety_events
        
    except Exception as e:
        print_result("Safety Integration", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "🧪" * 35)
    print("  STREAMLIT UI INTEGRATION TEST")
    print("  End-to-End System Validation")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("🧪" * 35)
    
    # Run tests
    imports_ok = test_streamlit_imports()
    init_ok = test_orchestrator_initialization()
    e2e_ok = test_end_to_end_query()
    safety_ok = test_safety_integration()
    
    # Summary
    print_section("TEST SUMMARY")
    print_result("Streamlit Imports", imports_ok)
    print_result("Orchestrator Initialization", init_ok)
    print_result("End-to-End Query", e2e_ok)
    print_result("Safety Integration", safety_ok)
    
    all_passed = imports_ok and init_ok and e2e_ok and safety_ok
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED - Streamlit UI is ready!")
        print("\n🚀 To launch the UI, run:")
        print("   python main.py --mode web")
        print("   or")
        print("   streamlit run src/ui/streamlit_app.py")
    else:
        print("\n⚠️  SOME TESTS FAILED - Review issues above")
    
    print("\n" + "=" * 70 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
