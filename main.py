"""
Main Entry Point
Can be used to run the system or evaluation.

Usage:
  python main.py --mode web           # Run web interface
  python main.py --mode evaluate      # Run evaluation
  python main.py --mode test          # Test single query
"""

import argparse
import asyncio
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def run_web():
    """Run web interface."""
    import subprocess
    print("Starting Streamlit web interface...")
    subprocess.run(["streamlit", "run", "src/ui/streamlit_app.py"])


def run_evaluation():
    """Run system evaluation."""
    print("\n" + "=" * 70)
    print("RUNNING EVALUATION")
    print("=" * 70 + "\n")
    
    # Load configuration
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test queries
    with open("data/example_queries.json", 'r') as f:
        test_queries = json.load(f)
    
    print(f"Loaded {len(test_queries)} test queries")
    
    # Initialize orchestrator
    print("\nInitializing LangGraph orchestrator...")
    from src.langgraph_orchestrator import LangGraphOrchestrator
    orchestrator = LangGraphOrchestrator(config)
    print("✓ Orchestrator initialized")
    
    # Run evaluation
    print("\nRunning evaluation...")
    from src.evaluation.judge import run_evaluation
    
    eval_config = {
        **config.get("evaluation", {}),
        "judge_model": config.get("models", {}).get("judge", {})
    }
    
    results = run_evaluation(
        config=eval_config,
        test_queries=test_queries[:5],  # Evaluate first 5 queries to save time
        orchestrator=orchestrator
    )
    
    # Save results
    output_dir = Path("outputs/evaluations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"evaluation_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Evaluation results saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    if results.get("aggregate_scores"):
        print(f"\nEvaluated {results['num_queries']} queries")
        print("\nAggregate Scores:")
        
        for criterion, stats in results["aggregate_scores"].items():
            if criterion == "overall":
                print(f"\n  Overall Score: {stats['mean']:.2f}/10")
                print(f"    Range: {stats['min']:.2f} - {stats['max']:.2f}")
            else:
                print(f"\n  {criterion}: {stats['mean']:.2f}/10")
                print(f"    Range: {stats['min']:.2f} - {stats['max']:.2f}")
    
    print("\n" + "=" * 70)
    print(f"Full results available in: {output_file}")
    print("=" * 70 + "\n")


def run_test():
    """Run a test query."""
    print("\n" + "=" * 70)
    print("TEST MODE - Single Query")
    print("=" * 70 + "\n")
    
    # Load configuration
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize orchestrator
    print("Initializing LangGraph orchestrator...")
    from src.langgraph_orchestrator import LangGraphOrchestrator
    orchestrator = LangGraphOrchestrator(config)
    print("✓ Orchestrator initialized\n")
    
    # Test query
    test_query = "Design patterns for accessible user interfaces in mobile applications"
    test_description = "I'm researching how to make mobile apps more accessible for users with disabilities."
    
    print(f"Query: {test_query}")
    print(f"Description: {test_description}\n")
    
    print("Processing query...")
    result = orchestrator.process_query(test_query, test_description)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70 + "\n")
    
    print("Response:")
    print("-" * 70)
    print(result.get("response", "No response generated"))
    print()
    
    if result.get("bibliography"):
        print("\nBibliography:")
        print("-" * 70)
        for i, citation in enumerate(result["bibliography"], 1):
            print(f"{i}. {citation}")
    
    print("\nMetadata:")
    print("-" * 70)
    metadata = result.get("metadata", {})
    print(f"Papers Found: {metadata.get('num_papers', 0)}")
    print(f"Citations: {metadata.get('num_citations', 0)}")
    print(f"Revisions: {metadata.get('num_revisions', 0)}")
    print(f"Duration: {metadata.get('duration_seconds', 0):.2f}s")
    print(f"Success: {metadata.get('success', False)}")
    
    if result.get("safety_events"):
        print("\nSafety Events:")
        print("-" * 70)
        for event in result["safety_events"]:
            print(f"- {event.get('category')}: {event.get('reason')}")
    
    print("\nAgent Traces:")
    print("-" * 70)
    for trace in result.get("agent_traces", []):
        agent = trace.get("agent", "Unknown")
        action = trace.get("action", "")
        duration = trace.get("duration_seconds", 0)
        print(f"- {agent}: {action} ({duration:.2f}s)")
    
    print("\n" + "=" * 70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Literature Review Assistant - Multi-Agent System"
    )
    parser.add_argument(
        "--mode",
        choices=["web", "evaluate", "test"],
        default="web",
        help="Mode to run: web (default), evaluate, or test"
    )

    args = parser.parse_args()

    if args.mode == "web":
        run_web()
    elif args.mode == "evaluate":
        run_evaluation()
    elif args.mode == "test":
        run_test()


if __name__ == "__main__":
    main()
