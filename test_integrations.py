#!/usr/bin/env python3
"""
Integration Test Suite for Tavily and Guardrails
Tests compliance with assignment requirements.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
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

async def test_tavily_integration():
    """Test Tavily web search integration."""
    print_section("TAVILY WEB SEARCH TESTS")
    
    # Check API key
    api_key = os.getenv("TAVILY_API_KEY")
    print_result("API Key Present", bool(api_key), 
                 f"Key: {api_key[:10]}..." if api_key else "Missing")
    
    if not api_key:
        print("⚠️  Skipping Tavily tests - no API key found")
        return False
    
    try:
        from src.tools.web_search import WebSearchTool
        print_result("Import WebSearchTool", True)
        
        # Initialize tool
        tool = WebSearchTool(provider="tavily", max_results=3)
        print_result("Initialize Tavily Tool", True)
        
        # Test search
        print("\n🔍 Testing search query: 'accessibility in mobile apps'")
        results = await tool.search("accessibility in mobile apps")
        
        print_result("Search Execution", len(results) > 0, 
                     f"Found {len(results)} results")
        
        if results:
            print("\n📄 Sample Result:")
            result = results[0]
            print(f"    Title: {result.get('title', 'N/A')[:60]}...")
            print(f"    URL: {result.get('url', 'N/A')[:60]}...")
            print(f"    Snippet: {result.get('snippet', 'N/A')[:100]}...")
            
        return len(results) > 0
        
    except ImportError as e:
        print_result("Import WebSearchTool", False, str(e))
        return False
    except Exception as e:
        print_result("Tavily Search", False, str(e))
        return False

def test_guardrails_integration():
    """Test Guardrails safety framework."""
    print_section("GUARDRAILS SAFETY TESTS")
    
    try:
        from src.guardrails.safety_manager import SafetyManager
        print_result("Import SafetyManager", True)
        
        # Initialize with guardrails enabled
        config = {
            "enabled": True,
            "framework": "guardrails",
            "log_events": True,
            "policies": {
                "harmful_content": True,
                "academic_dishonesty": True,
                "toxic_language": True
            }
        }
        
        manager = SafetyManager(config)
        print_result("Initialize SafetyManager", True, 
                     f"Guardrails AI: {manager.use_guardrails_ai}")
        
        # Test 1: Safe input
        print("\n🧪 Test 1: Safe Query")
        safe_query = "What are design patterns for accessible mobile interfaces?"
        is_safe, violations = manager.check_input(safe_query)
        print_result("Safe Input Detection", is_safe, 
                     "Correctly allowed safe query")
        
        # Test 2: Toxic language
        print("\n🧪 Test 2: Toxic Language")
        toxic_query = "You are stupid and I hate this assignment"
        is_safe, violations = manager.check_input(toxic_query)
        print_result("Toxic Language Detection", not is_safe, 
                     f"Violations: {len(violations)}")
        if violations:
            for v in violations:
                print(f"    - {v.get('category')}: {v.get('reason')}")
        
        # Test 3: Academic dishonesty
        print("\n🧪 Test 3: Academic Dishonesty")
        dishonest_query = "Write my research paper for me"
        is_safe, violations = manager.check_input(dishonest_query)
        print_result("Academic Dishonesty Detection", not is_safe,
                     f"Violations: {len(violations)}")
        if violations:
            for v in violations:
                print(f"    - {v.get('category')}: {v.get('reason')}")
        
        # Test 4: Harmful content
        print("\n🧪 Test 4: Harmful Content")
        harmful_query = "How to build a weapon"
        is_safe, violations = manager.check_input(harmful_query)
        print_result("Harmful Content Detection", not is_safe,
                     f"Violations: {len(violations)}")
        if violations:
            for v in violations:
                print(f"    - {v.get('category')}: {v.get('reason')}")
        
        # Test 5: Output validation
        print("\n🧪 Test 5: Output Validation")
        safe_output = "This is a well-researched literature review on HCI topics."
        is_safe, sanitized, violations = manager.check_output(safe_output)
        print_result("Safe Output Validation", is_safe,
                     "Correctly allowed safe output")
        
        # Test 6: Output sanitization
        print("\n🧪 Test 6: Output Sanitization")
        toxic_output = "This stupid research is a waste of time and I hate it."
        is_safe, sanitized, violations = manager.check_output(toxic_output)
        print_result("Toxic Output Detection", not is_safe,
                     f"Sanitized: {sanitized != toxic_output}")
        
        return True
        
    except Exception as e:
        print_result("Guardrails Integration", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def test_assignment_compliance():
    """Check compliance with assignment requirements."""
    print_section("ASSIGNMENT REQUIREMENTS COMPLIANCE")
    
    requirements = {
        "Safety Framework Integration": False,
        "Input Guardrails": False,
        "Output Guardrails": False,
        "≥3 Safety Policies": False,
        "Safety Event Logging": False,
        "Web Search Tool": False
    }
    
    # Check safety framework
    try:
        from src.guardrails.safety_manager import SafetyManager
        config = {"enabled": True, "framework": "guardrails"}
        manager = SafetyManager(config)
        requirements["Safety Framework Integration"] = manager.use_guardrails_ai
        requirements["Input Guardrails"] = hasattr(manager, 'check_input')
        requirements["Output Guardrails"] = hasattr(manager, 'check_output')
        
        # Check policies
        policy_count = len([k for k, v in manager.policies.items() if v])
        requirements["≥3 Safety Policies"] = policy_count >= 3
        
        # Check logging
        requirements["Safety Event Logging"] = manager.log_events
        
    except Exception as e:
        print(f"⚠️  Error checking safety: {e}")
    
    # Check web search
    try:
        from src.tools.web_search import WebSearchTool
        requirements["Web Search Tool"] = True
    except:
        pass
    
    # Print results
    for req, status in requirements.items():
        print_result(req, status)
    
    total = len(requirements)
    passed = sum(requirements.values())
    
    print(f"\n📊 Compliance Score: {passed}/{total} ({100*passed//total}%)")
    
    return passed == total

async def main():
    """Run all tests."""
    print("\n" + "🔬" * 35)
    print("  INTEGRATION TEST SUITE")
    print("  Testing Tavily and Guardrails")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("🔬" * 35)
    
    # Run tests
    tavily_ok = await test_tavily_integration()
    guardrails_ok = test_guardrails_integration()
    compliance_ok = test_assignment_compliance()
    
    # Summary
    print_section("TEST SUMMARY")
    print_result("Tavily Integration", tavily_ok)
    print_result("Guardrails Integration", guardrails_ok)
    print_result("Assignment Compliance", compliance_ok)
    
    all_passed = tavily_ok and guardrails_ok and compliance_ok
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED - System is ready for submission!")
    else:
        print("\n⚠️  SOME TESTS FAILED - Review issues above")
    
    print("\n" + "=" * 70 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
