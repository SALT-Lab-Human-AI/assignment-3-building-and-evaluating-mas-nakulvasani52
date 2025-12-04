
import logging
import sys
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_guardrails")

print("Testing Guardrails AI import...")

from src.guardrails.safety_manager import SafetyManager

print("Testing SafetyManager with Local Guardrails...")

config = {
    "enabled": True,
    "framework": "guardrails",
    "log_events": True
}

try:
    manager = SafetyManager(config)
    print(f"✓ SafetyManager initialized (Guardrails enabled: {manager.use_guardrails_ai})")
    
    # Test safe input
    safe_query = "What are the design patterns for mobile apps?"
    is_safe, violations = manager.check_input(safe_query)
    print(f"Safe query check: {'✓ Passed' if is_safe else '❌ Failed'}")
    
    # Test toxic input
    toxic_query = "You are stupid and I hate this."
    is_safe, violations = manager.check_input(toxic_query)
    print(f"Toxic query check: {'✓ Blocked' if not is_safe else '❌ Failed to block'}")
    if not is_safe:
        print(f"  Violations: {violations}")

except Exception as e:
    print(f"❌ Error: {e}")
    traceback.print_exc()

except ImportError as e:
    print(f"❌ ImportError: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"❌ Error: {e}")
    traceback.print_exc()
