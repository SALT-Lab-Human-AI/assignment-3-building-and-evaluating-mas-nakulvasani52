"""
Safety Manager
Coordinates safety guardrails and logs safety events.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import json
import os


class SafetyManager:
    """
    Manages safety guardrails for the multi-agent system.
    
    Features:
    - Guardrails AI integration (optional)
    - Input validation (harmful content, off-topic, academic dishonesty)
    - Output validation (toxic language, bias, hallucinations)
    - Safety event logging
    - Configurable response strategies
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize safety manager.

        Args:
            config: Safety configuration
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.log_events = config.get("log_events", True)
        self.logger = logging.getLogger("safety")

        # Safety event log
        self.safety_events: List[Dict[str, Any]] = []

        # Prohibited categories
        self.prohibited_categories = config.get("prohibited_categories", [
            "harmful_content",
            "personal_attacks",
            "misinformation",
            "off_topic_queries"
        ])

        # Violation response strategy
        self.on_violation = config.get("on_violation", {})

        # Initialize guardrails
        self.use_guardrails_ai = config.get("framework", "") == "guardrails"
        
        if self.enabled and self.use_guardrails_ai:
            try:
                from guardrails import Guard
                from guardrails.validators import Validator, register_validator, ValidationResult, PassResult, FailResult
                
                # Define local validator to avoid Hub dependency
                @register_validator(name="local_toxic_language", data_type="string")
                class LocalToxicLanguage(Validator):
                    def __init__(self, threshold: float = 0.5, on_fail: str = "fix"):
                        super().__init__(on_fail=on_fail)
                        self.threshold = threshold
                        self.toxic_words = [
                            "idiot", "stupid", "dumb", "hate", "kill", "attack", 
                            "racist", "sexist", "scam", "fraud"
                        ]

                    def validate(self, value: Any, metadata: Dict = {}) -> ValidationResult:
                        lower_value = str(value).lower()
                        found_words = [w for w in self.toxic_words if w in lower_value]
                        
                        if found_words:
                            return FailResult(
                                error_message=f"Found toxic words: {', '.join(found_words)}",
                                fix_value=value  # Simple pass-through for fix, or could redact
                            )
                        return PassResult()
                
                # Create guard for input validation
                self.input_guard = Guard().use(
                    LocalToxicLanguage(threshold=0.5, on_fail="exception")
                )
                
                # Create guard for output validation
                self.output_guard = Guard().use(
                    LocalToxicLanguage(threshold=0.5, on_fail="fix")
                )
                
                self.logger.info("Guardrails AI initialized successfully with local validator")
            except Exception as e:
                self.logger.warning(f"Error initializing Guardrails AI: {e}, falling back to basic checks")
                self.use_guardrails_ai = False
        else:
            self.use_guardrails_ai = False

    def check_input(self, query: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Check if input query is safe to process.

        Args:
            query: User query to check

        Returns:
            Tuple of (is_safe, violations_list)
        """
        if not self.enabled:
            return True, []

        violations = []

        # Check 1: Harmful research topics
        harmful_keywords = [
            "weapon", "bomb", "terrorist", "illegal", "drug synthesis",
            "hack", "exploit", "malware", "virus creation"
        ]
        for keyword in harmful_keywords:
            if keyword in query.lower():
                violations.append({
                    "category": "harmful_research",
                    "reason": f"Query contains prohibited topic: {keyword}",
                    "severity": "high"
                })

        # Check 2: Personal attacks or bias
        attack_keywords = ["racist", "sexist", "discriminat", "hate"]
        for keyword in attack_keywords:
            if keyword in query.lower():
                violations.append({
                    "category": "inappropriate_content",
                    "reason": f"Query contains potentially inappropriate language: {keyword}",
                    "severity": "medium"
                })

        # Check 3: Academic dishonesty
        dishonesty_keywords = ["write my paper", "plagiarize", "cheat", "fake data"]
        for keyword in dishonesty_keywords:
            if keyword in query.lower():
                violations.append({
                    "category": "academic_dishonesty",
                    "reason": "Query suggests academic dishonesty",
                    "severity": "high"
                })

        # Check 4: Use Guardrails AI if available
        if self.use_guardrails_ai:
            try:
                result = self.input_guard.validate(query)
                if not result.validation_passed:
                    violations.append({
                        "category": "guardrails_ai_violation",
                        "reason": "Toxic language detected by Guardrails AI",
                        "severity": "medium"
                    })
            except Exception as e:
                self.logger.warning(f"Guardrails AI check failed: {e}")

        is_safe = len(violations) == 0

        # Log safety event
        if not is_safe and self.log_events:
            self._log_safety_event("input", query, violations, is_safe)

        return is_safe, violations

    def check_output(self, response: str) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        Check if output response is safe to return.

        Args:
            response: Generated response to check

        Returns:
            Tuple of (is_safe, sanitized_response, violations_list)
        """
        if not self.enabled:
            return True, response, []

        violations = []
        sanitized_response = response

        # Check 1: No hallucinated references (basic check)
        if "et al." in response:
            # Check for suspicious patterns like (Author et al., n.d.) or missing years
            import re
            if re.search(r'\(.*et al\.,\s*n\.d\.\s*\)', response):
                violations.append({
                    "category": "potential_hallucination",
                    "reason": "Found citation with 'n.d.' which may indicate hallucinated reference",
                    "severity": "low"
                })

        # Check 2: No personal attacks or biased language
        bias_keywords = ["obviously inferior", "clearly wrong", "stupid", "idiotic"]
        for keyword in bias_keywords:
            if keyword in response.lower():
                violations.append({
                    "category": "biased_language",
                    "reason": f"Response contains potentially biased language: {keyword}",
                    "severity": "medium"
                })

        # Check 3: Use Guardrails AI if available
        if self.use_guardrails_ai:
            try:
                result = self.output_guard.validate(response)
                if not result.validation_passed:
                    sanitized_response = result.validated_output or response
                    violations.append({
                        "category": "guardrails_ai_output",
                        "reason": "Output was sanitized by Guardrails AI",
                        "severity": "medium"
                    })
            except Exception as e:
                self.logger.warning(f"Guardrails AI output check failed: {e}")

        is_safe = len(violations) == 0

        # Log safety event
        if not is_safe and self.log_events:
            self._log_safety_event("output", response, violations, is_safe)

        # Apply configured action on violations
        if not is_safe:
            action = self.on_violation.get("action", "refuse")
            if action == "refuse":
                sanitized_response = self.on_violation.get(
                    "message",
                    "This response was blocked due to safety policy violations."
                )
            elif action == "sanitize":
                sanitized_response = self._sanitize_response(response, violations)

        return is_safe, sanitized_response, violations
    
    # Legacy methods for backward compatibility
    
    def check_input_safety(self, query: str) -> Dict[str, Any]:
        """
        Check if input query is safe to process (legacy method).

        Args:
            query: User query to check

        Returns:
            Dictionary with 'safe' boolean and optional 'violations' list
        """
        is_safe, violations = self.check_input(query)
        return {
            "safe": is_safe,
            "violations": violations
        }

    def check_output_safety(self, response: str) -> Dict[str, Any]:
        """
        Check if output response is safe to return (legacy method).

        Args:
            response: Generated response to check

        Returns:
            Dictionary with 'safe' boolean and optional 'violations' list
        """
        is_safe, sanitized, violations = self.check_output(response)
        return {
            "safe": is_safe,
            "violations": violations,
            "response": sanitized
        }

    def _sanitize_response(self, response: str, violations: List[Dict[str, Any]]) -> str:
        """
        Sanitize response by removing or redacting unsafe content.
        """
        # Basic sanitization - remove offensive keywords
        sanitized = response
        for violation in violations:
            if violation["category"] == "biased_language":
                # Replace biased keywords with [REDACTED]
                sanitized = sanitized.replace(
                    violation.get("reason", "").split(": ")[-1],
                    "[REDACTED]"
                )
        
        return sanitized

    def _log_safety_event(
        self,
        event_type: str,
        content: str,
        violations: List[Dict[str, Any]],
        is_safe: bool
    ):
        """
        Log a safety event.

        Args:
            event_type: "input" or "output"
            content: The content that was checked
            violations: List of violations found
            is_safe: Whether content passed safety checks
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "safe": is_safe,
            "violations": violations,
            "content_preview": content[:100] + "..." if len(content) > 100 else content
        }

        self.safety_events.append(event)
        self.logger.warning(f"Safety event: {event_type} - safe={is_safe}")

        # Write to safety log file if configured
        log_file = self.config.get("safety_log_file") or "logs/safety_events.log"
        if self.log_events:
            try:
                # Create logs directory if it doesn't exist
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                
                with open(log_file, "a") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception as e:
                self.logger.error(f"Failed to write safety log: {e}")

    def get_safety_events(self) -> List[Dict[str, Any]]:
        """Get all logged safety events."""
        return self.safety_events

    def get_safety_stats(self) -> Dict[str, Any]:
        """
        Get statistics about safety events.

        Returns:
            Dictionary with safety statistics
        """
        total = len(self.safety_events)
        input_events = sum(1 for e in self.safety_events if e["type"] == "input")
        output_events = sum(1 for e in self.safety_events if e["type"] == "output")
        violations = sum(1 for e in self.safety_events if not e["safe"])

        return {
            "total_events": total,
            "input_checks": input_events,
            "output_checks": output_events,
            "violations": violations,
            "violation_rate": violations / total if total > 0 else 0
        }

    def clear_events(self):
        """Clear safety event log."""
        self.safety_events = []
