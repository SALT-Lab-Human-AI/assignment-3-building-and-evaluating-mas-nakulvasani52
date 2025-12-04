"""
LLM-as-a-Judge Evaluation
Implements evaluation using LLM to judge system outputs.
"""

from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import os
import logging
import json


logger = logging.getLogger("evaluation.judge")


class LLMJudge:
    """
    Evaluates system outputs using an LLM as a judge.
    
    Features:
    - Multiple evaluation criteria
    - Scoring on 0-10 scale
    - Detailed feedback for each criterion
    - Multiple judge prompts for robustness
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM judge.
        
        Args:
            config: Judge configuration with model settings and criteria
        """
        self.config = config
        self.criteria = config.get("criteria", [])
        
        # Initialize judge LLM
        model_config = config.get("judge_model", {})
        provider = model_config.get("provider", "groq")
        model_name = model_config.get("name", "llama-3.1-70b-versatile")
        temperature = model_config.get("temperature", 0.3)
        max_tokens = model_config.get("max_tokens", 2048)
        
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY is required for judge")
            
            self.llm = ChatGroq(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        logger.info(f"Initialized LLM Judge with model: {model_name}")
    
    def evaluate(
        self,
        query: str,
        response: str,
        papers: List[Dict[str, Any]],
        bibliography: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate a system output using multiple criteria.
        
        Args:
            query: Original research query
            response: Generated literature review
            papers: List of source papers used
            bibliography: List of citations
            
        Returns:
            Dictionary with scores and feedback for each criterion
        """
        logger.info(f"Evaluating response for query: {query}")
        
        results = {
            "query": query,
            "criteria_scores": {},
            "overall_score": 0.0,
            "feedback": {}
        }
        
        # Evaluate each criterion
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for criterion in self.criteria:
            criterion_name = criterion.get("name")
            weight = criterion.get("weight", 0.0)
            description = criterion.get("description", "")
            
            logger.info(f"Evaluating criterion: {criterion_name}")
            
            # Get score and feedback
            score, feedback = self._evaluate_criterion(
                criterion_name=criterion_name,
                description=description,
                query=query,
                response=response,
                papers=papers,
                bibliography=bibliography
            )
            
            results["criteria_scores"][criterion_name] = {
                "score": score,
                "weight": weight,
                "weighted_score": score * weight,
                "description": description
            }
            results["feedback"][criterion_name] = feedback
            
            total_weighted_score += score * weight
            total_weight += weight
        
        # Calculate overall score
        if total_weight > 0:
            results["overall_score"] = total_weighted_score / total_weight
        
        logger.info(f"Overall score: {results['overall_score']:.2f}/10")
        
        return results
    
    def _evaluate_criterion(
        self,
        criterion_name: str,
        description: str,
        query: str,
        response: str,
        papers: List[Dict[str, Any]],
        bibliography: List[str]
    ) -> tuple[float, str]:
        """
        Evaluate a single criterion.
        
        Returns:
            Tuple of (score, feedback)
        """
        # Create evaluation prompt based on criterion
        prompt = self._create_evaluation_prompt(
            criterion_name, description, query, response, papers, bibliography
        )
        
        try:
            messages = [
                SystemMessage(content="You are an expert evaluator of academic literature reviews."),
                HumanMessage(content=prompt)
            ]
            
            judge_response = self.llm.invoke(messages)
            content = judge_response.content
            
            # Parse score and feedback
            score, feedback = self._parse_judge_response(content)
            
            return score, feedback
            
        except Exception as e:
            logger.error(f"Error evaluating criterion {criterion_name}: {e}")
            return 5.0, f"Error during evaluation: {str(e)}"
    
    def _create_evaluation_prompt(
        self,
        criterion_name: str,
        description: str,
        query: str,
        response: str,
        papers: List[Dict[str, Any]],
        bibliography: List[str]
    ) -> str:
        """Create evaluation prompt for a specific criterion."""
        
        # Base information
        prompt = f"""You are evaluating a literature review on the following criterion:

**Criterion:** {criterion_name}
**Description:** {description}

**Original Research Query:**
{query}

**Generated Literature Review:**
{response}

**Number of Source Papers:** {len(papers)}
**Number of Citations:** {len(bibliography)}

"""
        
        # Add criterion-specific instructions
        if criterion_name == "relevance_coverage":
            prompt += """
Evaluate how well the review covers relevant papers and whether the scope is appropriate.

Consider:
- Are the papers relevant to the research topic?
- Is the coverage comprehensive or are major works missing?
- Is the scope well-defined?
- Are different perspectives included?

Score 0-10 where:
- 0-3: Poor coverage, many relevant works missing, unclear scope
- 4-6: Adequate coverage but missing some important works
- 7-9: Good coverage of relevant papers with clear scope
- 10: Excellent, comprehensive coverage with well-defined scope
"""
        
        elif criterion_name == "evidence_quality":
            prompt += f"""
Evaluate the quality and authority of sources, and proper citation formatting.

**Citations in Review:**
{chr(10).join(bibliography[:5])}... (showing first 5)

Consider:
- Are sources from reputable venues/journals?
- Are citations properly formatted (APA style)?
- Are highly-cited, authoritative papers included?
- Is citation placement appropriate in text?

Score 0-10 where:
- 0-3: Poor sources, improper citations, low authority
- 4-6: Adequate sources but some issues with citation quality
- 7-9: Good sources with proper citations
- 10: Excellent, authoritative sources with perfect citations
"""
        
        elif criterion_name == "comparative_analysis":
            prompt += """
Evaluate the identification of patterns, comparison of approaches, and state-of-the-art discussion.

Consider:
- Are common themes and patterns identified?
- Are different approaches compared and contrasted?
- Is the state-of-the-art discussed?
- Are methodologies analyzed?

Score 0-10 where:
- 0-3: No comparative analysis, just lists papers
- 4-6: Some comparison but superficial
- 7-9: Good comparative analysis with clear patterns
- 10: Excellent, deep comparative analysis with insights
"""
        
        elif criterion_name == "factual_accuracy":
            prompt += """
Evaluate the correctness of paper details and absence of hallucinated references.

Consider:
- Are paper titles, authors, and years accurate (based on provided sources)?
- Are claims properly supported by cited papers?
- Are there any suspicious citations (e.g., "n.d." or vague references)?
- Is information consistent?

Score 0-10 where:
- 0-3: Multiple factual errors or hallucinated references
- 4-6: Some minor inaccuracies
- 7-9: Mostly accurate with very minor issues
- 10: Completely accurate, no hallucinations
"""
        
        elif criterion_name == "safety_compliance":
            prompt += """
Evaluate whether the content is appropriate and follows academic standards.

Consider:
- Is the tone professional and academic?
- Is there any biased or offensive language?
- Is content appropriate for academic context?
- Are ethical considerations addressed if relevant?

Score 0-10 where:
- 0-3: Inappropriate content, biased language
- 4-6: Mostly appropriate but some unprofessional elements
- 7-9: Professional with minor issues
- 10: Perfectly appropriate academic tone
"""
        
        elif criterion_name == "clarity_organization":
            prompt += """
Evaluate the logical structure, clear writing, and smooth transitions.

Consider:
- Is the review well-structured and organized?
- Are transitions between sections smooth?
- Is the writing clear and easy to follow?
- Is there a logical flow of ideas?

Score 0-10 where:
- 0-3: Poorly organized, confusing, hard to follow
- 4-6: Adequate organization but could be clearer
- 7-9: Well-organized with clear writing
- 10: Excellently structured, crystal clear, perfect flow
"""
        
        # Add output format instruction
        prompt += """

**Output Format:**
Provide your evaluation in exactly this format:

SCORE: [0-10]
FEEDBACK: [Your detailed feedback explaining the score, specific strengths and weaknesses]

Be specific and constructive in your feedback.
"""
        
        return prompt
    
    def _parse_judge_response(self, response: str) -> tuple[float, str]:
        """
        Parse judge response to extract score and feedback.
        
        Returns:
            Tuple of (score, feedback)
        """
        try:
            # Look for score pattern
            score = 5.0  # default
            feedback = response
            
            lines = response.strip().split('\n')
            for i, line in enumerate(lines):
                if line.startswith("SCORE:"):
                    # Extract score
                    score_str = line.replace("SCORE:", "").strip()
                    try:
                        score = float(score_str)
                        score = max(0.0, min(10.0, score))  # Clamp to 0-10
                    except ValueError:
                        pass
                    
                elif line.startswith("FEEDBACK:"):
                    # Extract feedback (rest of the text)
                    feedback = '\n'.join(lines[i:]).replace("FEEDBACK:", "").strip()
                    break
            
            return score, feedback
            
        except Exception as e:
            logger.error(f"Error parsing judge response: {e}")
            return 5.0, response


def run_evaluation(
    config: Dict[str, Any],
    test_queries: List[Dict[str, Any]],
    orchestrator: Any
) -> Dict[str, Any]:
    """
    Run full evaluation on test queries.
    
    Args:
        config: Evaluation configuration
        test_queries: List of test queries with optional ground truth
        orchestrator: Orchestrator instance to process queries
        
    Returns:
        Dictionary with evaluation results
    """
    judge = LLMJudge(config)
    
    results = {
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "num_queries": len(test_queries),
        "query_results": [],
        "aggregate_scores": {}
    }
    
    # Evaluate each query
    for i, test_item in enumerate(test_queries, 1):
        logger.info(f"Evaluating query {i}/{len(test_queries)}")
        
        query = test_item.get("query", "")
        description = test_item.get("description", "")
        
        # Process query through orchestrator
        system_output = orchestrator.process_query(query, description)
        
        # Evaluate output
        evaluation = judge.evaluate(
            query=query,
            response=system_output.get("response", ""),
            papers=system_output.get("papers", []),
            bibliography=system_output.get("bibliography", [])
        )
        
        results["query_results"].append({
            "query": query,
            "system_output": system_output,
            "evaluation": evaluation
        })
    
    # Calculate aggregate scores
    if results["query_results"]:
        for criterion in config.get("criteria", []):
            criterion_name = criterion.get("name")
            scores = [
                r["evaluation"]["criteria_scores"].get(criterion_name, {}).get("score", 0)
                for r in results["query_results"]
            ]
            if scores:
                results["aggregate_scores"][criterion_name] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "scores": scores
                }
        
        # Overall scores
        overall_scores = [
            r["evaluation"]["overall_score"]
            for r in results["query_results"]
        ]
        results["aggregate_scores"]["overall"] = {
            "mean": sum(overall_scores) / len(overall_scores),
            "min": min(overall_scores),
            "max": max(overall_scores),
            "scores": overall_scores
        }
    
    return results
