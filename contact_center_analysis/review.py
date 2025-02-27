from typing import List, Dict, Any, Optional
from .base import BaseAnalyzer
import json

class Reviewer(BaseAnalyzer):
    """Review and refine analysis results from LLM outputs."""
    
    async def review_analysis(self,
                            analysis: Dict[str, Any],
                            criteria: List[str],
                            original_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Review LLM analysis results against specified criteria.
        
        Args:
            analysis: The LLM-generated analysis to review
            criteria: List of criteria to evaluate the analysis against
            original_prompt: Optional original prompt that generated the analysis
            
        Returns:
            Dictionary containing evaluation scores and improvement suggestions
        """
        prompt = f"""You are an expert evaluator of LLM-generated content. Review this LLM output against ONLY the specified criteria.

{"Original Prompt:" + chr(10) + original_prompt + chr(10) + chr(10) if original_prompt else ""}
LLM Output to Evaluate:
{json.dumps(analysis, indent=2)}

Evaluation Criteria (ONLY evaluate against these specific criteria):
{chr(10).join(f"- {criterion}" for criterion in criteria)}

Evaluate how well the LLM output meets EACH of the criteria listed above. Do not add additional criteria.

Return as JSON:
{{
    "criteria_scores": [
        {{
            "criterion": str,  # Must be one of the criteria listed above
            "score": float,    # 0.0-1.0 where 1.0 is perfect
            "assessment": str,
            "improvement_needed": bool
        }}
    ],
    "overall_quality": {{
        "score": float,  # 0.0-1.0 where 1.0 is perfect
        "strengths": [str],
        "weaknesses": [str]
    }},
    "prompt_effectiveness": {{
        "assessment": str,
        "suggested_improvements": [str]
    }}
}}"""

        return await self._generate_content(prompt)
    
    async def suggest_improvements(self,
                                 current_results: Dict[str, Any],
                                 original_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Suggest improvements for LLM-generated content.
        
        Args:
            current_results: The LLM output to improve
            original_prompt: Optional original prompt that generated the results
            
        Returns:
            List of suggested improvements with specific guidance
        """
        prompt = f"""You are an expert at improving LLM outputs. Analyze this LLM-generated content and suggest specific improvements.

{"Original Prompt:" + chr(10) + original_prompt + chr(10) + chr(10) if original_prompt else ""}
LLM Output to Improve:
{json.dumps(current_results, indent=2)}

Provide specific suggestions to improve this LLM output. Consider:
1. Missing information or insights
2. Structure and organization
3. Clarity and specificity
4. Actionability of recommendations
5. Potential biases or limitations
6. Prompt engineering improvements

Return as JSON:
{{
    "content_improvements": [
        {{
            "issue": str,
            "suggestion": str,
            "priority": int  # 1-5 where 5 is highest priority
        }}
    ],
    "prompt_improvements": [
        {{
            "issue": str,
            "suggested_prompt_modification": str,
            "rationale": str
        }}
    ],
    "revised_prompt": str  # A complete revised prompt that would likely produce better results
}}"""

        return await self._generate_content(prompt) 