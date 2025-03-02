from typing import List, Dict, Any, Optional
from .base import BaseAnalyzer
import json

class DataAnalyzer(BaseAnalyzer):
    """Analyze conversation data for patterns and insights."""
    
    async def analyze_trends(self, 
                           data: Dict[str, Any],
                           focus_areas: List[str]) -> Dict[str, Any]:
        """Analyze trends in the data for specified focus areas."""
        pass
    
    async def identify_patterns(self, 
                              data: Dict[str, Any],
                              pattern_types: List[str]) -> List[Dict[str, Any]]:
        """Identify specific patterns in the conversation data."""
        pass
    
    async def compare_segments(self,
                             segments: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare different segments of the data."""
        pass

    async def analyze_findings(
        self,
        attribute_values: Dict[str, Any],
        questions: List[str],
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze findings from attribute extraction.
        
        Args:
            attribute_values: Dictionary of extracted attribute values
            questions: List of research questions to answer
            batch_size: Optional number of items to process in each batch.
                       If not provided, defaults to 10
            
        Returns:
            Analysis results and answers to research questions
        """
        effective_batch_size = batch_size or 10
        
        prompt = f"""Based on the analysis of customer service conversations, help answer these questions:

Questions:
{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(questions))}

Analysis Data:
{json.dumps(attribute_values, indent=2)}

Please provide:
1. Specific answers to each question, citing the data
2. Key metrics (1-2 words or numbers) that quantify the answer when applicable
3. Confidence level (High/Medium/Low) for each answer
4. Identification of any data gaps

Format as JSON:
{{
    "answers": [
        {{
            "question": str,
            "answer": str,
            "key_metrics": [str],  
            "confidence": str,
            "supporting_data": str
        }}
    ],
    "data_gaps": [str]
}}"""

        return await self._generate_content(prompt)

    async def analyze_distribution(
        self,
        attribute_values: Dict[str, List[Any]],
        max_categories: int = 100
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze the distribution of attribute values.
        
        Args:
            attribute_values: Dictionary of attribute values
            max_categories: Maximum number of categories to analyze (default: 100)
            
        Returns:
            Dictionary containing analysis results
        """
        pass

    async def analyze_findings_batch(
        self,
        attribute_values: List[Dict[str, Any]],
        questions: List[str],
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze findings from multiple attribute extractions in parallel.
        
        Args:
            attribute_values: List of attribute value dictionaries to analyze
            questions: List of research questions to answer
            batch_size: Optional number of items to process in each batch.
                       If not provided, defaults to 10
            
        Returns:
            List of analysis results for each set of attribute values
        """
        async def process_attributes(attrs):
            return await self.analyze_findings(attrs, questions)
            
        return await self.process_in_batches(
            attribute_values,
            batch_size=batch_size,
            process_func=process_attributes
        )

    async def identify_key_attributes(
        self,
        questions: List[str],
        available_attributes: List[Dict[str, Any]],
        max_attributes: int = 5
    ) -> List[str]:
        """
        Identify which attributes are most important for answering the research questions.
        
        Args:
            questions: List of research questions to answer
            available_attributes: List of attribute dictionaries that were generated
            max_attributes: Maximum number of key attributes to identify (default: 5)
            
        Returns:
            List of field_names for the most important attributes
        """
        prompt = f"""Given these research questions and available attributes, identify the {max_attributes} most important 
attributes that will provide the most valuable insights for answering these questions.

Research Questions:
{chr(10).join(f"- {q}" for q in questions)}

Available Attributes:
{json.dumps([{
    "field_name": attr["field_name"],
    "title": attr["title"],
    "description": attr["description"]
} for attr in available_attributes], indent=2)}

For each selected attribute, explain why it's important for answering the research questions.
Return your response as a JSON array of objects with field_name, importance_score (1-10), and rationale.

Example:
[
  {{
    "field_name": "dispute_type",
    "importance_score": 9,
    "rationale": "Critical for understanding the most common types of fee disputes"
  }}
]
"""

        try:
            result = await self._generate_content(
                prompt,
                expected_format=[{"field_name": str, "importance_score": int, "rationale": str}]
            )
            
            # Sort by importance score and take the top max_attributes
            sorted_attributes = sorted(result, key=lambda x: x["importance_score"], reverse=True)
            top_attributes = sorted_attributes[:max_attributes]
            
            # Extract just the field_names
            key_attribute_names = [attr["field_name"] for attr in top_attributes]
            
            if self.debug:
                print(f"Identified {len(key_attribute_names)} key attributes:")
                for attr in top_attributes:
                    print(f"  - {attr['field_name']} (score: {attr['importance_score']}): {attr['rationale']}")
            
            return key_attribute_names
        
        except Exception as e:
            if self.debug:
                print(f"Error identifying key attributes: {e}")
            # Fall back to a reasonable default if there's an error
            return [attr["field_name"] for attr in available_attributes[:max_attributes]]
