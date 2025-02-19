from typing import List, Dict, Any, Optional
from .base import BaseAnalyzer

class TextGenerator(BaseAnalyzer):
    """Generate and analyze text content and data attributes."""
    
    async def generate_required_attributes(
        self,
        questions: List[str],
        existing_attributes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a list of required data attributes needed to answer specific questions.
        
        Args:
            questions: List of questions that need to be answered
            existing_attributes: Optional list of data attributes already available
            
        Returns:
            Dictionary containing required attributes and their definitions
        """
        # Handle empty questions case
        if not questions:
            return {"attributes": []}
        
        # Format questions for prompt
        questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        
        # Format existing attributes if provided
        existing_text = ""
        if existing_attributes:
            existing_text = "\nExisting attributes:\n" + "\n".join(
                f"- {attr}" for attr in existing_attributes
            )
            
        prompt = f"""We need to determine what data attributes are required to answer these questions:
{questions_text}
{existing_text}

Return a JSON object with this structure:
{{
    "attributes": [
        {{
            "field_name": str,  # Database field name in snake_case
            "title": str,       # Human readable title
            "description": str, # Detailed description of the attribute
            "rationale": str    # Why this attribute is needed for the questions
        }}
    ]
}}

Ensure the response is a valid JSON object with an 'attributes' array containing the required attribute definitions."""

        response = await self._generate_content(
            prompt,
            expected_format={
                "attributes": [{
                    "field_name": str,
                    "title": str,
                    "description": str,
                    "rationale": str
                }]
            }
        )
        
        return response 