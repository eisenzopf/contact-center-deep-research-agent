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

    async def generate_attribute(
        self,
        text: str,
        attribute: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Generate a value for a specific attribute based on text input.
        
        Args:
            text: The input text to analyze
            attribute: Dictionary containing attribute definition with fields:
                      - field_name: attribute name
                      - title: Human readable title
                      - description: Detailed description
                      - rationale: Why this attribute is needed
                      
        Returns:
            Dictionary containing:
            - value: Generated value for the attribute
            - confidence: Confidence score (0-1)
            - explanation: Explanation of how the value was determined
        """
        # Handle empty text case explicitly
        if not text.strip():
            return {
                "value": "No content",
                "confidence": 0.0,
                "explanation": "No content was provided for analysis. The text input was empty."
            }

        prompt = f"""Analyze this text to determine the value for the following attribute:

Attribute: {attribute['title']}
Description: {attribute['description']}

Text to analyze:
{text}

Return a JSON object with this structure:
{{
    "value": str,           # The extracted or determined value
    "confidence": float,    # Confidence score between 0 and 1
    "explanation": str      # Explanation of how the value was determined
}}

Ensure the response is specific to the attribute definition and supported by the text content."""

        response = await self._generate_content(
            prompt,
            expected_format={
                "value": str,
                "confidence": float,
                "explanation": str
            }
        )
        
        return response 