from typing import List, Dict, Any, Optional, Tuple
from .base import BaseAnalyzer
import asyncio

class TextGenerator(BaseAnalyzer):
    """Generate and analyze text content and data attributes."""
    
    def __init__(self, api_key: str, debug: bool = False):
        super().__init__(api_key, debug)
        
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
            
        # Split into smaller chunks for parallel processing
        chunk_size = 3  # Process 3 questions at a time
        question_chunks = [questions[i:i + chunk_size] 
                          for i in range(0, len(questions), chunk_size)]
        
        # Create tasks for parallel processing
        tasks = []
        for chunk in question_chunks:
            chunk_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(chunk))
            prompt = f"""We need to determine what data attributes are required to answer these questions:
{chunk_text}
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
}}"""
            tasks.append(self._generate_content(
                prompt,
                expected_format={
                    "attributes": [{
                        "field_name": str,
                        "title": str,
                        "description": str,
                        "rationale": str
                    }]
                }
            ))
        
        # Process chunks in parallel
        chunk_results = await asyncio.gather(*tasks)
        
        # Combine results
        all_attributes = []
        for result in chunk_results:
            all_attributes.extend(result["attributes"])
        
        return {"attributes": all_attributes}

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

    async def generate_labeled_attribute(
        self,
        text: str,
        attribute: Dict[str, str],
        create_label: bool = True
    ) -> Dict[str, Any]:
        """Generate attribute value with optional label generation."""
        
        prompt = f"""Analyze this conversation text and extract the value for the following attribute:

Attribute: {attribute['title']}
Description: {attribute['description']}

Text:
{text}

Return as JSON with:
- value: extracted value
- confidence: float between 0-1
"""

        if create_label:
            prompt += """
- label: short categorical label for the value"""

        expected_format = {
            "value": str,
            "confidence": float
        }
        if create_label:
            expected_format["label"] = str

        response = await self._generate_content(
            prompt,
            expected_format=expected_format
        )
        
        return {
            "field_name": attribute["field_name"],
            "value": response.get("value", ""),
            "confidence": response.get("confidence", 0.0),
            **({"label": response.get("label")} if create_label else {})
        }

    async def _process_single_conversation(self, conversation: Dict[str, str], required_attributes: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process a single conversation to generate attribute values."""
        
        conv_text = conversation.get('text', '')
        conv_id = conversation.get('id', '')
        
        # Generate values for all attributes in parallel
        tasks = [
            self.generate_labeled_attribute(
                text=conv_text,
                attribute=attr
            )
            for attr in required_attributes
        ]
        attribute_values = await asyncio.gather(*tasks)
        
        return {
            "conversation_id": conv_id,
            "attribute_values": attribute_values
        }

    async def generate_attributes_batch(
        self,
        conversations: List[Dict[str, str]],
        required_attributes: List[Dict[str, str]],
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate attribute values for multiple conversations in batches."""
        
        tasks = [
            self._process_single_conversation(conv, required_attributes)
            for conv in conversations
        ]
        
        # Process all conversations in parallel
        return await asyncio.gather(*tasks)

    async def generate_required_attributes_batch(
        self,
        questions_sets: List[List[str]],
        existing_attributes: Optional[List[str]] = None,
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate required attributes for multiple sets of questions in parallel.
        
        Args:
            questions_sets: List of question sets to analyze
            existing_attributes: Optional list of data attributes already available
            batch_size: Optional number of sets to process in each batch
            
        Returns:
            List of dictionaries containing required attributes for each set
        """
        async def process_questions(questions):
            return await self.generate_required_attributes(
                questions=questions,
                existing_attributes=existing_attributes
            )
            
        return await self.process_in_batches(
            questions_sets,
            batch_size=batch_size,
            process_func=process_questions
        ) 