from typing import List, Dict, Any, Tuple, Optional
from .base import BaseAnalyzer
import json

class AttributeMatcher(BaseAnalyzer):
    """Match and compare attributes using semantic similarity."""
    
    async def find_matches(self,
                          required_attributes: List[Dict[str, Any]],
                          available_attributes: List[Dict[str, Any]],
                          batch_size: int = 200,
                          confidence_threshold: float = 0.7) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        """
        Match required attributes against available attributes to identify matches and gaps.
        
        Args:
            required_attributes: List of RequiredAttribute objects from text.py
            available_attributes: List of existing database attributes
            batch_size: Number of comparisons to process in each batch
            confidence_threshold: Minimum confidence score to consider a match
        
        Returns:
            tuple(dict, list): (matches mapping required->available, missing attributes)
        """
        matches = {}
        missing_attributes = []

        # Process in batches to avoid overwhelming the LLM
        for i in range(0, len(required_attributes), batch_size):
            batch = required_attributes[i:i + batch_size]
            
            prompt = """Compare required attributes against available attributes to identify matches.

Required Attributes:"""
            
            # Add required attributes to prompt
            for attr in batch:
                prompt += f"""

{attr['field_name']}:
- Title: {attr['title']}
- Description: {attr['description']}"""

            prompt += """

Available Database Attributes:"""

            # Add available attributes to prompt
            for attr in available_attributes:
                description = json.loads(attr['description'])
                prompt += f"""

{attr['name']}:
- Title: {description['title']}
- Description: {description['description']}"""

            prompt += """

Return a JSON object with this structure (ensure no trailing commas):
{
    "matches": [
        {
            "required_field": str,         # Required attribute's field_name
            "matching_attributes": [{       # List of matching available attributes
                "available_field": str,     # Available attribute's name
                "confidence": float         # Match confidence score 0-1
            }]
        }
    ]
}

For each required attribute, identify if any available attributes capture the same information.
Consider:
1. Field names and titles
2. Descriptions and purposes
3. Semantic meaning
4. Type of information captured

Note: Ensure the JSON response is properly formatted with no trailing commas."""

            try:
                response = await self._generate_content(
                    prompt,
                    expected_format={
                        "matches": [{
                            "required_field": str,
                            "matching_attributes": [{
                                "available_field": str,
                                "confidence": float
                            }]
                        }]
                    }
                )

                # Process matches
                for match in response["matches"]:
                    required_field = match['required_field']
                    best_match = None
                    best_confidence = 0

                    # Find best matching attribute
                    for attr_match in match['matching_attributes']:
                        if attr_match['confidence'] > best_confidence:
                            best_confidence = attr_match['confidence']
                            best_match = attr_match['available_field']

                    if best_match and best_confidence > confidence_threshold:
                        matches[required_field] = best_match
                    else:
                        missing_attributes.append(next(
                            attr for attr in batch 
                            if attr['field_name'] == required_field
                        ))

            except Exception as e:
                if self.debug:
                    print(f"Error processing batch: {e}")
                # If error, consider all attributes in batch as missing
                missing_attributes.extend(batch)

        return matches, missing_attributes

    async def group_by_similarity(self,
                                items: List[str],
                                min_similarity: float = 0.8) -> List[List[str]]:
        """Group items by their semantic similarity.

        Args:
            items: List of strings to group
            min_similarity: Minimum similarity score to group items together

        Returns:
            List of groups (each group is a list of similar items)
        """
        prompt = f"""Group these items based on semantic similarity.
        
Items:
{self._format_candidates(items)}

Create groups where items have at least {min_similarity} similarity score.
Return JSON with:
- groups: List of groups
- rationale: Explanation for each group

Each group should have:
- items: List of items in the group
- theme: Short description of the common theme"""

        response = await self._generate_content(
            prompt,
            expected_format={
                "groups": [{
                    "items": [str],
                    "theme": str,
                    "rationale": str
                }]
            }
        )
        
        return response["groups"]

    def _format_candidates(self, items: List[str]) -> str:
        """Format list of items for prompt."""
        return "\n".join(f"- {item}" for item in items) 