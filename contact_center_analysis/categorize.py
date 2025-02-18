from typing import List, Dict, Any
from .base import BaseAnalyzer

class Categorizer(BaseAnalyzer):
    """Categorize and classify conversation elements."""
    
    async def categorize_intents(self, intents: List[Dict[str, str]], target_category: str, examples: List[str], batch_size: int = 200) -> Dict[str, bool]:
        """
        Categorize intents based on their similarity to provided examples.
        
        Args:
            intents: List of intent dictionaries with 'name' field
            target_category: Name of the category being classified (e.g. "cancellation", "billing", etc.)
            examples: List of example intents that represent the target category
            batch_size: Number of intents to process in each batch (default=200)
            
        Returns:
            Dictionary mapping intent names to boolean classification
        """
        # Format examples into bullet points
        example_bullets = "\n".join(f"- {example}" for example in examples)
        
        classification_task_description = f"""Analyze each intent text and classify if it represents a {target_category} request.

Use this JSON schema:
IntentClassification = {{'intent': str, 'is_match': bool}}
Return: list[IntentClassification]

Examples of {target_category} intents:
{example_bullets}

Classify the following intents:"""

        classified_intents = {}
        
        # Process intents in batches
        for i in range(0, len(intents), batch_size):
            batch = intents[i:i + batch_size]
            batch_prompt = classification_task_description + "\n\n"
            
            for intent in batch:
                intent_text = intent['name']
                batch_prompt += f"- {intent_text}\n"

            try:
                response = await self._generate_content(
                    batch_prompt,
                    expected_format={
                        "classifications": [{
                            "intent": str,
                            "is_match": bool
                        }]
                    }
                )
                
                for classification in response['classifications']:
                    intent_text = classification['intent']
                    is_match = classification['is_match']
                    classified_intents[intent_text] = is_match

            except Exception as e:
                print(f"Error processing batch: {e}")
                for intent in batch:
                    classified_intents[intent['name']] = False

        return classified_intents
    
    async def classify_sentiment(self,
                               conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify sentiment in conversations."""
        pass 