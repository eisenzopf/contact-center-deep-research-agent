from typing import List, Dict, Any, Optional
from .base import BaseAnalyzer

class Categorizer(BaseAnalyzer):
    """Categorize and classify conversation elements."""
    
    async def is_in_class(self, intents: List[Dict[str, str]], target_class: str, examples: Optional[List[str]] = None, batch_size: int = 200) -> Dict[str, bool]:
        """
        Determine if intents belong to a specified class based on optional examples.
        
        Args:
            intents: List of intent dictionaries with 'name' field
            target_class: Name of the class being classified (e.g. "cancellation", "billing", etc.)
            examples: Optional list of example intents that represent the target class
            batch_size: Number of intents to process in each batch (default=200)
            
        Returns:
            Dictionary mapping intent names to boolean classification
        """
        classification_task_description = f"""Analyze each intent text and classify if it represents a {target_class} request.

Use this JSON schema:
IntentClassification = {{"intent": str, "is_match": bool}}
Return: list[IntentClassification]"""

        if examples:
            example_bullets = "\n".join(f"- {example}" for example in examples)
            classification_task_description += f"\n\nExamples of {target_class} intents:\n{example_bullets}"

        classification_task_description += "\n\nClassify the following intents:"

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