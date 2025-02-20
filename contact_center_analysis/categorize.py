from typing import List, Dict, Any, Optional, Tuple
from .base import BaseAnalyzer
import math
import asyncio

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

    async def consolidate_labels(
        self,
        labels: List[Tuple[str, int]],
        max_groups: int = 100
    ) -> Dict[str, str]:
        """
        Consolidate similar labels into semantic groups.
        
        Args:
            labels: List of tuples containing (label, count)
            max_groups: Maximum number of consolidated groups to create
            
        Returns:
            Dictionary mapping original labels to consolidated labels
        """
        consolidated_mapping = {}
        
        prompt = """You are a label clustering expert. Your task is to consolidate similar labels into semantic groups.

INPUT LABELS TO PROCESS:
{}

Rules:
1. Group similar labels together under a common, descriptive label
2. Maintain semantic meaning
3. Use consistent labeling style
4. Use Title Case
5. Focus on accuracy over reducing groups
6. Maximum number of groups: {}

IMPORTANT: Output your response in valid CSV format with exactly these columns:
original_label,grouped_label

Example format:
original_label,grouped_label
"cancel subscription","Cancel Service"
"end my membership","Cancel Service"
"billing help needed","Billing Support"
""".format('\n'.join(f"- {label} ({count})" for label, count in labels), max_groups)

        try:
            response = await self._generate_content(
                prompt,
                expected_format={
                    "csv_data": str
                }
            )
            
            # Clean and parse response
            response_text = response["csv_data"]
            if response_text.startswith('```'):
                response_text = response_text.split('\n', 1)[1]
                response_text = response_text.rsplit('\n', 1)[0]
                response_text = response_text.replace('```csv\n', '').replace('```', '')
            
            # Parse CSV response
            for line in response_text.strip().split('\n'):
                if ',' in line and not line.startswith('original_label'):
                    original, grouped = line.strip().split(',', 1)
                    original = original.strip('"').strip()
                    grouped = grouped.strip('"').strip()
                    consolidated_mapping[original] = grouped
                    
        except Exception as e:
            if self.debug:
                print(f"Error consolidating labels: {e}")
            # If error, keep original labels
            for label, _ in labels:
                consolidated_mapping[label] = label
        
        return consolidated_mapping

    async def process_labels_in_batches(
        self,
        value_distribution: List[Tuple[str, int]],
        batch_size: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Process all labels in batches, maintaining consistent grouping.
        
        Args:
            value_distribution: List of tuples (value, count)
            batch_size: Optional number of labels to process in each batch. If not provided,
                       will use the default batch size of 200
            
        Returns:
            Dictionary mapping original labels to consolidated labels
        """
        final_mapping = {}
        
        # Use default batch size if none provided
        effective_batch_size = batch_size or 200
        
        # Process in batches
        for i in range(0, len(value_distribution), effective_batch_size):
            batch = value_distribution[i:i + effective_batch_size]
            if self.debug:
                print(f"\nProcessing batch {i//effective_batch_size + 1}/{math.ceil(len(value_distribution)/effective_batch_size)}")
            
            # Get consolidated labels for this batch
            batch_mapping = await self.consolidate_labels(batch)
            
            # Update mapping
            final_mapping.update(batch_mapping)
            
            if self.debug:
                print(f"Processed {len(batch)} labels")
                print(f"Total unique group labels: {len(set(batch_mapping.values()))}")
        
        return final_mapping

    async def is_in_class_batch(
        self,
        intents: List[Dict[str, str]],
        target_class: str,
        examples: Optional[List[str]] = None,
        batch_size: Optional[int] = None
    ) -> List[Dict[str, bool]]:
        """Process multiple intents in batches efficiently."""
        
        # Use default batch size if none provided
        effective_batch_size = batch_size or 50  # Smaller default batch size for parallel processing
        
        # Process intents in parallel batches
        all_results = []
        tasks = []
        
        for i in range(0, len(intents), effective_batch_size):
            batch = intents[i:i + effective_batch_size]
            task = self.is_in_class(
                intents=batch,
                target_class=target_class,
                examples=examples,
                batch_size=effective_batch_size
            )
            tasks.append(task)
        
        # Execute batches in parallel
        batch_results = await asyncio.gather(*tasks)
        
        # Convert dictionary results to list in correct order
        for i, intent in enumerate(intents):
            batch_index = i // effective_batch_size
            result = batch_results[batch_index].get(intent['name'], False)
            all_results.append({intent['name']: result})
        
        return all_results