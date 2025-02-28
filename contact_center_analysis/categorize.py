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
        
        # Create a list of labels without the count information for the prompt
        label_list = [label for label, _ in labels]
        
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

IMPORTANT: Output your response as a JSON object with a single field named 'csv_data' containing the CSV data.
The CSV should have exactly these columns:
original_label,grouped_label

Example format:
{{
  "csv_data": "original_label,grouped_label\\n\\"cancel subscription\\",\\"Cancel Service\\"\\n\\"end my membership\\",\\"Cancel Service\\"\\n\\"billing help needed\\",\\"Billing Support\\""
}}
""".format('\n'.join(f"- {label}" for label in label_list), max_groups)

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
        batch_size: Optional[int] = None,
        max_groups: int = 100
    ) -> Dict[str, str]:
        """
        Process all labels in batches, maintaining consistent grouping.
        
        Args:
            value_distribution: List of tuples (value, count)
            batch_size: Optional number of labels to process in each batch. If not provided,
                       will use the default batch size of 200
            max_groups: Maximum number of consolidated groups to create (default=100)
            
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
            batch_mapping = await self.consolidate_labels(batch, max_groups=max_groups)
            
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

    async def process_labels_hierarchically(
        self,
        value_distribution: List[Tuple[str, int]],
        target_groups: int = 100,
        batch_size: Optional[int] = None,
        max_iterations: int = 3
    ) -> Dict[str, str]:
        """
        Process labels hierarchically to achieve target number of groups.
        
        Args:
            value_distribution: List of tuples (value, count)
            target_groups: Target number of consolidated groups (default=100)
            batch_size: Optional number of labels to process in each batch
            max_iterations: Maximum number of consolidation iterations (default=3)
            
        Returns:
            Dictionary mapping original labels to consolidated labels
        """
        # First pass - standard consolidation
        if self.debug:
            print(f"Starting hierarchical consolidation with {len(value_distribution)} labels")
            print(f"Target number of groups: {target_groups}")
        
        # Initial consolidation
        mapping = await self.process_labels_in_batches(
            value_distribution=value_distribution,
            batch_size=batch_size,
            max_groups=target_groups
        )
        
        # Get current unique groups
        unique_groups = set(mapping.values())
        current_group_count = len(unique_groups)
        
        if self.debug:
            print(f"First pass consolidation: {len(value_distribution)} → {current_group_count} groups")
        
        # If we're already at or below target, return
        if current_group_count <= target_groups:
            return mapping
        
        # Hierarchical consolidation - consolidate the consolidated groups
        iteration = 1
        while current_group_count > target_groups and iteration < max_iterations:
            # Create a new distribution of the consolidated groups with their counts
            group_counts = {}
            for original, group in mapping.items():
                # Find the original count for this label
                original_count = 0
                for label, count in value_distribution:
                    if label == original:
                        original_count = count
                        break
                
                if group in group_counts:
                    group_counts[group] += original_count
                else:
                    group_counts[group] = original_count
            
            # Convert to list of tuples for next consolidation
            group_distribution = [(group, count) for group, count in group_counts.items()]
            
            # Consolidate the groups further
            if self.debug:
                print(f"Iteration {iteration+1}: Consolidating {len(group_distribution)} groups")
            
            group_mapping = await self.consolidate_labels(
                labels=group_distribution,
                max_groups=target_groups
            )
            
            # Update the original mapping with the new hierarchy
            for original, group in mapping.items():
                if group in group_mapping:
                    mapping[original] = group_mapping[group]
            
            # Update current group count
            unique_groups = set(mapping.values())
            current_group_count = len(unique_groups)
            
            if self.debug:
                print(f"Iteration {iteration+1} consolidation: {len(group_distribution)} → {current_group_count} groups")
            
            iteration += 1
        
        return mapping

    async def process_labels_iteratively(
        self,
        value_distribution: List[Tuple[str, int]],
        target_groups: int = 100,
        batch_size: Optional[int] = None,
        max_iterations: int = 15,
        reduction_factor: float = 0.5,
        force_target: bool = False
    ) -> Dict[str, str]:
        """
        Process labels iteratively until target number of groups is reached.
        
        Args:
            value_distribution: List of tuples (value, count)
            target_groups: Target number of consolidated groups (default=100)
            batch_size: Optional number of labels to process in each batch
            max_iterations: Maximum number of iterations to attempt (default=15)
            reduction_factor: Target reduction per iteration (default=0.5)
            force_target: Boolean to force reaching the target number of groups
            
        Returns:
            Dictionary mapping original labels to consolidated labels
        """
        # Keep track of original labels
        original_labels = [label for label, _ in value_distribution]
        original_mapping = {label: label for label in original_labels}
        
        # Current working set
        current_distribution = value_distribution.copy()
        current_mapping = {label: label for label, _ in value_distribution}
        
        if self.debug:
            print(f"Starting iterative consolidation with {len(current_distribution)} labels")
            print(f"Target number of groups: {target_groups}")
        
        iteration = 0
        while iteration < max_iterations:
            # Calculate target groups for this iteration
            current_group_count = len(set(current_mapping.values()))
            
            if current_group_count <= target_groups:
                if self.debug:
                    print(f"Target reached: {current_group_count} groups")
                break
                
            # Calculate intermediate target for this iteration
            # Use a gradual approach to avoid over-consolidation
            intermediate_target = max(
                target_groups,
                int(current_group_count * reduction_factor)
            )
            
            if self.debug:
                print(f"\nIteration {iteration+1}: {current_group_count} groups → target {intermediate_target} groups")
            
            # Create a new distribution with current consolidated labels
            group_counts = {}
            for label, count in current_distribution:
                # Handle the case where a label might not be in the mapping
                if label in current_mapping:
                    group = current_mapping[label]
                else:
                    # If the label isn't in the mapping, use it as its own group
                    group = label
                    current_mapping[label] = label
                    
                if group in group_counts:
                    group_counts[group] += count
                else:
                    group_counts[group] = count
            
            # Convert to list of tuples for next consolidation
            next_distribution = [(group, count) for group, count in group_counts.items()]
            
            # Consolidate the current groups
            next_mapping = await self.consolidate_labels(
                labels=next_distribution,
                max_groups=intermediate_target
            )
            
            # Update the mapping chain
            new_mapping = {}
            for original, current in current_mapping.items():
                if current in next_mapping:
                    new_mapping[original] = next_mapping[current]
                else:
                    new_mapping[original] = current
            
            # Check if we made progress
            new_group_count = len(set(new_mapping.values()))
            if new_group_count >= current_group_count:
                if self.debug:
                    print(f"No further consolidation possible: {new_group_count} groups")
                break
            
            # Update for next iteration
            current_mapping = new_mapping
            current_distribution = next_distribution
            
            if self.debug:
                print(f"Iteration {iteration+1} result: {new_group_count} groups")
            
            iteration += 1
        
        # Create final mapping from original labels to final groups
        final_mapping = {}
        for label in original_labels:
            if label in current_mapping:
                final_mapping[label] = current_mapping[label]
            else:
                # If an original label somehow isn't in the current mapping, keep it as is
                final_mapping[label] = label
        
        # At the end of the method, if force_target is True and we haven't reached the target:
        if force_target and len(set(final_mapping.values())) > target_groups:
            if self.debug:
                print(f"Forcing consolidation to target {target_groups} groups...")
            
            # Get the current groups and their frequencies
            group_counts = {}
            for label in original_labels:
                group = final_mapping[label]
                # Find the original count for this label
                for orig_label, count in value_distribution:
                    if orig_label == label:
                        if group in group_counts:
                            group_counts[group] += count
                        else:
                            group_counts[group] = count
                        break
            
            # Sort groups by frequency (least frequent first)
            sorted_groups = sorted([(g, c) for g, c in group_counts.items()], key=lambda x: x[1])
            
            # Calculate how many groups we need to eliminate
            current_groups = len(set(final_mapping.values()))
            groups_to_eliminate = current_groups - target_groups
            
            if groups_to_eliminate > 0:
                # Take the least frequent groups
                groups_to_merge = [g for g, _ in sorted_groups[:groups_to_eliminate]]
                
                # Create a prompt to consolidate these specific groups
                prompt = """You are a label clustering expert. Your task is to consolidate these low-frequency labels into existing higher-frequency groups.

LOW FREQUENCY LABELS TO CONSOLIDATE:
{}

EXISTING HIGH FREQUENCY GROUPS (DO NOT MODIFY THESE):
{}

Rules:
1. Assign each low-frequency label to the most semantically similar high-frequency group
2. Every low-frequency label MUST be mapped to one of the existing high-frequency groups
3. Do not create new groups

IMPORTANT: Output your response as a JSON object with a single field named 'csv_data' containing the CSV data.
The CSV should have exactly these columns:
original_label,grouped_label

Example format:
{{
  "csv_data": "original_label,grouped_label\\n\\"niche feature request\\",\\"Feature Request\\"\\n\\"minor billing concern\\",\\"Billing Issue\\""
}}
""".format(
                    '\n'.join(f"- {g}" for g in groups_to_merge),
                    '\n'.join(f"- {g}" for g, _ in sorted_groups[groups_to_eliminate:])
                )
                
                try:
                    response = await self._generate_content(
                        prompt,
                        expected_format={
                            "csv_data": str
                        }
                    )
                    
                    # Parse the response
                    response_text = response["csv_data"]
                    if response_text.startswith('```'):
                        response_text = response_text.split('\n', 1)[1]
                        response_text = response_text.rsplit('\n', 1)[0]
                        response_text = response_text.replace('```csv\n', '').replace('```', '')
                    
                    # Update the mapping
                    for line in response_text.strip().split('\n'):
                        if ',' in line and not line.startswith('original_label'):
                            original, grouped = line.strip().split(',', 1)
                            original = original.strip('"').strip()
                            grouped = grouped.strip('"').strip()
                            
                            # Update all labels that were mapped to the original group
                            for label in original_labels:
                                if final_mapping[label] == original:
                                    final_mapping[label] = grouped
                
                except Exception as e:
                    if self.debug:
                        print(f"Error in forced consolidation: {e}")
        
        return final_mapping