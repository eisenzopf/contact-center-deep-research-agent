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

    async def resolve_data_gaps(
        self,
        analysis_results: Dict[str, Any],
        attribute_results: List[Dict[str, Any]],
        statistics: Dict[str, Any],
        questions: List[str],
        detailed_insights: Optional[Dict[str, Any]] = None,
        max_iterations: int = 1
    ) -> Dict[str, Any]:
        """
        Analyze data gaps and attempt to resolve them with additional analysis.
        
        Args:
            analysis_results: The original analysis results containing data gaps
            attribute_results: The raw attribute values extracted from conversations
            statistics: The compiled statistics on attribute values
            questions: The original research questions
            detailed_insights: Any detailed insights already generated
            max_iterations: Maximum number of refinement iterations to perform (default: 1)
            
        Returns:
            Enhanced analysis results with data gaps addressed where possible
        """
        current_analysis = analysis_results
        
        for iteration in range(max_iterations):
            if "data_gaps" not in current_analysis or not current_analysis["data_gaps"]:
                if self.debug:
                    print(f"No more data gaps to address after iteration {iteration+1}.")
                break
            
            data_gaps = current_analysis["data_gaps"]
            if self.debug:
                print(f"Iteration {iteration+1}: Analyzing {len(data_gaps)} identified data gaps...")
            
            # Steps 1-3: Analyze gaps, collect insights, combine with existing data
            gap_analysis = await self._analyze_data_gaps(data_gaps, attribute_results, statistics)
            enhanced_data = await self._collect_gap_insights(gap_analysis, attribute_results, statistics)
            
            # Track what we've already addressed to avoid redundant work
            if iteration == 0:
                cumulative_enhanced_data = enhanced_data
            else:
                # Merge new insights with previous ones
                for key, value in enhanced_data.items():
                    if key in cumulative_enhanced_data:
                        # For lists, append new items
                        if isinstance(value, list):
                            cumulative_enhanced_data[key].extend(value)
                        # For dictionaries, update with new values
                        elif isinstance(value, dict):
                            cumulative_enhanced_data[key].update(value)
                    else:
                        cumulative_enhanced_data[key] = value
            
            # Prepare enhanced data for analysis
            enhanced_formatted_data = {
                "attribute_statistics": statistics,
                "sample_size": sum(stats["total_values"] for stats in statistics.values()) // len(statistics) if statistics else 0,
                "gap_resolutions": cumulative_enhanced_data
            }
            
            if detailed_insights:
                enhanced_formatted_data["detailed_insights"] = detailed_insights
            
            # Step 4: Re-run analysis with enhanced data
            if self.debug:
                print(f"Re-running analysis with enhanced data (iteration {iteration+1})...")
            
            current_analysis = await self.analyze_findings(
                attribute_values=enhanced_formatted_data,
                questions=questions
            )
            
            # Check if we've made progress
            previous_gap_count = len(data_gaps)
            current_gap_count = len(current_analysis.get("data_gaps", []))
            
            if self.debug:
                if current_gap_count < previous_gap_count:
                    print(f"Resolved {previous_gap_count - current_gap_count} data gaps in iteration {iteration+1}.")
                else:
                    print(f"No additional data gaps resolved in iteration {iteration+1}.")
            
            # Stop early if no progress is being made
            if current_gap_count >= previous_gap_count and iteration > 0:
                if self.debug:
                    print("No improvement in data gaps. Stopping iterations.")
                break
        
        return current_analysis

    async def _analyze_data_gaps(
        self, 
        data_gaps: List[str], 
        attribute_results: List[Dict[str, Any]],
        statistics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Analyze each data gap to determine if and how it can be addressed.
        """
        prompt = f"""Analyze these data gaps identified in a conversation analysis and determine if they can be addressed 
with additional analysis of the existing data.

Data Gaps:
{json.dumps(data_gaps, indent=2)}

Available Data:
1. Raw attribute values from {len(attribute_results)} conversations
2. Statistical summaries for these attributes: {", ".join(statistics.keys())}

For each data gap, determine:
1. If it can be addressed with the existing data
2. What specific analysis would help address it
3. Which attributes would need to be examined more closely

Return a JSON array with this structure:
[
  {{
    "data_gap": str,                   # The original data gap description
    "addressable": bool,               # Whether this gap can be addressed with existing data
    "approach": str,                   # How to address this gap (if addressable)
    "required_attributes": [str],      # Which attributes to examine (if addressable)
    "analysis_type": str               # Type of analysis needed (e.g., "correlation", "examples", "patterns")
  }}
]
"""
        
        try:
            result = await self._generate_content(
                prompt,
                expected_format=[{
                    "data_gap": str,
                    "addressable": bool,
                    "approach": str,
                    "required_attributes": [str],
                    "analysis_type": str
                }]
            )
            
            if self.debug:
                addressable_gaps = sum(1 for gap in result if gap["addressable"])
                print(f"Found {addressable_gaps} addressable data gaps out of {len(result)}")
            
            return result
        
        except Exception as e:
            if self.debug:
                print(f"Error analyzing data gaps: {e}")
            # Return empty list if analysis fails
            return []

    async def _collect_gap_insights(
        self,
        gap_analysis: List[Dict[str, Any]],
        attribute_results: List[Dict[str, Any]],
        statistics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Collect additional insights to address identified data gaps.
        """
        enhanced_data = {}
        
        # Process only addressable gaps
        addressable_gaps = [gap for gap in gap_analysis if gap["addressable"]]
        
        for gap in addressable_gaps:
            gap_id = f"gap_{len(enhanced_data)}"
            
            if self.debug:
                print(f"Addressing data gap: {gap['data_gap'][:100]}...")
            
            # Collect relevant attribute values
            relevant_attributes = {}
            for attr_name in gap["required_attributes"]:
                if attr_name in statistics:
                    relevant_attributes[attr_name] = statistics[attr_name]
            
            # For gaps requiring examples, extract specific conversation examples
            if gap["analysis_type"] == "examples":
                examples = await self._extract_conversation_examples(
                    attribute_results, 
                    gap["required_attributes"],
                    max_examples=5
                )
                
                if examples:
                    relevant_attributes["examples"] = examples
            
            # For gaps requiring correlation analysis
            elif gap["analysis_type"] == "correlation":
                correlation = await self._analyze_attribute_correlation(
                    attribute_results,
                    gap["required_attributes"]
                )
                
                if correlation:
                    relevant_attributes["correlation"] = correlation
            
            # For gaps requiring pattern analysis
            elif gap["analysis_type"] == "patterns":
                patterns = await self._extract_attribute_patterns(
                    attribute_results,
                    gap["required_attributes"]
                )
                
                if patterns:
                    relevant_attributes["patterns"] = patterns
            
            # Add the enhanced data for this gap
            enhanced_data[gap_id] = {
                "data_gap": gap["data_gap"],
                "approach": gap["approach"],
                "insights": relevant_attributes
            }
        
        return enhanced_data

    async def _extract_conversation_examples(
        self,
        attribute_results: List[Dict[str, Any]],
        target_attributes: List[str],
        max_examples: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Extract specific conversation examples that illustrate the target attributes.
        """
        # Filter conversations that have values for all target attributes
        relevant_conversations = []
        
        for result in attribute_results:
            if "error" in result:
                continue
            
            # Check if this conversation has all target attributes
            attr_values = {attr["field_name"]: attr["value"] 
                          for attr in result["attribute_values"] 
                          if attr["confidence"] >= 0.6}
            
            if all(attr in attr_values for attr in target_attributes):
                relevant_conversations.append({
                    "conversation_id": result["conversation_id"],
                    "attributes": {attr: attr_values[attr] for attr in target_attributes if attr in attr_values}
                })
        
        # Select a diverse set of examples (up to max_examples)
        if not relevant_conversations:
            return []
        
        # If we have fewer examples than max_examples, return all of them
        if len(relevant_conversations) <= max_examples:
            return relevant_conversations
        
        # Otherwise, select a diverse set
        prompt = f"""Select {max_examples} diverse examples from these conversations that best illustrate 
the target attributes: {', '.join(target_attributes)}.

Conversations:
{json.dumps(relevant_conversations, indent=2)}

Return the conversation_ids of the {max_examples} most representative examples as a JSON array.
"""
        
        try:
            selected_ids = await self._generate_content(
                prompt,
                expected_format=[str]
            )
            
            # Return the selected conversations
            return [conv for conv in relevant_conversations 
                    if conv["conversation_id"] in selected_ids[:max_examples]]
        
        except Exception as e:
            if self.debug:
                print(f"Error selecting example conversations: {e}")
            # Fall back to random selection
            import random
            return random.sample(relevant_conversations, min(max_examples, len(relevant_conversations)))

    async def _analyze_attribute_correlation(
        self,
        attribute_results: List[Dict[str, Any]],
        target_attributes: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Analyze correlations between target attributes.
        """
        # Extract attribute value pairs
        attribute_pairs = []
        
        for result in attribute_results:
            if "error" in result:
                continue
            
            # Get values for target attributes
            attr_values = {attr["field_name"]: attr["value"] 
                          for attr in result["attribute_values"] 
                          if attr["field_name"] in target_attributes and attr["confidence"] >= 0.6}
            
            # Only include if we have values for all target attributes
            if len(attr_values) == len(target_attributes):
                attribute_pairs.append(attr_values)
        
        if not attribute_pairs:
            return []
        
        # Analyze correlations
        prompt = f"""Analyze the correlation between these attributes: {', '.join(target_attributes)}.

Data (attribute value pairs from {len(attribute_pairs)} conversations):
{json.dumps(attribute_pairs[:100], indent=2)}

Identify patterns, correlations, or relationships between these attributes.
Return your analysis as a JSON array of insights, each with a title and description.
"""
        
        try:
            return await self._generate_content(
                prompt,
                expected_format=[{"title": str, "description": str}]
            )
        except Exception as e:
            if self.debug:
                print(f"Error analyzing attribute correlations: {e}")
            return []

    async def _extract_attribute_patterns(
        self,
        attribute_results: List[Dict[str, Any]],
        target_attributes: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Extract patterns within specific attributes.
        """
        # Collect all values for each target attribute
        attribute_values = {attr: [] for attr in target_attributes}
        
        for result in attribute_results:
            if "error" in result:
                continue
            
            for attr_value in result["attribute_values"]:
                if (attr_value["field_name"] in target_attributes and 
                    attr_value["confidence"] >= 0.6):
                    attribute_values[attr_value["field_name"]].append(attr_value["value"])
        
        # Check if we have enough values
        valid_attributes = {attr: values for attr, values in attribute_values.items() if values}
        
        if not valid_attributes:
            return []
        
        # Analyze patterns
        prompt = f"""Analyze patterns within these attribute values:

{json.dumps({attr: values[:100] for attr, values in valid_attributes.items()}, indent=2)}

Identify recurring patterns, themes, or structures within each attribute.
Return your analysis as a JSON array of insights, each with a title and description.
"""
        
        try:
            return await self._generate_content(
                prompt,
                expected_format=[{"title": str, "description": str}]
            )
        except Exception as e:
            if self.debug:
                print(f"Error extracting attribute patterns: {e}")
            return []
