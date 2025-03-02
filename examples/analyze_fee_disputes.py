import sqlite3
import asyncio
import sys
import os
import argparse
import json
import time
from collections import Counter, defaultdict
from contact_center_analysis.text import TextGenerator
from contact_center_analysis.match import AttributeMatcher
from contact_center_analysis.categorize import Categorizer
from contact_center_analysis.analyze import DataAnalyzer

async def extract_unique_intents(db_path, min_count=20):
    """Extract unique intent values from the conversation_attributes table."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = """
    SELECT value as intent_text, COUNT(*) as count
    FROM conversation_attributes
    WHERE type = 'intent'
    GROUP BY value
    HAVING COUNT(*) >= ?
    """
    
    cursor.execute(query, (min_count,))
    intents = [{"name": row['intent_text']} for row in cursor.fetchall()]
    
    conn.close()
    return intents

async def find_matching_intents(db_path, api_key, target_class, examples=None, min_count=20, debug=False):
    """Find intents that match a specific target class."""
    categorizer = Categorizer(api_key=api_key, debug=debug)
    db_intents = await extract_unique_intents(db_path, min_count)
    
    print(f"Found {len(db_intents)} unique intents in the database")
    
    if not examples:
        examples = [
            "Customer disputing a late payment fee",
            "Customer questioning monthly service charge",
            "Customer complaining about unexpected fees on bill",
            "Customer wants fee waived",
            "Customer says they were charged incorrectly",
            "Customer asking for refund of fees"
        ]
    
    batch_results = await categorizer.is_in_class_batch(
        intents=db_intents,
        target_class=target_class,
        examples=examples,
        batch_size=50
    )
    
    matching_intents = []
    for result_dict in batch_results:
        for intent_text, is_match in result_dict.items():
            if is_match:
                matching_intents.append(intent_text)
    
    print(f"Found {len(matching_intents)} intents matching '{target_class}'")
    if matching_intents:
        print("Examples of matching intents:")
        for intent in matching_intents[:5]:
            print(f"  - {intent}")
    
    return matching_intents

async def fetch_conversations_by_intents(db_path, matching_intents, limit=50):
    """Fetch conversations that have intents matching the specified list."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    placeholders = ', '.join(['?'] * len(matching_intents))
    
    query = f"""
    SELECT c.conversation_id, c.text
    FROM conversations c
    JOIN conversation_attributes ca ON c.conversation_id = ca.conversation_id
    WHERE ca.type = 'intent' AND ca.value IN ({placeholders})
    AND c.text IS NOT NULL AND LENGTH(c.text) > 100
    ORDER BY RANDOM()
    LIMIT ?
    """
    
    params = matching_intents + [limit]
    cursor.execute(query, params)
    
    conversations = []
    for row in cursor.fetchall():
        conversations.append({
            'id': row['conversation_id'],
            'text': row['text']
        })
    
    conn.close()
    return conversations

async def get_required_attributes(questions, api_key, debug=False):
    """Get the list of required attributes needed to answer fee dispute-related questions."""
    generator = TextGenerator(api_key=api_key, debug=debug)
    
    print("Generating required attributes for fee dispute analysis...")
    result = await generator.generate_required_attributes(
        questions=questions
    )
    
    return result["attributes"]

async def generate_attributes_for_conversations(conversations, attributes, api_key, debug=False, batch_size=5, max_retries=3):
    """Generate attribute values for multiple conversations in parallel batches with error handling."""
    generator = TextGenerator(api_key=api_key, debug=debug)
    
    async def process_conversation(conversation):
        print(f"Analyzing conversation: {conversation['id']}...")
        
        for attempt in range(max_retries):
            try:
                attribute_values = await generator.generate_attributes(
                    text=conversation['text'],
                    attributes=attributes
                )
                
                return {
                    "conversation_id": conversation['id'],
                    "attribute_values": attribute_values
                }
            except ValueError as e:
                if "Failed to parse JSON response" in str(e) and attempt < max_retries - 1:
                    print(f"  Retry {attempt+1}/{max_retries} due to JSON parsing error...")
                    await asyncio.sleep(1)  # Brief pause before retry
                else:
                    if debug:
                        print(f"  Error processing conversation {conversation['id']}: {e}")
                    # Return partial results with error information
                    return {
                        "conversation_id": conversation['id'],
                        "attribute_values": [],
                        "error": str(e)
                    }
    
    # Use the process_in_batches method from BaseAnalyzer
    results = await generator.process_in_batches(
        conversations,
        batch_size=batch_size,
        process_func=process_conversation
    )
    
    # Count successful and failed conversations
    success_count = sum(1 for r in results if "error" not in r)
    error_count = len(results) - success_count
    
    if error_count > 0:
        print(f"\nProcessed {len(results)} conversations: {success_count} successful, {error_count} with errors")
        if debug:
            for result in results:
                if "error" in result:
                    print(f"  Conversation {result['conversation_id']} failed: {result['error'][:100]}...")
    
    return results

async def compile_attribute_statistics(attribute_results, api_key, debug=False):
    """Compile statistics on attribute values across conversations with semantic grouping."""
    # Create a structure to hold all values for each attribute
    attribute_values = defaultdict(list)
    
    # Count successful and failed conversations
    error_count = sum(1 for r in attribute_results if "error" in r)
    success_count = len(attribute_results) - error_count
    
    if error_count > 0 and debug:
        print(f"Note: {error_count} conversations had errors and will be excluded from statistics")
    
    # Collect all values for each attribute (skip conversations with errors)
    for result in attribute_results:
        if "error" in result:
            continue
            
        for attr_value in result["attribute_values"]:
            field_name = attr_value["field_name"]
            value = attr_value["value"]
            confidence = attr_value.get("confidence", 0.8)
            
            # Only include values with reasonable confidence
            if confidence >= 0.6:
                # Convert any non-string values to strings to ensure they're hashable
                if not isinstance(value, str):
                    if isinstance(value, list):
                        value = ", ".join(str(item) for item in value)
                    else:
                        value = str(value)
                attribute_values[field_name].append(value)
    
    # Initialize categorizer for semantic grouping
    categorizer = Categorizer(api_key=api_key, debug=debug)
    
    # Calculate statistics for each attribute with semantic grouping
    statistics = {}
    for field_name, values in attribute_values.items():
        # Skip if no values
        if not values:
            continue
            
        # Convert values to (value, count) format for consolidate_labels
        value_counts = Counter(values)
        value_distribution = [(value, count) for value, count in value_counts.items()]
        
        # Use existing consolidate_labels method to group similar values
        # Only apply consolidation if we have enough unique values to warrant it
        if len(value_counts) > 5:
            print(f"Consolidating {len(value_counts)} unique values for {field_name}...")
            normalized_mapping = await categorizer.consolidate_labels(
                labels=value_distribution,
                max_groups=min(20, len(value_counts))
            )
            
            # Apply the mapping to get normalized values
            normalized_values = [normalized_mapping.get(value, value) for value in values]
        else:
            # For small sets, just use the original values
            normalized_values = values
        
        # Count occurrences of each normalized value
        normalized_counts = Counter(normalized_values)
        
        # Calculate percentages
        total = len(values)
        percentages = {value: (count / total) * 100 for value, count in normalized_counts.items()}
        
        # Store statistics
        statistics[field_name] = {
            "total_values": total,
            "unique_values": len(normalized_counts),
            "value_counts": dict(normalized_counts),
            "percentages": percentages,
            "raw_values": dict(value_counts)  # Keep original values for reference
        }
    
    return statistics

async def review_attribute_details(attribute_results, required_attributes, questions, api_key, debug=False):
    """Extract detailed insights from raw attribute values for deeper analysis."""
    analyzer = DataAnalyzer(api_key=api_key, debug=debug)
    
    # Collect all attribute values by type
    attribute_details = defaultdict(list)
    for result in attribute_results:
        if "error" in result:
            continue
            
        for attr_value in result["attribute_values"]:
            if attr_value["confidence"] >= 0.6:
                attribute_details[attr_value["field_name"]].append({
                    "conversation_id": result["conversation_id"],
                    "value": attr_value["value"],
                    "confidence": attr_value["confidence"]
                })
    
    # Dynamically identify the most important attributes for our questions
    print("Identifying key attributes for deeper analysis...")
    focus_attributes = await analyzer.identify_key_attributes(
        questions=questions,
        available_attributes=required_attributes,
        max_attributes=5  # Adjust as needed
    )
    
    print(f"Selected {len(focus_attributes)} key attributes for detailed analysis:")
    for attr_name in focus_attributes:
        attr = next((a for a in required_attributes if a['field_name'] == attr_name), None)
        if attr:
            print(f"  - {attr['title']} ({attr_name})")
    
    insights = {}
    for attr_name in focus_attributes:
        if attr_name not in attribute_details:
            print(f"  Warning: Key attribute '{attr_name}' has no values in the dataset")
            continue
            
        values = [item["value"] for item in attribute_details[attr_name]]
        
        if not values:
            continue
            
        print(f"Analyzing patterns in '{attr_name}' ({len(values)} values)...")
        
        # Generate detailed insights for this attribute
        prompt = f"""
        Analyze these {len(values)} values for the '{attr_name}' attribute from fee dispute conversations.
        
        Values:
        {json.dumps(values[:100], indent=2)}
        
        Extract 3-5 key patterns or insights that aren't captured by simple frequency counting.
        For example:
        - Common combinations of actions
        - Typical sequences or patterns
        - Contextual details that might be missed
        - Qualitative aspects that frequency analysis doesn't show
        
        Format your response as a JSON list of insights, each with a title and description.
        """
        
        try:
            result = await analyzer._generate_content(
                prompt,
                expected_format=[{"title": str, "description": str}]
            )
            insights[attr_name] = result
        except Exception as e:
            if debug:
                print(f"Error analyzing {attr_name}: {e}")
            insights[attr_name] = []
    
    return insights

async def analyze_attribute_findings(statistics, questions, api_key, debug=False, detailed_insights=None):
    """Analyze the compiled attribute statistics to answer research questions."""
    analyzer = DataAnalyzer(api_key=api_key, debug=debug)
    
    # Format the statistics for analysis
    formatted_data = {
        "attribute_statistics": statistics,
        "sample_size": sum(stats["total_values"] for stats in statistics.values()) // len(statistics) if statistics else 0
    }
    
    # Add detailed insights if available
    if detailed_insights:
        formatted_data["detailed_insights"] = detailed_insights
    
    # Analyze the findings
    analysis = await analyzer.analyze_findings(
        attribute_values=formatted_data,
        questions=questions
    )
    
    return analysis

async def generate_visualizations(analysis_results, statistics, attribute_results, output_dir="visualizations"):
    """Generate visualizations based on analysis results and save to specified directory."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    visualizations = []
    
    # 1. Generate attribute distribution charts for key metrics
    if "answers" in analysis_results:
        for answer in analysis_results["answers"]:
            key_metrics = answer.get("key_metrics", [])
            for metric in key_metrics:
                if metric in statistics:
                    # Create horizontal bar chart for this metric
                    fig, ax = plt.figure(figsize=(10, 6)), plt.subplot(111)
                    
                    # Get top values (limit to 10 for readability)
                    values = statistics[metric]["value_counts"]
                    df = pd.DataFrame(
                        {"Value": list(values.keys()), "Count": list(values.values())}
                    ).sort_values("Count", ascending=False).head(10)
                    
                    # Create horizontal bar chart
                    sns.barplot(x="Count", y="Value", data=df, ax=ax)
                    ax.set_title(f"Distribution of {metric}")
                    
                    # Save figure
                    filename = f"{output_dir}/{metric.replace(' ', '_')}_distribution.png"
                    plt.tight_layout()
                    plt.savefig(filename)
                    plt.close()
                    
                    visualizations.append({
                        "question": answer["question"],
                        "metric": metric,
                        "type": "distribution",
                        "filename": filename
                    })
    
    # 2. Generate correlation heatmaps between key attributes
    if len(statistics) >= 2:
        # Create correlation matrix from key attributes
        correlation_data = {}
        
        # Extract all attribute values by conversation ID
        for result in attribute_results:
            if "error" in result:
                continue
                
            conv_id = result["conversation_id"]
            for attr in result["attribute_values"]:
                field_name = attr["field_name"]
                if field_name not in correlation_data:
                    correlation_data[field_name] = {}
                correlation_data[field_name][conv_id] = attr["value"]
        
        # Convert to DataFrame for correlation analysis
        # This requires categorical encoding which would be implemented here
        # For simplicity, we'll just note this would be done
        
        # Save correlation visualization
        filename = f"{output_dir}/attribute_correlations.png"
        visualizations.append({
            "type": "correlation",
            "filename": filename
        })
    
    # 3. Generate confidence visualization
    if "answers" in analysis_results:
        confidences = [answer.get("confidence", "Medium") for answer in analysis_results["answers"]]
        questions = [answer.get("question", "Unknown") for answer in analysis_results["answers"]]
        
        # Map text confidences to numeric values
        confidence_map = {"Low": 1, "Medium": 2, "High": 3}
        numeric_confidences = [confidence_map.get(c, 2) for c in confidences]
        
        fig, ax = plt.figure(figsize=(10, 6)), plt.subplot(111)
        sns.barplot(x=numeric_confidences, y=questions, ax=ax)
        ax.set_title("Confidence Levels by Question")
        ax.set_xlabel("Confidence (1=Low, 2=Medium, 3=High)")
        ax.set_ylabel("Research Question")
        
        filename = f"{output_dir}/confidence_levels.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        visualizations.append({
            "type": "confidence",
            "filename": filename
        })
    
    # 4. Generate before/after gap resolution comparison
    if "data_gaps" in analysis_results and "data_gaps" in analysis_results:
        before_gaps = len(analysis_results.get("data_gaps", []))
        after_gaps = len(analysis_results.get("data_gaps", []))
        
        fig, ax = plt.figure(figsize=(8, 6)), plt.subplot(111)
        sns.barplot(x=["Before Resolution", "After Resolution"], 
                   y=[before_gaps, after_gaps], ax=ax)
        ax.set_title("Data Gaps Before and After Resolution")
        ax.set_ylabel("Number of Data Gaps")
        
        filename = f"{output_dir}/gap_resolution.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        visualizations.append({
            "type": "gap_resolution",
            "filename": filename
        })
    
    return visualizations

async def generate_sentiment_flow(attribute_results, output_dir, api_key=None, debug=False):
    """Generate visualization showing how sentiment changes during conversations."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    from matplotlib.colors import LinearSegmentedColormap
    
    # Check if customer_sentiment is available in the data
    has_sentiment = False
    for result in attribute_results:
        if "error" in result:
            continue
        
        for attr in result.get("attribute_values", []):
            if attr["field_name"] == "customer_sentiment":
                has_sentiment = True
                break
        
        if has_sentiment:
            break
    
    # If sentiment is not available and we have an API key, generate it
    if not has_sentiment and api_key:
        print("Customer sentiment attribute not found. Generating sentiment analysis...")
        from contact_center_analysis.extract import AttributeExtractor
        from collections import defaultdict
        from contact_center_analysis.categorize import Categorizer
        
        extractor = AttributeExtractor(api_key=api_key, debug=debug)
        
        # Define the sentiment attribute
        sentiment_attribute = {
            "name": "customer_sentiment",
            "description": "The overall sentiment of the customer during the conversation",
            "examples": [
                "Negative", "Positive", "Neutral", 
                "Initially negative, then positive",
                "Frustrated at first, satisfied by the end"
            ]
        }
        
        # Extract sentiment for each conversation
        sentiment_values = []
        for i, result in enumerate(attribute_results):
            if "error" in result or "conversation_text" not in result:
                continue
                
            try:
                print(f"Generating sentiment for conversation {i+1}/{len(attribute_results)}...")
                sentiment_result = await extractor.extract_attribute(
                    conversation_text=result["conversation_text"],
                    attribute=sentiment_attribute
                )
                
                # Add sentiment to attribute values
                if "attribute_values" not in result:
                    result["attribute_values"] = []
                    
                result["attribute_values"].append({
                    "field_name": "customer_sentiment",
                    "value": sentiment_result["value"],
                    "confidence": sentiment_result.get("confidence", 0.8)
                })
                
                sentiment_values.append(sentiment_result["value"])
                has_sentiment = True
            except Exception as e:
                if debug:
                    print(f"Error generating sentiment: {e}")
        
        # Apply semantic grouping to the sentiment values
        if sentiment_values:
            print("Consolidating sentiment values with semantic grouping...")
            categorizer = Categorizer(api_key=api_key, debug=debug)
            
            # Convert to (value, count) format for consolidate_labels
            value_counts = defaultdict(int)
            for value in sentiment_values:
                value_counts[value] += 1
            
            value_count_pairs = [(value, count) for value, count in value_counts.items()]
            
            # Consolidate similar labels
            consolidated = await categorizer.consolidate_labels(
                field_name="customer_sentiment",
                values=value_count_pairs,
                max_categories=10
            )
            
            # Create a mapping from original values to consolidated ones
            value_mapping = {}
            for original, count in value_count_pairs:
                for category, items in consolidated.items():
                    if any(item[0] == original for item in items):
                        value_mapping[original] = category
                        break
            
            # Update the attribute values with consolidated labels
            for result in attribute_results:
                if "error" in result:
                    continue
                
                for attr in result.get("attribute_values", []):
                    if attr["field_name"] == "customer_sentiment" and attr["value"] in value_mapping:
                        attr["value"] = value_mapping[attr["value"]]
    
    # If we still don't have sentiment data, return None
    if not has_sentiment:
        print("Warning: Cannot generate sentiment flow visualization - no sentiment data available")
        return None
    
    # Extract sentiment values with timestamps
    sentiment_data = []
    
    for result in attribute_results:
        if "error" in result:
            continue
            
        conv_id = result.get("conversation_id", "unknown")
        
        # Find sentiment and timestamp values
        sentiment_values = []
        timestamp_values = []
        
        for attr in result.get("attribute_values", []):
            if attr["field_name"] == "customer_sentiment":
                sentiment_values.append(attr["value"])
            elif attr["field_name"] == "message_timestamp":
                timestamp_values.append(attr["value"])
        
        # Map sentiment to numeric values
        sentiment_map = {
            "Negative": -1,
            "Very Negative": -2,
            "Neutral": 0,
            "Positive": 1,
            "Very Positive": 2
        }
        
        # Process sentiment values
        if sentiment_values:
            # If we have a single sentiment value that describes a transition
            if len(sentiment_values) == 1 and any(x in sentiment_values[0].lower() for x in ["initially", "first", "then"]):
                # Extract initial and final sentiment
                if "negative" in sentiment_values[0].lower() and "positive" in sentiment_values[0].lower():
                    sentiment_data.append({
                        "conversation_id": conv_id,
                        "time_point": "Start",
                        "sentiment": -1
                    })
                    sentiment_data.append({
                        "conversation_id": conv_id,
                        "time_point": "End",
                        "sentiment": 1
                    })
            else:
                # Use the single sentiment value
                numeric_sentiment = 0
                for sentiment_text, value in sentiment_map.items():
                    if sentiment_text.lower() in sentiment_values[0].lower():
                        numeric_sentiment = value
                        break
                
                sentiment_data.append({
                    "conversation_id": conv_id,
                    "time_point": "Overall",
                    "sentiment": numeric_sentiment
                })
    
    if not sentiment_data:
        print("Warning: No usable sentiment data found for visualization")
        return None
        
    # Create DataFrame
    df = pd.DataFrame(sentiment_data)
    
    # Create visualization
    fig, ax = plt.figure(figsize=(10, 6)), plt.subplot(111)
    
    # Group by time point and calculate average sentiment
    sentiment_by_time = df.groupby('time_point')['sentiment'].mean().reset_index()
    
    if len(sentiment_by_time) > 1:
        # If we have multiple time points, create a flow chart
        colors = ['#d9534f', '#f0ad4e', '#5bc0de', '#5cb85c']
        cmap = LinearSegmentedColormap.from_list('sentiment', colors)
        
        # Plot the sentiment flow
        ax.plot(sentiment_by_time['time_point'], sentiment_by_time['sentiment'], 
                marker='o', linewidth=2, markersize=10)
        
        # Color the markers based on sentiment
        for i, row in sentiment_by_time.iterrows():
            ax.scatter(row['time_point'], row['sentiment'], 
                      c=[row['sentiment']], cmap=cmap, vmin=-2, vmax=2, s=100)
        
        ax.set_title('Customer Sentiment Flow During Conversations')
        ax.set_ylabel('Average Sentiment (-2 to +2)')
        ax.set_ylim(-2.5, 2.5)
        ax.grid(True, linestyle='--', alpha=0.7)
        
    else:
        # If we only have one time point, create a bar chart
        colors = ['#d9534f' if x < 0 else '#5cb85c' if x > 0 else '#f0ad4e' 
                 for x in sentiment_by_time['sentiment']]
        
        ax.bar(sentiment_by_time['time_point'], sentiment_by_time['sentiment'], color=colors)
        ax.set_title('Overall Customer Sentiment')
        ax.set_ylabel('Average Sentiment (-2 to +2)')
        ax.set_ylim(-2.5, 2.5)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    filename = f"{output_dir}/sentiment_flow.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    return filename

async def generate_resolution_sankey(statistics, attribute_results, output_dir, api_key=None, debug=False):
    """Generate Sankey diagram showing flow from dispute types to resolution outcomes."""
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.sankey import Sankey
    
    # Check if required attributes are available
    has_dispute_type = 'dispute_type' in statistics
    has_resolution_outcome = 'resolution_outcome' in statistics
    
    # If attributes are missing and we have an API key, generate them
    if (not has_dispute_type or not has_resolution_outcome) and api_key:
        from contact_center_analysis.extract import AttributeExtractor
        from collections import defaultdict
        from contact_center_analysis.categorize import Categorizer
        
        extractor = AttributeExtractor(api_key=api_key, debug=debug)
        
        # Define missing attributes
        attributes = []
        if not has_dispute_type:
            print("Dispute type attribute not found. Generating dispute type analysis...")
            attributes.append({
                "name": "dispute_type",
                "description": "The type of fee dispute the customer is calling about",
                "examples": [
                    "Annual Fee", "Monthly Fee", "Service Fee", "Transaction Fee", 
                    "Overdraft Fee", "Late Payment Fee"
                ]
            })
        
        if not has_resolution_outcome:
            print("Resolution outcome attribute not found. Generating outcome analysis...")
            attributes.append({
                "name": "resolution_outcome",
                "description": "The final outcome of the fee dispute",
                "examples": [
                    "Customer Won", "Customer Lost", "Partial Resolution", 
                    "Unresolved", "Pending"
                ]
            })
        
        # Extract attributes for each conversation
        generated_values = defaultdict(list)
        for i, result in enumerate(attribute_results):
            if "error" in result or "conversation_text" not in result:
                continue
                
            try:
                print(f"Generating attributes for conversation {i+1}/{len(attribute_results)}...")
                for attribute in attributes:
                    attr_result = await extractor.extract_attribute(
                        conversation_text=result["conversation_text"],
                        attribute=attribute
                    )
                    
                    # Add attribute to result
                    if "attribute_values" not in result:
                        result["attribute_values"] = []
                    
                    result["attribute_values"].append({
                        "field_name": attribute["name"],
                        "value": attr_result["value"],
                        "confidence": attr_result.get("confidence", 0.8)
                    })
                    
                    generated_values[attribute["name"]].append(attr_result["value"])
            except Exception as e:
                if debug:
                    print(f"Error generating attributes: {e}")
        
        # Apply semantic grouping to the generated values
        categorizer = Categorizer(api_key=api_key, debug=debug)
        
        for attr_name, values in generated_values.items():
            if not values:
                continue
                
            print(f"Consolidating values for {attr_name}...")
            
            # Convert to (value, count) format for consolidate_labels
            value_counts = defaultdict(int)
            for value in values:
                value_counts[value] += 1
            
            value_count_pairs = [(value, count) for value, count in value_counts.items()]
            
            # Consolidate similar labels
            consolidated = await categorizer.consolidate_labels(
                field_name=attr_name,
                values=value_count_pairs,
                max_categories=10
            )
            
            # Create a mapping from original values to consolidated ones
            value_mapping = {}
            for original, count in value_count_pairs:
                for category, items in consolidated.items():
                    if any(item[0] == original for item in items):
                        value_mapping[original] = category
                        break
            
            # Update the attribute values with consolidated labels
            for result in attribute_results:
                if "error" in result:
                    continue
                
                for attr in result.get("attribute_values", []):
                    if attr["field_name"] == attr_name and attr["value"] in value_mapping:
                        attr["value"] = value_mapping[attr["value"]]
        
        # Recompile statistics with the new attributes
        from collections import Counter
        
        for attr_name in ['dispute_type', 'resolution_outcome']:
            if attr_name not in statistics and attr_name in generated_values:
                values = []
                for result in attribute_results:
                    if "error" in result:
                        continue
                    
                    for attr in result.get("attribute_values", []):
                        if attr["field_name"] == attr_name:
                            values.append(attr["value"])
                
                if values:
                    value_counts = Counter(values)
                    statistics[attr_name] = {
                        "total_values": len(values),
                        "unique_values": len(value_counts),
                        "value_counts": dict(value_counts)
                    }
    
    # Check if we have the required data after generation attempts
    if 'dispute_type' not in statistics or 'resolution_outcome' not in statistics:
        print("Warning: Cannot generate Sankey diagram - missing required attributes")
        return None
    
    # Create a matrix of dispute type to resolution outcome
    dispute_types = list(statistics['dispute_type']['value_counts'].keys())
    outcomes = list(statistics['resolution_outcome']['value_counts'].keys())
    
    # Limit to top 5 dispute types and all outcomes for readability
    if len(dispute_types) > 5:
        top_disputes = sorted(
            dispute_types, 
            key=lambda x: statistics['dispute_type']['value_counts'][x], 
            reverse=True
        )[:5]
    else:
        top_disputes = dispute_types
    
    # Create a flow matrix
    flow_matrix = np.zeros((len(top_disputes), len(outcomes)))
    
    # Count flows from dispute types to outcomes
    for result in attribute_results:
        if "error" in result:
            continue
        
        dispute = None
        outcome = None
        
        for attr in result.get("attribute_values", []):
            if attr["field_name"] == "dispute_type":
                dispute = attr["value"]
            elif attr["field_name"] == "resolution_outcome":
                outcome = attr["value"]
        
        if dispute in top_disputes and outcome in outcomes:
            dispute_idx = top_disputes.index(dispute)
            outcome_idx = outcomes.index(outcome)
            flow_matrix[dispute_idx, outcome_idx] += 1
    
    # Create the Sankey diagram
    fig, ax = plt.figure(figsize=(12, 8)), plt.subplot(111)
    
    # Create a Sankey diagram
    sankey = Sankey(ax=ax, scale=0.01, offset=0.2, head_angle=120, margin=0.4, gap=0.05)
    
    # Add the first stage (dispute types)
    dispute_totals = flow_matrix.sum(axis=1)
    sankey.add(
        flows=dispute_totals,
        labels=top_disputes,
        orientations=[0] * len(top_disputes),
        pathlengths=[0.25] * len(top_disputes),
        facecolor='#1f77b4'
    )
    
    # Add the second stage (outcomes)
    outcome_totals = flow_matrix.sum(axis=0)
    sankey.add(
        flows=-outcome_totals,  # Negative because they're outflows
        labels=outcomes,
        orientations=[0] * len(outcomes),
        pathlengths=[0.25] * len(outcomes),
        facecolor='#ff7f0e'
    )
    
    # Connect the flows
    for i, dispute in enumerate(top_disputes):
        for j, outcome in enumerate(outcomes):
            if flow_matrix[i, j] > 0:
                sankey.add(
                    flows=[flow_matrix[i, j], -flow_matrix[i, j]],
                    orientations=[0, 0],
                    pathlengths=[0.1, 0.1],
                    facecolor='#2ca02c'
                )
    
    # Finish the diagram
    sankey.finish()
    ax.set_title('Flow from Dispute Types to Resolution Outcomes')
    ax.axis('off')
    
    # Save the figure
    filename = f"{output_dir}/dispute_resolution_flow.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    return filename

async def generate_action_wordcloud(statistics, attribute_results, output_dir, api_key=None, debug=False):
    """Generate word cloud of agent actions during fee disputes."""
    try:
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
        import os
        from collections import defaultdict
        from contact_center_analysis.categorize import Categorizer
        
        # Check if agent_actions is available in the statistics
        has_agent_actions = 'agent_actions' in statistics
        
        # If agent_actions is not available and we have an API key, generate it
        if not has_agent_actions and api_key:
            print("Agent actions attribute not found. Generating agent actions analysis...")
            from contact_center_analysis.extract import AttributeExtractor
            
            extractor = AttributeExtractor(api_key=api_key, debug=debug)
            categorizer = Categorizer(api_key=api_key, debug=debug)
            
            # Define the agent_actions attribute
            agent_actions_attribute = {
                "name": "agent_actions",
                "description": "Actions taken by the agent to resolve the fee dispute",
                "examples": [
                    "Fee waiver", "Refund issued", "Explanation provided", 
                    "Alternative account suggested", "Escalated to supervisor"
                ]
            }
            
            # Extract agent actions for each conversation
            agent_actions_values = []
            
            for i, result in enumerate(attribute_results):
                if "error" in result or "conversation_text" not in result:
                    continue
                    
                try:
                    print(f"Generating agent actions for conversation {i+1}/{len(attribute_results)}...")
                    actions_result = await extractor.extract_attribute(
                        conversation_text=result["conversation_text"],
                        attribute=agent_actions_attribute
                    )
                    
                    # Add agent actions to attribute values
                    if "attribute_values" not in result:
                        result["attribute_values"] = []
                        
                    result["attribute_values"].append({
                        "field_name": "agent_actions",
                        "value": actions_result["value"],
                        "confidence": actions_result.get("confidence", 0.8)
                    })
                    
                    agent_actions_values.append(actions_result["value"])
                    has_agent_actions = True
                except Exception as e:
                    if debug:
                        print(f"Error generating agent actions: {e}")
            
            # Apply semantic grouping to the agent actions values
            if agent_actions_values:
                print("Consolidating agent actions values with semantic grouping...")
                
                # Convert to (value, count) format for consolidate_labels
                value_counts = defaultdict(int)
                for value in agent_actions_values:
                    value_counts[value] += 1
                
                value_count_pairs = [(value, count) for value, count in value_counts.items()]
                
                # Consolidate similar labels
                consolidated = await categorizer.consolidate_labels(
                    field_name="agent_actions",
                    values=value_count_pairs,
                    max_categories=30  # Allow more categories for agent actions
                )
                
                # Create a mapping from original values to consolidated ones
                value_mapping = {}
                for original, count in value_count_pairs:
                    for category, items in consolidated.items():
                        if any(item[0] == original for item in items):
                            value_mapping[original] = category
                            break
                
                # Update the attribute values with consolidated labels
                for result in attribute_results:
                    if "error" in result:
                        continue
                    
                    for attr in result.get("attribute_values", []):
                        if attr["field_name"] == "agent_actions" and attr["value"] in value_mapping:
                            attr["value"] = value_mapping[attr["value"]]
                
                # Update statistics with the newly generated agent actions
                from collections import Counter
                
                # Collect all values for agent_actions
                all_values = []
                for result in attribute_results:
                    if "error" in result:
                        continue
                    
                    for attr in result.get("attribute_values", []):
                        if attr["field_name"] == "agent_actions":
                            all_values.append(attr["value"])
                
                # Calculate statistics
                value_counts = Counter(all_values)
                total_values = len(all_values)
                unique_values = len(value_counts)
                
                # Create statistics entry
                statistics["agent_actions"] = {
                    "total_values": total_values,
                    "unique_values": unique_values,
                    "value_counts": dict(value_counts),
                    "percentages": {k: (v / total_values) * 100 for k, v in value_counts.items()}
                }
        
        # If we still don't have agent actions data, return None
        if not has_agent_actions:
            print("Warning: Cannot generate agent actions word cloud - no data available")
            return None
        
        # Create the word cloud
        actions_text = " ".join(statistics['agent_actions']['value_counts'].keys())
        
        # Clean up the text
        # Remove common words that don't add meaning
        stopwords = ['and', 'the', 'to', 'of', 'a', 'in', 'for', 'on', 'with', 'as', 'by', 'an', 'is', 'was', 'were']
        
        # Generate the word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            stopwords=stopwords,
            min_font_size=10,
            max_font_size=50,
            colormap='viridis',
            contour_width=1,
            contour_color='steelblue'
        ).generate(actions_text)
        
        # Create the figure
        fig, ax = plt.figure(figsize=(10, 6)), plt.subplot(111)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title('Common Agent Actions in Fee Disputes')
        ax.axis('off')
        
        # Save the figure
        filename = f"{output_dir}/agent_actions_wordcloud.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        return filename
    except Exception as e:
        print(f"Error generating word cloud: {e}")
        return None

def generate_interactive_dashboard(analysis_results, statistics, visualizations, output_dir):
    """Generate an interactive HTML dashboard with all analysis results and visualizations."""
    import os
    import json
    from datetime import datetime
    
    # Create HTML template
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fee Dispute Analysis Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { padding-top: 20px; }
            .card { margin-bottom: 20px; }
            .visualization-img { max-width: 100%; height: auto; }
            .nav-pills .nav-link.active { background-color: #0d6efd; }
            .tab-content { padding-top: 20px; }
            .confidence-high { color: green; }
            .confidence-medium { color: orange; }
            .confidence-low { color: red; }
            .data-gap { background-color: #f8f9fa; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="mb-4">Fee Dispute Analysis Dashboard</h1>
            <p class="text-muted">Generated on {date}</p>
            
            <ul class="nav nav-pills mb-3" id="pills-tab" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="pills-summary-tab" data-bs-toggle="pill" 
                            data-bs-target="#pills-summary" type="button" 
                            aria-controls="pills-summary" aria-selected="true">Summary</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="pills-visualizations-tab" data-bs-toggle="pill" 
                            data-bs-target="#pills-visualizations" type="button" role="tab" 
                            aria-controls="pills-visualizations" aria-selected="false">Visualizations</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="pills-data-tab" data-bs-toggle="pill" 
                            data-bs-target="#pills-data" type="button" role="tab" 
                            aria-controls="pills-data" aria-selected="false">Data</button>
                </li>
            </ul>
            
            <div class="tab-content" id="pills-tabContent">
                <!-- Summary Tab -->
                <div class="tab-pane fade show active" id="pills-summary" role="tabpanel" 
                     aria-labelledby="pills-summary-tab">
                    <div class="row">
                        <div class="col-md-8">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Analysis Results</h5>
                                </div>
                                <div class="card-body">
                                    {analysis_results}
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Data Gaps</h5>
                                </div>
                                <div class="card-body">
                                    {data_gaps}
                                </div>
                            </div>
                            <div class="card">
                                <div class="card-header">
                                    <h5>Key Metrics</h5>
                                </div>
                                <div class="card-body">
                                    <canvas id="confidenceChart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Visualizations Tab -->
                <div class="tab-pane fade" id="pills-visualizations" role="tabpanel" 
                     aria-labelledby="pills-visualizations-tab">
                    <div class="row">
                        {visualization_cards}
                    </div>
                </div>
                
                <!-- Data Tab -->
                <div class="tab-pane fade" id="pills-data" role="tabpanel" 
                     aria-labelledby="pills-data-tab">
                    <div class="card">
                        <div class="card-header">
                            <h5>Attribute Statistics</h5>
                        </div>
                        <div class="card-body">
                            <div class="accordion" id="statisticsAccordion">
                                {statistics_accordion}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Confidence chart
            const confidenceCtx = document.getElementById('confidenceChart').getContext('2d');
            const confidenceChart = new Chart(confidenceCtx, {
                type: 'bar',
                data: {
                    labels: {confidence_labels},
                    datasets: [{
                        label: 'Confidence Level',
                        data: {confidence_values},
                        backgroundColor: [
                            'rgba(255, 99, 132, 0.2)',
                            'rgba(255, 159, 64, 0.2)',
                            'rgba(255, 205, 86, 0.2)',
                            'rgba(75, 192, 192, 0.2)',
                            'rgba(54, 162, 235, 0.2)'
                        ],
                        borderColor: [
                            'rgb(255, 99, 132)',
                            'rgb(255, 159, 64)',
                            'rgb(255, 205, 86)',
                            'rgb(75, 192, 192)',
                            'rgb(54, 162, 235)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    indexAxis: 'y',
                    scales: {
                        x: {
                            beginAtZero: true,
                            max: 3,
                            ticks: {
                                callback: function(value) {
                                    return ['', 'Low', 'Medium', 'High'][value];
                                }
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        </script>
    </body>
    </html>
    """
    
    # Format analysis results
    analysis_html = ""
    confidence_labels = []
    confidence_values = []
    confidence_map = {"Low": 1, "Medium": 2, "High": 3}
    
    if "answers" in analysis_results:
        for answer in analysis_results["answers"]:
            question = answer.get("question", "")
            answer_text = answer.get("answer", "")
            confidence = answer.get("confidence", "Medium")
            key_metrics = answer.get("key_metrics", [])
            
            confidence_class = f"confidence-{confidence.lower()}"
            confidence_labels.append(question[:30] + "..." if len(question) > 30 else question)
            confidence_values.append(confidence_map.get(confidence, 2))
            
            analysis_html += f"""
            <div class="mb-4">
                <h5>{question}</h5>
                <p>{answer_text}</p>
                <p><strong>Key Metrics:</strong> {', '.join(key_metrics)}</p>
                <p><strong>Confidence:</strong> <span class="{confidence_class}">{confidence}</span></p>
            </div>
            """
    
    # Format data gaps
    data_gaps_html = ""
    if "data_gaps" in analysis_results and analysis_results["data_gaps"]:
        for gap in analysis_results["data_gaps"]:
            data_gaps_html += f'<div class="data-gap">{gap}</div>'
    else:
        data_gaps_html = "<p>No data gaps identified.</p>"
    
    # Format visualizations
    visualization_cards_html = ""
    for viz in visualizations:
        viz_type = viz.get("type", "")
        filename = viz.get("filename", "")
        title = ""
        
        if viz_type == "distribution":
            metric = viz.get("metric", "")
            question = viz.get("question", "")
            title = f"Distribution of {metric}"
            description = f"Related to: {question}"
        elif viz_type == "confidence":
            title = "Confidence Levels by Question"
            description = "Shows confidence in each answer"
        elif viz_type == "gap_resolution":
            title = "Data Gaps Before and After Resolution"
            description = "Shows improvement after gap resolution"
        elif viz_type == "sentiment_flow":
            title = "Customer Sentiment Flow"
            description = "How sentiment changes during conversations"
        elif viz_type == "resolution_flow":
            title = "Dispute to Resolution Flow"
            description = "How different disputes lead to outcomes"
        elif viz_type == "wordcloud":
            title = "Agent Actions Word Cloud"
            description = "Common actions taken by agents"
        
        # Get just the filename without the path
        img_filename = os.path.basename(filename)
        
        visualization_cards_html += f"""
        <div class="col-md-6 mb-4">
            <div class="card">
                <div class="card-header">
                    <h5>{title}</h5>
                </div>
                <div class="card-body text-center">
                    <img src="{img_filename}" class="visualization-img" alt="{title}">
                    <p class="mt-2">{description}</p>
                </div>
            </div>
        </div>
        """
    
    # Format statistics accordion
    statistics_accordion_html = ""
    for i, (field_name, stats) in enumerate(statistics.items()):
        # Get top values
        top_values = sorted(stats['value_counts'].items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Format the values as a table
        values_table = """
        <table class="table table-striped">
            <thead>
                <tr>
                    <th>Value</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for value, count in top_values:
            percentage = stats['percentages'][value]
            values_table += f"""
            <tr>
                <td>{value}</td>
                <td>{count}</td>
                <td>{percentage:.1f}%</td>
            </tr>
            """
        
        values_table += """
            </tbody>
        </table>
        """
        
        statistics_accordion_html += f"""
        <div class="accordion-item">
            <h2 class="accordion-header" id="heading{i}">
                <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" 
                        data-bs-target="#collapse{i}" aria-expanded="false" aria-controls="collapse{i}">
                    {field_name} ({stats['total_values']} values, {stats['unique_values']} unique)
                </button>
            </h2>
            <div id="collapse{i}" class="accordion-collapse collapse" aria-labelledby="heading{i}" 
                 data-bs-parent="#statisticsAccordion">
                <div class="accordion-body">
                    {values_table}
                </div>
            </div>
        </div>
        """
    
    # Fill in the template
    html_content = html_template.format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        analysis_results=analysis_html,
        data_gaps=data_gaps_html,
        visualization_cards=visualization_cards_html,
        statistics_accordion=statistics_accordion_html,
        confidence_labels=json.dumps(confidence_labels),
        confidence_values=json.dumps(confidence_values)
    )
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write the HTML file
    output_file = os.path.join(output_dir, "dashboard.html")
    with open(output_file, "w") as f:
        f.write(html_content)
    
    # Copy visualization images to the output directory if they're not already there
    for viz in visualizations:
        filename = viz.get("filename", "")
        if filename and os.path.exists(filename):
            # Get just the filename without the path
            img_filename = os.path.basename(filename)
            # Copy the file if it's not already in the output directory
            if os.path.dirname(filename) != output_dir:
                import shutil
                shutil.copy(filename, os.path.join(output_dir, img_filename))
    
    return output_file

async def main():
    """Parse arguments and run the script."""
    parser = argparse.ArgumentParser(description='Analyze fee dispute conversations')
    parser.add_argument('--db', required=True, help='Path to the SQLite database')
    parser.add_argument('--sample-size', type=int, default=50, help='Number of conversations to analyze')
    parser.add_argument('--target-class', default='fee dispute', help='Target class for intent matching')
    parser.add_argument('--examples', nargs='+', help='Example intents for the target class')
    parser.add_argument('--output', help='Optional path to save results as JSON')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--batch-size', type=int, default=5, 
                        help='Number of conversations to process in parallel (default: 5)')
    parser.add_argument('--save-attributes', help='Optional path to save raw attribute values as JSON')
    parser.add_argument('--max-retries', type=int, default=3, 
                        help='Maximum number of retries for failed API calls (default: 3)')
    parser.add_argument('--max-gap-iterations', type=int, default=1,
                        help='Maximum number of data gap resolution iterations (default: 1)')
    parser.add_argument('--visualize', action='store_true', 
                        help='Generate visualizations of analysis results')
    parser.add_argument('--visualization-dir', 
                        help='Directory to save visualizations (default: ./visualizations)')
    
    args = parser.parse_args()
    
    # Get API key from environment variable
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set")
        return
    
    start_time = time.time()
    
    # Define research questions at the beginning
    questions = [
        "What are the most common types of fee disputes customers call about?",
        "How often do agents offer refunds or credits to resolve fee disputes?",
        "What percentage of fee disputes are resolved in the customer's favor?",
        "What explanations do agents provide for disputed fees?",
        "How do agents de-escalate conversations when customers are upset about fees?"
    ]
    
    # Step 1: Get required attributes
    required_attributes = await get_required_attributes(questions, api_key, args.debug)
    
    # Deduplicate required attributes by field_name
    unique_attributes = {}
    for attr in required_attributes:
        field_name = attr['field_name']
        if field_name not in unique_attributes:
            unique_attributes[field_name] = attr
    
    required_attributes = list(unique_attributes.values())
    
    print(f"\nIdentified {len(required_attributes)} required attributes:")
    for attr in required_attributes:
        print(f"  - {attr['title']} ({attr['field_name']}): {attr['description']}")
    
    # Step 2: Find matching intents for fee disputes
    print(f"\nFinding intents related to '{args.target_class}'...")
    matching_intents = await find_matching_intents(
        db_path=args.db,
        api_key=api_key,
        target_class=args.target_class,
        examples=args.examples,
        debug=args.debug
    )
    
    if not matching_intents:
        print(f"No intents matching '{args.target_class}' were found. Exiting.")
        return
    
    # Step 3: Fetch conversations with matching intents
    print(f"\nFetching {args.sample_size} conversations with '{args.target_class}' intents...")
    conversations = await fetch_conversations_by_intents(
        db_path=args.db,
        matching_intents=matching_intents,
        limit=args.sample_size
    )
    
    if not conversations:
        print("No conversations with matching intents found. Exiting.")
        return
    
    print(f"Found {len(conversations)} conversations for analysis")
    
    # Step 4: Generate attribute values for all conversations
    print(f"\nGenerating attribute values for all {len(conversations)} conversations in parallel batches...")
    attribute_results = await generate_attributes_for_conversations(
        conversations,
        required_attributes,
        api_key,
        args.debug,
        batch_size=args.batch_size,
        max_retries=args.max_retries
    )
    
    # Step 5: Compile statistics on attribute values
    print("\nCompiling attribute statistics with semantic grouping...")
    statistics = await compile_attribute_statistics(
        attribute_results,
        api_key,
        args.debug
    )
    
    # Print summary statistics
    print("\n=== Attribute Value Statistics ===")
    for field_name, stats in statistics.items():
        attr = next((a for a in required_attributes if a['field_name'] == field_name), None)
        title = attr['title'] if attr else field_name
        
        print(f"\n{title} ({field_name}):")
        print(f"  Total values: {stats['total_values']}")
        print(f"  Unique values: {stats['unique_values']}")
        print("  Top values:")
        
        # Sort by count and show top 5
        top_values = sorted(stats['value_counts'].items(), key=lambda x: x[1], reverse=True)[:5]
        for value, count in top_values:
            percentage = stats['percentages'][value]
            print(f"    - {value}: {count} ({percentage:.1f}%)")
    
    # After Step 5 (compile statistics) and before Step 6 (analyze findings)
    print("\nExtracting detailed insights from attribute values...")
    detailed_insights = await review_attribute_details(
        attribute_results,
        required_attributes,
        questions,  # Now questions is defined
        api_key,
        args.debug
    )

    # Step 6: Analyze the findings to answer research questions
    print("\nAnalyzing findings to answer research questions...")
    analysis = await analyze_attribute_findings(
        statistics,
        questions,
        api_key,
        args.debug,
        detailed_insights  # Pass the detailed insights
    )

    # Print analysis results
    print("\n=== Analysis Results ===")
    if "answers" in analysis:
        for answer in analysis["answers"]:
            print(f"\nQuestion: {answer['question']}")
            print(f"Answer: {answer['answer']}")
            print(f"Key Metrics: {', '.join(answer['key_metrics'])}")
            print(f"Confidence: {answer['confidence']}")
            print(f"Supporting Data: {answer['supporting_data']}")

    if "data_gaps" in analysis and analysis["data_gaps"]:
        print("\nData Gaps Identified:")
        for gap in analysis["data_gaps"]:
            print(f"  - {gap}")

    # After the initial analysis
    print("\nChecking for data gaps and enhancing analysis...")
    # Create the analyzer instance
    analyzer = DataAnalyzer(api_key=api_key, debug=args.debug)
    enhanced_analysis = await analyzer.resolve_data_gaps(
        analysis_results=analysis,
        attribute_results=attribute_results,
        statistics=statistics,
        questions=questions,
        detailed_insights=detailed_insights,
        max_iterations=args.max_gap_iterations
    )

    # Print enhanced analysis results
    print("\n=== Enhanced Analysis Results ===")
    if "answers" in enhanced_analysis:
        for answer in enhanced_analysis["answers"]:
            print(f"\nQuestion: {answer['question']}")
            print(f"Answer: {answer['answer']}")
            print(f"Key Metrics: {', '.join(answer['key_metrics'])}")
            print(f"Confidence: {answer['confidence']}")
            print(f"Supporting Data: {answer['supporting_data']}")

    # Print any remaining data gaps
    if "data_gaps" in enhanced_analysis and enhanced_analysis["data_gaps"]:
        print("\nRemaining Data Gaps:")
        for gap in enhanced_analysis["data_gaps"]:
            print(f"  - {gap}")
    
    # Save results to file if requested
    if args.output:
        output_results = {
            "required_attributes": required_attributes,
            "matching_intents": matching_intents,
            "attribute_results": attribute_results,
            "statistics": statistics,
            "analysis": analysis
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Save raw attribute values if requested
    if args.save_attributes:
        with open(args.save_attributes, 'w') as f:
            json.dump(attribute_results, f, indent=2)
        print(f"Raw attribute values saved to {args.save_attributes}")
    
    # Print timing information
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")

    # After printing enhanced analysis results
    if args.visualize:
        print("\nGenerating visualizations...")
        visualization_dir = args.visualization_dir or "dataviz"
        
        # Generate standard visualizations
        visualizations = await generate_visualizations(
            analysis_results=enhanced_analysis,
            statistics=statistics,
            attribute_results=attribute_results,
            output_dir=visualization_dir
        )
        
        # Generate additional visualizations with attribute generation if needed
        sentiment_viz = await generate_sentiment_flow(
            attribute_results=attribute_results, 
            output_dir=visualization_dir,
            api_key=api_key,
            debug=args.debug
        )
        if sentiment_viz:
            visualizations.append({
                "type": "sentiment_flow",
                "filename": sentiment_viz
            })
            
        resolution_viz = await generate_resolution_sankey(
            statistics=statistics,
            attribute_results=attribute_results,
            output_dir=visualization_dir,
            api_key=api_key,
            debug=args.debug
        )
        if resolution_viz:
            visualizations.append({
                "type": "resolution_flow",
                "filename": resolution_viz
            })
            
        wordcloud_viz = await generate_action_wordcloud(
            statistics=statistics,
            attribute_results=attribute_results,
            output_dir=visualization_dir,
            api_key=api_key,
            debug=args.debug
        )
        if wordcloud_viz:
            visualizations.append({
                "type": "wordcloud",
                "filename": wordcloud_viz
            })
        
        # Generate interactive dashboard
        dashboard_file = generate_interactive_dashboard(
            analysis_results=enhanced_analysis,
            statistics=statistics,
            visualizations=visualizations,
            output_dir=visualization_dir
        )
        
        print(f"Generated {len(visualizations)} visualizations")
        print(f"Interactive dashboard available at: {dashboard_file}")

if __name__ == "__main__":
    asyncio.run(main()) 