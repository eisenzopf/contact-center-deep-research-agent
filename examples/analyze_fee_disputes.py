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
            confidence = attr_value["confidence"]
            
            # Only include values with reasonable confidence
            if confidence >= 0.6:
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
                max_groups=min(20, len(value_counts))  # Limit groups to 20 or fewer
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

if __name__ == "__main__":
    asyncio.run(main()) 