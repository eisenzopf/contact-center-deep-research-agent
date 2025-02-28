import sqlite3
import asyncio
import sys
import os
import argparse
import json
import time
from contact_center_analysis.text import TextGenerator
from contact_center_analysis.match import AttributeMatcher
from contact_center_analysis.categorize import Categorizer

async def get_required_attributes(api_key, debug=False):
    """
    Get the list of required attributes needed to answer fee dispute-related questions.
    
    Args:
        api_key: API key for the TextGenerator
        debug: Enable debug mode for more verbose output
        
    Returns:
        List of required attribute dictionaries
    """
    # Initialize the TextGenerator
    generator = TextGenerator(api_key=api_key, debug=debug)
    
    # Define research questions about fee disputes
    questions = [
        "What are the most common types of fee disputes customers call about?",
        "How often do agents offer refunds or credits to resolve fee disputes?",
        "What percentage of fee disputes are resolved in the customer's favor?",
        "What explanations do agents provide for disputed fees?",
        "How do agents de-escalate conversations when customers are upset about fees?"
    ]
    
    # Generate required attributes
    print("Generating required attributes for fee dispute analysis...")
    result = await generator.generate_required_attributes(
        questions=questions
    )
    
    return result["attributes"]

async def extract_unique_intents(db_path, min_count=20):
    """
    Extract unique intent values from the conversation_attributes table.
    Only includes intents that appear at least min_count times.
    
    Args:
        db_path: Path to the SQLite database
        min_count: Minimum count for intents to be considered
        
    Returns:
        List of unique intent dictionaries with 'name' field
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Query for unique intent values that appear at least min_count times
    query = """
    SELECT value as intent_text, COUNT(*) as count
    FROM conversation_attributes
    WHERE type = 'intent'
    GROUP BY value
    HAVING COUNT(*) >= ?
    """
    
    cursor.execute(query, (min_count,))
    # Format intents as dictionaries with 'name' field for Categorizer
    intents = [{"name": row['intent_text']} for row in cursor.fetchall()]
    
    # Close connection
    conn.close()
    
    return intents

async def find_matching_intents(db_path, api_key, target_class, examples=None, min_count=20, debug=False):
    """
    Find intents that match a specific target class.
    
    Args:
        db_path: Path to the SQLite database
        api_key: API key for the Categorizer
        target_class: The class to classify intents into (e.g., "fee dispute")
        examples: Optional list of example intents for the target class
        min_count: Minimum count for intents to be considered
        debug: Enable debug mode for more verbose output
        
    Returns:
        List of matching intent strings
    """
    # Initialize the Categorizer
    categorizer = Categorizer(api_key=api_key, debug=debug)
    
    # Extract unique intents from the database
    db_intents = await extract_unique_intents(db_path, min_count)
    
    print(f"Found {len(db_intents)} unique intents in the database")
    
    # If no examples provided, use default examples for fee disputes
    if not examples:
        examples = [
            "Customer disputing a late payment fee",
            "Customer questioning monthly service charge",
            "Customer complaining about unexpected fees on bill",
            "Customer wants fee waived",
            "Customer says they were charged incorrectly",
            "Customer asking for refund of fees"
        ]
    
    # Classify the intents using the Categorizer's batch method
    batch_results = await categorizer.is_in_class_batch(
        intents=db_intents,
        target_class=target_class,
        examples=examples,
        batch_size=50  # Adjust based on your needs
    )
    
    # Process results to get matching intents
    matching_intents = []
    
    # Process the batch results (list of dictionaries)
    for result_dict in batch_results:
        for intent_text, is_match in result_dict.items():
            if is_match:
                matching_intents.append(intent_text)
    
    print(f"Found {len(matching_intents)} intents matching '{target_class}'")
    if matching_intents:
        print("Examples of matching intents:")
        for intent in matching_intents[:5]:  # Show first 5 examples
            print(f"  - {intent}")
    
    return matching_intents

async def fetch_conversations_by_intents(db_path, matching_intents, limit=5):
    """
    Fetch conversations that have intents matching the specified list.
    
    Args:
        db_path: Path to the SQLite database
        matching_intents: List of intent strings to match
        limit: Maximum number of conversations to fetch
        
    Returns:
        List of conversation dictionaries
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Create placeholders for SQL IN clause
    placeholders = ', '.join(['?'] * len(matching_intents))
    
    # Query for conversations with matching intents
    query = f"""
    SELECT c.conversation_id, c.text
    FROM conversations c
    JOIN conversation_attributes ca ON c.conversation_id = ca.conversation_id
    WHERE ca.type = 'intent' AND ca.value IN ({placeholders})
    AND c.text IS NOT NULL AND LENGTH(c.text) > 100
    ORDER BY RANDOM()
    LIMIT ?
    """
    
    # Add limit to the parameters
    params = matching_intents + [limit]
    cursor.execute(query, params)
    
    # Format conversations as dictionaries
    conversations = []
    for row in cursor.fetchall():
        conversations.append({
            'id': row['conversation_id'],
            'text': row['text']
        })
    
    # Close connection
    conn.close()
    
    return conversations

async def fetch_existing_attributes(db_path, min_count=20):
    """
    Fetch existing attributes from the database.
    
    Args:
        db_path: Path to the SQLite database
        min_count: Minimum count for attributes to be considered
        
    Returns:
        List of existing attribute dictionaries
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Query for unique attributes that appear at least min_count times
    query = """
    SELECT name, type, COUNT(*) as count
    FROM conversation_attributes
    WHERE type = 'attribute'
    GROUP BY name
    HAVING COUNT(*) >= ?
    """
    
    cursor.execute(query, (min_count,))
    
    # Format attributes as dictionaries
    attributes = []
    for row in cursor.fetchall():
        # For each unique attribute name, get a sample value
        value_query = """
        SELECT value, description
        FROM conversation_attributes
        WHERE name = ? AND type = 'attribute'
        LIMIT 1
        """
        cursor.execute(value_query, (row['name'],))
        value_row = cursor.fetchone()
        
        attributes.append({
            'name': row['name'],
            'type': row['type'],
            'value': value_row['value'] if value_row else '',
            'description': value_row['description'] if value_row and value_row['description'] else '',
            'count': row['count']
        })
    
    # Close connection
    conn.close()
    
    return attributes

async def match_attributes(required_attributes, existing_attributes, api_key, debug=False, confidence_threshold=0.7):
    """
    Match required attributes against existing attributes.
    
    Args:
        required_attributes: List of required attribute dictionaries
        existing_attributes: List of existing attribute dictionaries
        api_key: API key for the AttributeMatcher
        debug: Enable debug mode for more verbose output
        confidence_threshold: Confidence threshold for attribute matching
        
    Returns:
        Tuple of (matched attributes, missing attributes)
    """
    # Initialize the AttributeMatcher
    matcher = AttributeMatcher(api_key=api_key, debug=debug)
    
    # Prepare existing attributes to ensure description is valid JSON
    prepared_attributes = []
    for attr in existing_attributes:
        # Create a copy of the attribute
        prepared_attr = attr.copy()
        
        # Ensure description is valid JSON with all required fields
        if not prepared_attr['description'] or prepared_attr['description'] == '':
            prepared_attr['description'] = json.dumps({
                "title": attr['name'].replace('_', ' ').title(),
                "description": f"Attribute: {attr['name']}",
                "type": attr['type'],
                "value": attr['value']
            })
        else:
            # Try to parse existing description, and add missing fields if needed
            try:
                desc_obj = json.loads(prepared_attr['description'])
                if not isinstance(desc_obj, dict):
                    desc_obj = {"description": str(desc_obj)}
                
                # Ensure all required fields exist
                if 'title' not in desc_obj:
                    desc_obj['title'] = attr['name'].replace('_', ' ').title()
                if 'description' not in desc_obj:
                    desc_obj['description'] = f"Attribute: {attr['name']}"
                
                prepared_attr['description'] = json.dumps(desc_obj)
            except json.JSONDecodeError:
                # If parsing fails, create a new description
                prepared_attr['description'] = json.dumps({
                    "title": attr['name'].replace('_', ' ').title(),
                    "description": prepared_attr['description'] or f"Attribute: {attr['name']}",
                    "type": attr['type'],
                    "value": attr['value']
                })
        
        prepared_attributes.append(prepared_attr)
    
    # Match attributes
    print("Matching attributes...")
    matches, missing = await matcher.find_matches(
        required_attributes=required_attributes,
        available_attributes=prepared_attributes,
        confidence_threshold=confidence_threshold
    )
    
    return matches, missing

async def generate_missing_attributes(conversations, missing_attributes, api_key, debug=False):
    """
    Generate values for missing attributes from sample conversations.
    
    Args:
        conversations: List of conversation dictionaries
        missing_attributes: List of missing attribute dictionaries
        api_key: API key for the TextGenerator
        debug: Enable debug mode for more verbose output
        
    Returns:
        List of dictionaries with generated attribute values
    """
    # Initialize the TextGenerator
    generator = TextGenerator(api_key=api_key, debug=debug)
    
    results = []
    
    # Process each conversation
    for conversation in conversations:
        print(f"\nAnalyzing conversation {conversation['id']}...")
        
        # Generate values for all missing attributes in a single call
        attribute_values = await generator.generate_attributes(
            text=conversation['text'],
            attributes=missing_attributes
        )
        
        results.append({
            "conversation_id": conversation['id'],
            "attribute_values": attribute_values
        })
    
    return results

async def main():
    """Parse arguments and run the script."""
    parser = argparse.ArgumentParser(description='Generate values for missing attributes from fee dispute conversations')
    parser.add_argument('--db', required=True, help='Path to the SQLite database')
    parser.add_argument('--min-count', type=int, default=20, help='Minimum count for attributes to be considered')
    parser.add_argument('--sample-size', type=int, default=3, help='Number of sample conversations to analyze')
    parser.add_argument('--target-class', default='fee dispute', help='Target class for intent matching (default: "fee dispute")')
    parser.add_argument('--examples', nargs='+', help='Example intents for the target class')
    parser.add_argument('--output', help='Optional path to save results as JSON')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--threshold', type=float, default=0.7, help='Confidence threshold for attribute matching')
    
    args = parser.parse_args()
    
    # Get API key from environment variable
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set")
        return
    
    start_time = time.time()
    
    # Step 1: Get required attributes
    required_attributes = await get_required_attributes(api_key, args.debug)
    
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
    
    # Step 2: Fetch existing attributes from database
    existing_attributes = await fetch_existing_attributes(args.db, args.min_count)
    print(f"\nFound {len(existing_attributes)} existing attributes in the database")
    
    # Step 3: Match required attributes against existing ones
    matches, missing = await match_attributes(
        required_attributes, 
        existing_attributes, 
        api_key, 
        args.debug, 
        confidence_threshold=args.threshold
    )
    
    # Process matching results
    print("\n=== Attribute Analysis Summary ===")
    print(f"Total required attributes: {len(required_attributes)}")
    print(f"Existing attributes: {len(matches)}")
    print(f"Missing attributes: {len(missing)}")
    
    # Print existing attributes with matches
    if matches:
        print("\n=== Existing Attributes ===")
        for req_field, match_info in matches.items():
            # Find the required attribute definition
            req_attr = next((attr for attr in required_attributes if attr['field_name'] == req_field), None)
            if req_attr:
                print(f"\n✓ Found: {req_attr['title']} ({req_field})")
                print(f"  - Matched to database field: {match_info['field']}")
                print(f"  - Confidence: {match_info['confidence']:.2f}")
    
    # Print missing attributes
    if missing:
        print("\n=== Missing Attributes ===")
        for attr in missing:
            print(f"\n✗ Missing: {attr['title']} ({attr['field_name']})")
            print(f"  - Description: {attr['description']}")
            
            # If there was a low-confidence match, show it
            if 'best_match' in attr and attr['best_match']:
                print(f"  - Best potential match: {attr['best_match']['field']} (confidence: {attr['best_match']['confidence']:.2f})")
    
    # Step 4: Find matching intents for the target class
    if missing:
        print(f"\nFinding intents related to '{args.target_class}'...")
        matching_intents = await find_matching_intents(
            db_path=args.db,
            api_key=api_key,
            target_class=args.target_class,
            examples=args.examples,
            min_count=args.min_count,
            debug=args.debug
        )
        
        if not matching_intents:
            print(f"No intents matching '{args.target_class}' were found. Using random conversations instead.")
            conversations = await fetch_sample_conversations(args.db, args.sample_size)
        else:
            # Step 5: Fetch conversations with matching intents
            print(f"\nFetching {args.sample_size} conversations with '{args.target_class}' intents...")
            conversations = await fetch_conversations_by_intents(
                db_path=args.db,
                matching_intents=matching_intents,
                limit=args.sample_size
            )
            
            if not conversations:
                print("No conversations with matching intents found. Using random conversations instead.")
                conversations = await fetch_sample_conversations(args.db, args.sample_size)
        
        if not conversations:
            print("No conversations found in the database.")
            return
        
        # Step 6: Generate values for missing attributes
        print("\nGenerating values for missing attributes...")
        results = await generate_missing_attributes(
            conversations, 
            missing, 
            api_key, 
            args.debug
        )
        
        # Print generated attribute values
        print("\n=== Generated Attribute Values ===")
        for result in results:
            print(f"\nConversation ID: {result['conversation_id']}")
            for attr_value in result['attribute_values']:
                print(f"\n  Attribute: {attr_value['field_name']}")
                print(f"  Value: {attr_value['value']}")
                print(f"  Confidence: {attr_value['confidence']:.2f}")
                print(f"  Explanation: {attr_value['explanation']}")
        
        # Save results to file if requested
        if args.output:
            output_results = {
                "required_attributes": required_attributes,
                "existing_attributes": existing_attributes,
                "matches": matches,
                "missing_attributes": missing,
                "matching_intents": matching_intents,
                "generated_values": results
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_results, f, indent=2)
            print(f"\nResults saved to {args.output}")
    
    # Print timing information
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main()) 