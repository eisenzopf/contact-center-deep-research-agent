import sqlite3
import asyncio
import sys
import os
import argparse
import json
import time
from contact_center_analysis.text import TextGenerator
from contact_center_analysis.match import AttributeMatcher

async def get_required_attributes(api_key, debug=False):
    """
    Get the list of required attributes needed to answer cancellation-related questions.
    
    Args:
        api_key: API key for the TextGenerator
        debug: Enable debug mode for more verbose output
        
    Returns:
        List of required attribute dictionaries
    """
    # Initialize the TextGenerator
    generator = TextGenerator(api_key=api_key, debug=debug)
    
    # Define research questions about cancellations
    questions = [
        "When customers call to cancel their service, how often do agents try to save the customer?",
        "When agents do try to save, how often do they succeed?",
        "What are things that agents offer that most often retain customers?",
        "What did the customer say the reason was they were cancelling?"
    ]
    
    # Generate required attributes
    print("Generating required attributes...")
    result = await generator.generate_required_attributes(
        questions=questions
    )
    
    return result["attributes"]

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

async def main():
    """Parse arguments and run the script."""
    parser = argparse.ArgumentParser(description='Identify required attributes for cancellation analysis')
    parser.add_argument('--db', required=True, help='Path to the SQLite database')
    parser.add_argument('--min-count', type=int, default=20, help='Minimum count for attributes to be considered')
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
    
    # Step 2: Get required attributes
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
    
    # Step 3: Fetch existing attributes from database
    existing_attributes = await fetch_existing_attributes(args.db, args.min_count)
    print(f"\nFound {len(existing_attributes)} existing attributes in the database")
    
    # Match required attributes against existing ones
    matches, missing = await match_attributes(required_attributes, existing_attributes, api_key, args.debug, 
                                             confidence_threshold=args.threshold)
    
    # Process results
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
    
    # Save results to file if requested
    if args.output:
        results = {
            "required_attributes": required_attributes,
            "existing_attributes": [attr for attr in existing_attributes],
            "matches": matches,
            "missing_attributes": missing
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Print timing information
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())