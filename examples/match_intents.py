import sqlite3
import asyncio
import sys
import os
import argparse
import json
import time
from contact_center_analysis.categorize import Categorizer

async def extract_unique_intents(db_path):
    """
    Extract unique intent values from the conversation_attributes table.
    Only includes intents that appear at least 20 times.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        List of unique intent dictionaries with 'name' field
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Query for unique intent values that appear at least 20 times
    query = """
    SELECT value as intent_text, COUNT(*) as count
    FROM conversation_attributes
    WHERE type = 'intent'
    GROUP BY value
    HAVING COUNT(*) >= 20
    """
    
    cursor.execute(query)
    # Format intents as dictionaries with 'name' field for Categorizer
    intents = [{"name": row['intent_text']} for row in cursor.fetchall()]
    
    # Close connection
    conn.close()
    
    return intents

async def classify_intents(db_path, api_key, target_class, examples=None, debug=False):
    """
    Classify intents from the database using the Categorizer.
    
    Args:
        db_path: Path to the SQLite database
        api_key: API key for the Categorizer
        target_class: The class to classify intents into (e.g., "cancellation", "billing")
        examples: Optional list of example intents for the target class
        debug: Enable debug mode for more verbose output
        
    Returns:
        Dictionary with classification results
    """
    # Initialize the Categorizer
    categorizer = Categorizer(api_key=api_key, debug=debug)
    
    # Extract unique intents from the database
    db_intents = await extract_unique_intents(db_path)
    
    print(f"Found {len(db_intents)} unique intents in the database")
    
    # Classify the intents using the Categorizer's batch method
    start_time = time.time()
    batch_results = await categorizer.is_in_class_batch(
        intents=db_intents,
        target_class=target_class,
        examples=examples,
        batch_size=50  # Adjust based on your needs
    )
    
    # Process results
    matching_intents = []
    non_matching_intents = []
    
    # Process the batch results (list of dictionaries)
    for result_dict in batch_results:
        for intent_text, is_match in result_dict.items():
            if is_match:
                matching_intents.append(intent_text)
            else:
                non_matching_intents.append(intent_text)
    
    return {
        "matching_intents": matching_intents,
        "non_matching_intents": non_matching_intents,
        "db_intents": [intent["name"] for intent in db_intents],
        "target_class": target_class,
        "examples": examples or []
    }

async def main():
    """Parse arguments and run the script."""
    parser = argparse.ArgumentParser(description='Classify intents from the database using semantic matching')
    parser.add_argument('--db', required=True, help='Path to the SQLite database')
    parser.add_argument('--class', dest='target_class', required=True, 
                        help='Target class for classification (e.g., "cancellation", "billing")')
    parser.add_argument('--examples', nargs='+', help='Example intents for the target class')
    parser.add_argument('--output', help='Optional path to save results as JSON')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Get API key from environment variable
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set")
        return
    
    start_time = time.time()
    results = await classify_intents(
        db_path=args.db,
        api_key=api_key,
        target_class=args.target_class,
        examples=args.examples,
        debug=args.debug
    )
    
    # Print results
    print(f"\nClassification Results:")
    print(f"Target class: {args.target_class}")
    if args.examples:
        print(f"Examples provided: {len(args.examples)}")
    print(f"Total database intents: {len(results['db_intents'])}")
    print(f"Matching intents: {len(results['matching_intents'])}")
    print(f"Non-matching intents: {len(results['non_matching_intents'])}")
    
    # Print examples of matching intents
    if results['matching_intents']:
        print("\nMatching intents:")
        for intent in results['matching_intents']:
            print(f"  - {intent}")
    else:
        print("\nNo matching intents found.")
    
    # Save results to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Print timing information
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main()) 