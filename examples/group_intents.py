import sqlite3
import asyncio
import sys
import os
import argparse
import json
import time
import csv
from contact_center_analysis.categorize import Categorizer

async def extract_intent_distribution(db_path, min_count=1):
    """
    Extract intent values and their counts from the conversation_attributes table.
    
    Args:
        db_path: Path to the SQLite database
        min_count: Minimum count to include an intent (default=1)
        
    Returns:
        List of tuples (intent_text, count)
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Query for intent values and their counts
    query = """
    SELECT value as intent_text, COUNT(*) as count
    FROM conversation_attributes
    WHERE type = 'intent'
    GROUP BY value
    HAVING COUNT(*) >= ?
    ORDER BY COUNT(*) DESC
    """
    
    cursor.execute(query, (min_count,))
    # Format as list of tuples (intent_text, count)
    intents = [(row['intent_text'], row['count']) for row in cursor.fetchall()]
    
    # Close connection
    conn.close()
    
    return intents

async def consolidate_intents(db_path, api_key, min_count=1, max_groups=100, batch_size=None, debug=False, output=None, force_target=False, reduction_factor=0.5):
    """
    Consolidate intents from the database using the Categorizer.
    
    Args:
        db_path: Path to the SQLite database
        api_key: API key for the Categorizer
        min_count: Minimum count to include an intent (default=1)
        max_groups: Maximum number of consolidated groups to create (default=100)
        batch_size: Optional batch size for processing
        debug: Enable debug mode for more verbose output
        output: Optional path to save results as CSV
        force_target: Force consolidation to reach target number of groups
        reduction_factor: Controls aggressiveness of consolidation (0.1-0.9, lower is more aggressive)
        
    Returns:
        Dictionary with consolidation results
    """
    # Initialize the Categorizer
    categorizer = Categorizer(api_key=api_key, debug=debug)
    
    # Extract intent distribution from the database
    intent_distribution = await extract_intent_distribution(db_path, min_count)
    
    print(f"Found {len(intent_distribution)} unique intents in the database")
    
    # Consolidate the intents using the Categorizer's iterative method
    start_time = time.time()
    consolidated_mapping = await categorizer.process_labels_iteratively(
        value_distribution=intent_distribution,
        target_groups=max_groups,
        batch_size=batch_size,
        force_target=force_target,
        reduction_factor=reduction_factor
    )
    
    # Calculate statistics
    original_count = len(intent_distribution)
    consolidated_count = len(set(consolidated_mapping.values()))
    
    # Create results dictionary
    results = {
        "original_count": original_count,
        "consolidated_count": consolidated_count,
        "reduction_percentage": (1 - consolidated_count / original_count) * 100 if original_count > 0 else 0,
        "mapping": consolidated_mapping
    }
    
    # Save results to CSV if requested
    if output:
        with open(output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['original_label', 'consolidated_label'])
            for original, consolidated in consolidated_mapping.items():
                writer.writerow([original, consolidated])
        print(f"\nResults saved to {output}")
    
    return results

async def main():
    """Parse arguments and run the script."""
    parser = argparse.ArgumentParser(description='Consolidate intents from the database into semantic groups')
    parser.add_argument('--db', required=True, help='Path to the SQLite database')
    parser.add_argument('--min-count', type=int, default=1, help='Minimum count to include an intent (default=1)')
    parser.add_argument('--max-groups', type=int, default=100, help='Maximum number of consolidated groups (default=100)')
    parser.add_argument('--batch-size', type=int, help='Optional batch size for processing')
    parser.add_argument('--output', help='Optional path to save results as CSV')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--force-target', action='store_true', help='Force consolidation to reach target number of groups')
    parser.add_argument('--reduction-factor', type=float, default=0.5, 
                        help='Controls consolidation aggressiveness (0.1-0.9, lower is more aggressive)')
    parser.add_argument('--show-all', action='store_true', help='Show all consolidated groups')
    
    args = parser.parse_args()
    
    # Validate reduction factor
    if args.reduction_factor < 0.1 or args.reduction_factor > 0.9:
        print("Error: reduction-factor must be between 0.1 and 0.9")
        return
    
    # Get API key from environment variable
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set")
        return
    
    start_time = time.time()
    results = await consolidate_intents(
        db_path=args.db,
        api_key=api_key,
        min_count=args.min_count,
        max_groups=args.max_groups,
        batch_size=args.batch_size,
        debug=args.debug,
        output=args.output,
        force_target=args.force_target,
        reduction_factor=args.reduction_factor
    )
    
    # Print results
    print(f"\nConsolidation Results:")
    print(f"Original unique intents: {results['original_count']}")
    print(f"Consolidated groups: {results['consolidated_count']}")
    print(f"Reduction: {results['reduction_percentage']:.1f}%")
    
    # Organize consolidated groups and calculate total counts for each group
    consolidated_groups = {}
    group_total_counts = {}
    
    # First, group original intents by their consolidated label
    for original, consolidated in results['mapping'].items():
        if consolidated not in consolidated_groups:
            consolidated_groups[consolidated] = []
            group_total_counts[consolidated] = 0
        
        consolidated_groups[consolidated].append(original)
        
        # Find the count for this original intent
        for intent, count in await extract_intent_distribution(args.db, args.min_count):
            if intent == original:
                group_total_counts[consolidated] += count
                break
    
    # Sort groups by their total count (descending)
    sorted_groups = sorted(consolidated_groups.items(), 
                          key=lambda x: group_total_counts.get(x[0], 0), 
                          reverse=True)
    
    if args.show_all:
        print("\nAll Consolidated Groups (sorted by frequency):")
        for i, (group, members) in enumerate(sorted_groups):
            total_count = group_total_counts.get(group, 0)
            print(f"\n{i+1}. {group} (Total: {total_count})")
            for member in members:
                # Find the count for this original intent
                member_count = 0
                for intent, count in await extract_intent_distribution(args.db, args.min_count):
                    if intent == member:
                        member_count = count
                        break
                print(f"  - {member} ({member_count})")
    else:
        print("\nExample consolidated groups:")
        # Print up to 5 groups with their members
        for i, (group, members) in enumerate(sorted_groups):
            if i >= 5:
                break
            total_count = group_total_counts.get(group, 0)
            print(f"\nGroup: {group} (Total: {total_count})")
            for j, member in enumerate(members):
                if j >= 5:  # Print up to 5 members per group
                    print(f"  ... and {len(members) - 5} more")
                    break
                
                # Find the count for this original intent
                member_count = 0
                for intent, count in await extract_intent_distribution(args.db, args.min_count):
                    if intent == member:
                        member_count = count
                        break
                        
                print(f"  - {member} ({member_count})")
    
    # Print timing information
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())