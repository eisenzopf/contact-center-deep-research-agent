import sqlite3
import asyncio
import sys
import os
from contact_center_analysis.text import TextGenerator
import argparse
import json
import time
import re

def fix_json_escapes(json_str):
    """Fix common JSON escape issues, particularly with dollar signs."""
    # Replace escaped dollar signs with unescaped ones
    fixed_str = json_str.replace('\\$', '$')
    return fixed_str

async def process_conversations_without_intent(db_path, api_key, test_mode=False, mini_test=False):
    """
    Process conversations that don't have an intent attribute and generate one.
    
    Args:
        db_path: Path to the SQLite database
        api_key: API key for the TextGenerator
        test_mode: If True, only process 50 conversations and don't modify the database
        mini_test: If True, only process 2 conversations and update the database
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Initialize the TextGenerator
    text_generator = TextGenerator(api_key=api_key)
    
    # Find conversations without intent attributes
    query = """
    SELECT c.conversation_id, c.text 
    FROM conversations c
    WHERE NOT EXISTS (
        SELECT 1 
        FROM conversation_attributes ca 
        WHERE ca.conversation_id = c.conversation_id 
        AND ca.type = 'intent'
    )
    """
    
    cursor.execute(query)
    conversations = cursor.fetchall()
    
    total_conversations = len(conversations)
    print(f"Found {total_conversations} conversations without intent attributes")
    
    # Apply appropriate limits based on test mode
    if test_mode:
        conversations = conversations[:50]
        print("TEST MODE: Processing 50 conversations without database updates")
    elif mini_test:
        conversations = conversations[:2]
        print("MINI TEST MODE: Processing only 2 conversations with database updates")
    
    # Format conversations for batch processing
    formatted_conversations = [
        {"id": conv["conversation_id"], "text": conv["text"]} 
        for conv in conversations
    ]
    
    # Process conversations in batches
    start_time = time.time()
    processed_count = 0
    error_count = 0
    
    # Process in smaller batches to show progress
    batch_size = 50
    for i in range(0, len(formatted_conversations), batch_size):
        batch = formatted_conversations[i:i + batch_size]
        
        try:
            # Create a custom method to process each conversation with error handling
            async def safe_process_conversation(conversation):
                try:
                    conv_text = conversation.get('text', '')
                    conv_id = conversation.get('id', '')
                    
                    intent = await text_generator.generate_intent(text=conv_text)
                    
                    return {
                        "conversation_id": conv_id,
                        "intent": intent
                    }
                except Exception as e:
                    if "Invalid \\escape" in str(e) and "\\$" in str(e):
                        # Extract the JSON from the error message
                        error_msg = str(e)
                        json_start = error_msg.find('{')
                        json_end = error_msg.rfind('}') + 1
                        
                        if json_start >= 0 and json_end > json_start:
                            json_str = error_msg[json_start:json_end]
                            fixed_json = fix_json_escapes(json_str)
                            
                            try:
                                intent_data = json.loads(fixed_json)
                                return {
                                    "conversation_id": conv_id,
                                    "intent": intent_data
                                }
                            except:
                                pass
                    
                    print(f"Error processing conversation {conv_id}: {str(e)}")
                    return None
            
            # Process the batch with our custom function
            tasks = [safe_process_conversation(conv) for conv in batch]
            batch_results = await asyncio.gather(*tasks)
            
            # Process results
            for result in batch_results:
                if not result:  # Skip None results (failed processing)
                    error_count += 1
                    processed_count += 1
                    continue
                    
                conversation_id = result["conversation_id"]
                intent_data = result["intent"]
                
                processed_count += 1
                
                try:
                    if test_mode:
                        # Just print the results in test mode
                        print(f"  Generated intent for conversation {conversation_id}:")
                        print(f"    Label: {intent_data['label']}")
                        print(f"    Label Name: {intent_data['label_name']}")
                        print(f"    Description: {intent_data['description']}")
                    else:
                        # Insert the intent data into the conversation_attributes table
                        insert_query = """
                        INSERT INTO conversation_attributes 
                        (conversation_id, name, type, value, description, title)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """
                        
                        cursor.execute(
                            insert_query, 
                            (
                                conversation_id,
                                intent_data['label'],         # name
                                'intent',                     # type
                                intent_data['label_name'],    # value
                                intent_data['description'],   # description
                                intent_data['label_name']     # title
                            )
                        )
                    
                except Exception as e:
                    error_count += 1
                    print(f"  Error processing conversation {conversation_id}: {str(e)}")
        
        except Exception as e:
            error_count += len(batch)
            processed_count += len(batch)
            print(f"Error processing batch: {str(e)}")
            
        # Commit after each batch if not in test mode
        if not test_mode:
            conn.commit()
        
        # Show progress after every 100 conversations or at the end of each batch
        if processed_count % 100 == 0 or processed_count == len(formatted_conversations):
            elapsed_time = time.time() - start_time
            conversations_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
            
            print(f"Progress: {processed_count}/{len(formatted_conversations)} conversations processed "
                  f"({processed_count/len(formatted_conversations)*100:.1f}%) - "
                  f"{conversations_per_second:.2f} conversations/second - "
                  f"Errors: {error_count}")
    
    # Close connection
    conn.close()
    
    # Final message
    if test_mode:
        print("Test processing complete - no database changes were made")
    elif mini_test:
        print("Mini test complete - database updated with 2 conversations")
    else:
        print("Processing complete")
        
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average processing speed: {processed_count/total_time:.2f} conversations/second")
    print(f"Total errors: {error_count} ({error_count/processed_count*100:.1f}%)")

async def main():
    """Parse arguments and run the script."""
    parser = argparse.ArgumentParser(description='Generate intent attributes for conversations')
    parser.add_argument('--db', required=True, help='Path to the SQLite database')
    parser.add_argument('--test', action='store_true', help='Run in test mode (no database changes)')
    parser.add_argument('--mini-test', action='store_true', help='Process only 2 conversations and update the database')
    
    args = parser.parse_args()
    
    if args.test and args.mini_test:
        print("Error: Cannot specify both --test and --mini-test")
        return
    
    # Get API key from environment variable
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set")
        return
    
    await process_conversations_without_intent(
        db_path=args.db,
        api_key=api_key,
        test_mode=args.test,
        mini_test=args.mini_test
    )

if __name__ == "__main__":
    asyncio.run(main())