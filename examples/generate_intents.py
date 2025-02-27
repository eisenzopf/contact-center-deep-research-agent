import sqlite3
import asyncio
import sys
import os
from contact_center_analysis.text import TextGenerator

async def process_conversations_without_intent(db_path, api_key, test_mode=False):
    """
    Process conversations that don't have an intent attribute and generate one.
    
    Args:
        db_path: Path to the SQLite database
        api_key: API key for the TextGenerator
        test_mode: If True, only process 2 conversations and don't modify the database
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
    
    print(f"Found {len(conversations)} conversations without intent attributes")
    
    # Limit to 2 conversations in test mode
    if test_mode:
        conversations = conversations[:2]
        print("TEST MODE: Processing only 2 conversations without database updates")
    
    # Process each conversation
    for conversation in conversations:
        conversation_id = conversation['conversation_id']
        text = conversation['text']
        
        print(f"Processing conversation: {conversation_id}")
        
        # Generate intent using the TextGenerator
        try:
            intent_data = await text_generator.generate_intent(text)
            
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
                
                print(f"  Added intent: {intent_data['label_name']}")
            
        except Exception as e:
            print(f"  Error processing conversation {conversation_id}: {str(e)}")
    
    # Commit changes and close connection (only if not in test mode)
    if not test_mode:
        conn.commit()
    conn.close()
    
    if test_mode:
        print("Test processing complete - no database changes were made")
    else:
        print("Processing complete")

async def main():
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python script.py <database_path> [--test]")
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    # Check for test mode flag
    test_mode = "--test" in sys.argv
    
    # Get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Verify database exists
    if not os.path.exists(db_path):
        print(f"Error: Database file not found: {db_path}")
        sys.exit(1)
    
    await process_conversations_without_intent(db_path, api_key, test_mode)

if __name__ == "__main__":
    asyncio.run(main())