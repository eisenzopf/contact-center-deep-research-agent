# app.py
import os
import json
import time
import asyncio
import argparse
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
from threading import Thread

# Import the core functions from the original script
from examples.analyze_fee_disputes import (
    get_required_attributes,
    find_matching_intents,
    fetch_conversations_by_intents,
    generate_attributes_for_conversations,
    compile_attribute_statistics,
    improve_attribute_consolidation,
    review_attribute_details,
    analyze_attribute_findings,
    analyze_attribute_relationships,
    resolve_data_gaps_dynamically,
    generate_visualizations,
    generate_sentiment_flow,
    generate_resolution_sankey,
    generate_action_wordcloud
)

app = Flask(__name__)
app.secret_key = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*")

# Store analysis results in memory (for demo purposes)
# In production, you might want to use a database
analysis_cache = {}

# Global variables to store configuration
db_path = None
sample_size = 100  # Default sample size
batch_size = 10    # Default batch size

def emit_progress(message):
    """Emit progress message to the client."""
    socketio.emit('progress', {'message': message})

async def run_analysis(questions, max_retries=3):
    """Run the analysis process with the given questions."""
    global db_path, sample_size, batch_size
    
    if not db_path:
        emit_progress("Error: Database path not provided")
        return None
        
    analysis_id = str(int(time.time()))
    
    # Get API key from environment variable
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        emit_progress("Error: GEMINI_API_KEY environment variable is not set")
        return None
    
    start_time = time.time()
    
    # Step 1: Get required attributes
    emit_progress("Step 1: Identifying required attributes...")
    required_attributes = await get_required_attributes(questions, api_key, debug=True)
    
    # Deduplicate required attributes by field_name
    unique_attributes = {}
    for attr in required_attributes:
        field_name = attr['field_name']
        if field_name not in unique_attributes:
            unique_attributes[field_name] = attr
    
    required_attributes = list(unique_attributes.values())
    
    attr_message = f"Identified {len(required_attributes)} required attributes:"
    for attr in required_attributes:
        attr_message += f"\n  - {attr['title']} ({attr['field_name']}): {attr['description']}"
    emit_progress(attr_message)
    
    # Step 2: Find matching intents for fee disputes
    emit_progress("Step 2: Finding intents related to 'fee dispute'...")
    matching_intents = await find_matching_intents(
        db_path=db_path,
        api_key=api_key,
        target_class="fee dispute",
        examples=None,
        debug=True
    )
    
    if not matching_intents:
        emit_progress("No matching intents found. Using default intent.")
        matching_intents = ["fee_dispute"]
    
    intent_message = f"Found {len(matching_intents)} matching intents:"
    for intent in matching_intents[:5]:  # Show just the first 5 to avoid flooding
        intent_message += f"\n  - {intent}"
    if len(matching_intents) > 5:
        intent_message += f"\n  - ... and {len(matching_intents) - 5} more"
    emit_progress(intent_message)
    
    # Step 3: Fetch conversations by intents
    emit_progress(f"Step 3: Fetching up to {sample_size} conversations with matching intents...")
    conversations = await fetch_conversations_by_intents(
        db_path=db_path,
        matching_intents=matching_intents,
        limit=sample_size
    )
    
    if not conversations:
        emit_progress("Error: No conversations found with the matching intents")
        return None
    
    emit_progress(f"Fetched {len(conversations)} conversations")
    
    # Step 4: Generate attributes for conversations
    emit_progress("Step 4: Generating attributes for conversations...")
    attribute_results = await generate_attributes_for_conversations(
        conversations=conversations,
        attributes=required_attributes,
        api_key=api_key,
        debug=True,
        batch_size=batch_size,
        max_retries=max_retries
    )
    
    # Step 5: Compile attribute statistics
    emit_progress("Step 5: Compiling attribute statistics...")
    statistics = await compile_attribute_statistics(
        attribute_results=attribute_results,
        api_key=api_key,
        debug=True
    )
    
    # Step 6: Improve attribute consolidation
    emit_progress("Step 6: Improving attribute consolidation...")
    attribute_results, statistics = await improve_attribute_consolidation(
        attribute_results=attribute_results,
        statistics=statistics,
        api_key=api_key,
        debug=True
    )
    
    # Step 7: Review attribute details
    emit_progress("Step 7: Reviewing attribute details...")
    detailed_insights = await review_attribute_details(
        attribute_results=attribute_results,
        required_attributes=required_attributes,
        questions=questions,
        api_key=api_key,
        debug=True
    )
    
    # Step 8: Analyze findings to answer research questions
    emit_progress("Step 8: Analyzing findings to answer research questions...")
    analysis = await analyze_attribute_findings(
        statistics=statistics,
        questions=questions,
        api_key=api_key,
        debug=True,
        detailed_insights=detailed_insights
    )
    
    # Step 9: Analyze attribute relationships
    emit_progress("Step 9: Analyzing attribute relationships...")
    relationship_analysis = await analyze_attribute_relationships(
        statistics=statistics,
        attribute_results=attribute_results,
        api_key=api_key,
        debug=True
    )
    
    # Add relationship analysis to the main analysis
    if "insights" not in analysis:
        analysis["insights"] = []
    
    analysis["insights"].extend(relationship_analysis.get("insights", []))
    
    # Resolve data gaps dynamically
    emit_progress("Step 10: Checking for data gaps and enhancing analysis...")
    enhanced_analysis = await resolve_data_gaps_dynamically(
        analysis=analysis,
        attribute_results=attribute_results,
        api_key=api_key,
        debug=True
    )
    
    # Generate visualizations
    emit_progress("Step 11: Generating visualizations...")
    visualization_dir = f"static/visualizations/{analysis_id}"
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Generate standard visualizations
    visualizations = await generate_visualizations(
        analysis_results=enhanced_analysis,
        statistics=statistics,
        attribute_results=attribute_results,
        output_dir=visualization_dir
    )
    
    # Generate additional visualizations
    sentiment_viz = await generate_sentiment_flow(
        attribute_results=attribute_results, 
        output_dir=visualization_dir,
        api_key=api_key,
        debug=True
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
        debug=True
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
        debug=True
    )
    if wordcloud_viz:
        visualizations.append({
            "type": "wordcloud",
            "filename": wordcloud_viz
        })
    
    # Print timing information
    elapsed_time = time.time() - start_time
    emit_progress(f"Total processing time: {elapsed_time:.2f} seconds")
    
    # Store results in cache
    analysis_cache[analysis_id] = {
        "required_attributes": required_attributes,
        "matching_intents": matching_intents,
        "statistics": statistics,
        "analysis": enhanced_analysis,
        "visualizations": visualizations,
        "visualization_dir": visualization_dir
    }
    
    return analysis_id

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', db_path=db_path)

@app.route('/results/<analysis_id>')
def results(analysis_id):
    """Render the results page."""
    if analysis_id not in analysis_cache:
        return "Analysis not found", 404
    
    analysis_data = analysis_cache[analysis_id]
    
    # Convert visualization paths to web-friendly paths
    for viz in analysis_data["visualizations"]:
        if "filename" in viz:
            # Extract just the filename from the path
            filename = os.path.basename(viz["filename"])
            viz["web_path"] = f"/static/visualizations/{analysis_id}/{filename}"
    
    return render_template(
        'results.html',
        analysis_id=analysis_id,
        analysis=analysis_data["analysis"],
        statistics=analysis_data["statistics"],
        visualizations=analysis_data["visualizations"]
    )

@socketio.on('start_analysis')
def handle_start_analysis(data):
    """Handle the start analysis event from the client."""
    questions_text = data.get('questions', '')
    
    # Split the text by newlines and filter out empty lines
    questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
    
    if not questions:
        emit('error', {'message': 'At least one question is required'})
        return
    
    # Start analysis in a background thread
    def run_async_analysis():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        analysis_id = loop.run_until_complete(run_analysis(
            questions=questions
        ))
        if analysis_id:
            emit('analysis_complete', {'analysis_id': analysis_id})
        else:
            emit('error', {'message': 'Analysis failed'})
    
    Thread(target=run_async_analysis).start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fee Dispute Analysis Web App')
    parser.add_argument('--db-path', required=True, help='Path to the SQLite database')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the web app on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--sample-size', type=int, default=100, help='Number of conversations to analyze')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of conversations to process in parallel')
    
    args = parser.parse_args()
    
    # Set the global variables
    db_path = args.db_path
    sample_size = args.sample_size
    batch_size = args.batch_size
    
    # Create static directory for visualizations
    os.makedirs('static/visualizations', exist_ok=True)
    
    print(f"Starting Fee Dispute Analysis Web App with database: {db_path}")
    print(f"Sample size: {sample_size}, Batch size: {batch_size}")
    socketio.run(app, host='0.0.0.0', port=args.port, debug=args.debug)