# app.py
import os
import json
import time
import asyncio
import argparse
import sqlite3
import logging
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
from threading import Thread

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set matplotlib to use a non-interactive backend before importing any visualization functions
import matplotlib
matplotlib.use('Agg')  # Use non-interactive Agg backend

# Import analysis functions from analyze_fee_disputes.py
try:
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
        refine_analysis_results,
        validate_and_boost_confidence
    )
    logger.info("Successfully imported functions from analyze_fee_disputes.py")
except ImportError as e:
    logger.error(f"Error importing functions: {str(e)}")
    # Define fallback functions
    async def get_required_attributes(*args, **kwargs):
        logger.warning("Using fallback get_required_attributes function")
        return [{"field_name": "dispute_type", "title": "Dispute Type", "description": "Type of fee dispute"}]

# Patch for rate limiting - we'll add this to the BaseAnalyzer class
# This will be applied when we create the first analyzer instance
def patch_rate_limiting():
    try:
        # Import the module that contains the rate limiting code
        from contact_center_analysis.base import BaseAnalyzer
        
        # Get the original _generate_content method
        original_generate_content = BaseAnalyzer._generate_content
        
        # Define a patched version that handles rate limiting errors
        async def patched_generate_content(self, prompt, **kwargs):
            try:
                return await original_generate_content(self, prompt, **kwargs)
            except KeyError as e:
                # If it's a rate limiting bucket error, initialize the bucket and retry
                if str(e).isdigit() or str(e).startswith("'") and str(e)[1:-1].isdigit():
                    logger.warning(f"Rate limiting bucket {e} not found, initializing and retrying")
                    # Initialize the bucket in the LLM instance
                    self.llm.buckets[int(str(e).strip("'"))] = 0
                    # Retry the request
                    return await original_generate_content(self, prompt, **kwargs)
                else:
                    # Re-raise other KeyErrors
                    raise
        
        # Apply the patch
        BaseAnalyzer._generate_content = patched_generate_content
        logger.info("Successfully patched rate limiting in BaseAnalyzer")
    except Exception as e:
        logger.error(f"Failed to patch rate limiting: {str(e)}")

# Apply the patch
patch_rate_limiting()

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this in production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
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
    logger.info(f"Progress: {message}")
    socketio.emit('progress', {'message': message})

# Add this socket event handler
@socketio.on('start_analysis')
def handle_start_analysis(data):
    """Handle start_analysis event from client."""
    logger.info(f"Received start_analysis event with data: {data}")
    questions = data.get('questions', '')
    if not questions:
        socketio.emit('error', {'message': 'No questions provided'})
        return
    
    # Generate a unique ID for this analysis
    analysis_id = f"analysis_{int(time.time())}"
    logger.info(f"Starting new analysis with ID: {analysis_id}")
    
    # Start analysis in a separate thread
    thread = Thread(
        target=run_async_analysis, 
        args=(analysis_id, questions, db_path, sample_size, batch_size),
        daemon=True
    )
    thread.start()
    
    socketio.emit('progress', {'message': f"Analysis started with ID: {analysis_id}"})

async def extract_conversations_from_db(db_path, required_attributes, sample_size, batch_size):
    """Fallback function to extract conversations from the database."""
    try:
        logger.info(f"Extracting conversations from {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get a sample of conversations
        cursor.execute("SELECT conversation_id, text FROM conversations LIMIT ?", (sample_size,))
        conversations = [{"id": row[0], "text": row[1]} for row in cursor.fetchall()]
        
        conn.close()
        logger.info(f"Extracted {len(conversations)} conversations")
        return conversations
    except Exception as e:
        logger.error(f"Error extracting conversations: {str(e)}")
        return []

def run_async_analysis(analysis_id, questions, db_path, sample_size, batch_size):
    """Run the analysis in a separate thread."""
    try:
        logger.info(f"Starting analysis {analysis_id} with questions: {questions}")
        asyncio.run(run_analysis(analysis_id, questions, db_path, sample_size, batch_size))
    except Exception as e:
        logger.error(f"Error in run_async_analysis: {str(e)}")
        socketio.emit('error', {'message': str(e)})

async def run_analysis(analysis_id, questions, db_path, sample_size, batch_size):
    """Run the analysis asynchronously."""
    try:
        start_time = time.time()
        logger.info(f"Starting analysis run for {analysis_id}")
        
        # Step 1: Extract required attributes from questions
        emit_progress("Step 1/12: Analyzing questions to determine required attributes...")
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            error_msg = "GEMINI_API_KEY environment variable is not set"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Convert questions string to a list of questions
        question_list = [q.strip() for q in questions.split('\n') if q.strip()]
        required_attributes = await get_required_attributes(question_list, api_key, debug=False)
        
        # Deduplicate required attributes by field_name
        unique_attributes = {}
        for attr in required_attributes:
            field_name = attr['field_name']
            if field_name not in unique_attributes:
                unique_attributes[field_name] = attr
        
        required_attributes = list(unique_attributes.values())
        required_attr_names = [attr['field_name'] for attr in required_attributes]
        emit_progress(f"Required attributes: {', '.join(required_attr_names)}")
        
        # Step 2: Find matching intents for fee disputes
        emit_progress(f"Step 2/12: Finding intents related to fee disputes...")
        matching_intents = await find_matching_intents(
            db_path=db_path,
            api_key=api_key,
            target_class="fee dispute",
            examples=None,
            debug=False
        )
        
        emit_progress(f"Found {len(matching_intents)} matching intents for fee disputes")
        
        # Step 3: Fetch conversations with matching intents
        emit_progress(f"Step 3/12: Fetching {sample_size} conversations with fee dispute intents...")
        conversations = await fetch_conversations_by_intents(
            db_path=db_path,
            matching_intents=matching_intents,
            limit=sample_size
        )
        
        emit_progress(f"Found {len(conversations)} conversations for analysis")
        
        if not conversations:
            error_msg = "No conversations found for analysis. Please check your database."
            logger.error(error_msg)
            socketio.emit('error', {'message': error_msg})
            return
        
        # Step 4: Generate attribute values for all conversations
        emit_progress(f"Step 4/12: Generating attribute values for all {len(conversations)} conversations...")
        attribute_results = await generate_attributes_for_conversations(
            conversations,
            required_attributes,
            api_key,
            False,
            batch_size=batch_size,
            max_retries=3
        )
        
        # Step 5: Compile statistics on attribute values
        emit_progress("Step 5/12: Compiling attribute statistics with semantic grouping...")
        statistics = await compile_attribute_statistics(
            attribute_results,
            api_key,
            False
        )
        
        # Step 6: Improve attribute consolidation
        emit_progress("Step 6/12: Improving attribute consolidation...")
        attribute_results, statistics = await improve_attribute_consolidation(
            attribute_results=attribute_results,
            statistics=statistics,
            api_key=api_key,
            debug=False
        )
        
        # Step 7: Recompile statistics after consolidation
        emit_progress("Step 7/12: Recompiling statistics after consolidation...")
        statistics = await compile_attribute_statistics(
            attribute_results,
            api_key,
            False
        )
        
        # Step 8: Extract detailed insights
        emit_progress("Step 8/12: Extracting detailed insights from attribute values...")
        question_list = [q.strip() for q in questions.split('\n') if q.strip()]
        detailed_insights = await review_attribute_details(
            attribute_results,
            required_attributes,
            question_list,
            api_key,
            False
        )
        
        # Step 9: Analyze findings to answer research questions
        emit_progress("Step 9/12: Analyzing findings to answer research questions...")
        analysis = await analyze_attribute_findings(
            statistics,
            question_list,
            api_key,
            False,
            detailed_insights
        )
        
        # Step 10: Analyze relationships between attributes
        emit_progress("Step 10/12: Analyzing relationships between attributes...")
        relationships = await analyze_attribute_relationships(
            statistics,
            attribute_results,
            api_key,
            False
        )
        
        if relationships:
            analysis["attribute_relationships"] = relationships
        
        # Step 11: Resolve data gaps dynamically
        emit_progress("Step 11/12: Checking for data gaps and enhancing analysis...")
        enhanced_attribute_results = await resolve_data_gaps_dynamically(
            analysis=analysis,
            attribute_results=attribute_results,
            api_key=api_key,
            debug=False
        )
        
        # Step 12: Recompile statistics and re-analyze with enhanced data
        emit_progress("Step 12/12: Finalizing analysis with enhanced data...")
        enhanced_statistics = await compile_attribute_statistics(
            attribute_results=enhanced_attribute_results,
            api_key=api_key,
            debug=False
        )
        
        enhanced_analysis = await analyze_attribute_findings(
            statistics=enhanced_statistics,
            questions=question_list,
            api_key=api_key,
            debug=False,
            detailed_insights=detailed_insights
        )
        
        # Add this after the initial analysis but before finalizing
        emit_progress("Refining analysis results for higher confidence...")
        analysis = await refine_analysis_results(
            analysis,
            statistics,
            api_key,
            False
        )
        
        # Add this before storing results
        emit_progress("Boosting confidence through additional validation...")
        enhanced_analysis = await validate_and_boost_confidence(
            enhanced_analysis,
            enhanced_statistics,
            api_key,
            False
        )
        
        # Store results in cache
        visualization_dir = os.path.join(app.config['STATIC_FOLDER'], 'visualizations', analysis_id)
        os.makedirs(visualization_dir, exist_ok=True)
        
        analysis_cache[analysis_id] = {
            "statistics": enhanced_statistics,
            "analysis": enhanced_analysis,
            "visualization_dir": visualization_dir
        }
        
        elapsed_time = time.time() - start_time
        emit_progress(f"Analysis completed in {elapsed_time:.2f} seconds")
        socketio.emit('analysis_complete', {'analysis_id': analysis_id})
        
    except Exception as e:
        import traceback
        error_message = f"Error during analysis: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        socketio.emit('error', {'message': str(e)})

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
    
    return render_template(
        'results.html',
        analysis_id=analysis_id,
        analysis=analysis_data.get("analysis", {}),
        statistics=analysis_data.get("statistics", {})
    )

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """API endpoint to start analysis."""
    questions = request.form.get('questions', '')
    if not questions:
        return jsonify({'error': 'No questions provided'}), 400
    
    # Generate a unique ID for this analysis
    analysis_id = f"analysis_{int(time.time())}"
    logger.info(f"Starting new analysis with ID: {analysis_id}")
    
    # Start analysis in a separate thread
    thread = Thread(
        target=run_async_analysis, 
        args=(analysis_id, questions, db_path, sample_size, batch_size),
        daemon=True
    )
    thread.start()
    
    return jsonify({'analysis_id': analysis_id})

@app.route('/api/config', methods=['POST'])
def update_config():
    """API endpoint to update configuration."""
    global db_path, sample_size, batch_size
    
    if 'db_path' in request.form:
        db_path = request.form['db_path']
        logger.info(f"Updated db_path to {db_path}")
    
    if 'sample_size' in request.form:
        try:
            sample_size = int(request.form['sample_size'])
            logger.info(f"Updated sample_size to {sample_size}")
        except ValueError:
            pass
    
    if 'batch_size' in request.form:
        try:
            batch_size = int(request.form['batch_size'])
            logger.info(f"Updated batch_size to {batch_size}")
        except ValueError:
            pass
    
    return jsonify({
        'db_path': db_path,
        'sample_size': sample_size,
        'batch_size': batch_size
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Flask app for fee dispute analysis')
    parser.add_argument('--db_path', type=str, help='Path to the SQLite database')
    parser.add_argument('--sample_size', type=int, default=100, help='Sample size for analysis')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    
    args = parser.parse_args()
    
    if args.db_path:
        db_path = args.db_path
        logger.info(f"Using database: {db_path}")
    if args.sample_size:
        sample_size = args.sample_size
    if args.batch_size:
        batch_size = args.batch_size
    
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'visualizations'), exist_ok=True)
    
    logger.info(f"Starting Flask app on port {args.port}")
    socketio.run(app, debug=True, port=args.port)