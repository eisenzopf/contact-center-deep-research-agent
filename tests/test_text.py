import pytest
from contact_center_analysis import TextGenerator
import os
import time

@pytest.fixture
def sample_questions():
    return [
        "When customers call to cancel their service, how often do agents try to save the customer?",
        "What are the most common reasons for cancellation?",
        "How effective are our retention offers?"
    ]

@pytest.fixture
def existing_attributes():
    return [
        "conversation_id",
        "customer_id",
        "agent_id",
        "transcript",
        "date_timestamp"
    ]

@pytest.fixture
def sample_text():
    """Sample conversation text for testing."""
    return """Agent: Thank you for calling customer support. How can I help you today?
Customer: Hi, I'm having trouble with my recent order. It hasn't arrived yet and it's been over a week.
Agent: I'm sorry to hear that. Let me look into this for you. Can I have your order number please?
Customer: Yes, it's ABC123456.
Agent: Thank you. I see your order was shipped 3 days ago and is currently in transit. According to the tracking information, it should be delivered by tomorrow.
Customer: That's good to know. I was worried because the confirmation email said it would arrive within 3-5 business days.
Agent: I understand your concern. There was a slight delay in our warehouse, but your order is on its way now. Would you like me to send you the tracking link?
Customer: Yes, please. That would be helpful.
Agent: I've just sent the tracking link to your email. Is there anything else I can help you with today?
Customer: No, that's all. Thank you for your help.
Agent: You're welcome. Thank you for your patience, and please don't hesitate to contact us if you have any other questions."""

@pytest.fixture
def multiple_attributes():
    """Multiple attributes for testing."""
    return [
        {
            "field_name": "order_status",
            "title": "Order Status",
            "description": "The current status of the customer's order (e.g., processing, shipped, delivered, delayed)."
        },
        {
            "field_name": "customer_issue",
            "title": "Customer Issue",
            "description": "The main problem or concern raised by the customer during the conversation."
        },
        {
            "field_name": "resolution_offered",
            "title": "Resolution Offered",
            "description": "The solution or resolution offered by the agent to address the customer's issue."
        }
    ]

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_generate_required_attributes(sample_questions, existing_attributes, llm_debug):
    """Test generation of required attributes for analysis questions."""
    
    generator = TextGenerator(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    result = await generator.generate_required_attributes(
        questions=sample_questions,
        existing_attributes=existing_attributes
    )
    
    # Verify response structure
    assert "attributes" in result
    assert isinstance(result["attributes"], list)
    assert len(result["attributes"]) > 0
    
    # Verify each attribute has required fields
    for attribute in result["attributes"]:
        assert "field_name" in attribute
        assert "title" in attribute
        assert "description" in attribute
        assert "rationale" in attribute
        
        # Verify field_name is in snake_case
        assert "_" in attribute["field_name"] or attribute["field_name"].islower()

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_generate_required_attributes_no_existing(sample_questions, llm_debug):
    """Test attribute generation without existing attributes."""
    
    generator = TextGenerator(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    result = await generator.generate_required_attributes(
        questions=sample_questions
    )
    
    assert "attributes" in result
    assert isinstance(result["attributes"], list)
    assert len(result["attributes"]) > 0

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_generate_required_attributes_empty_questions(llm_debug):
    """Test handling of empty questions list."""
    
    generator = TextGenerator(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    result = await generator.generate_required_attributes(
        questions=[]
    )
    
    assert "attributes" in result
    assert isinstance(result["attributes"], list)
    assert len(result["attributes"]) == 0

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_generate_required_attributes_batch(sample_questions, existing_attributes, llm_debug):
    """Test batch processing of required attributes generation."""
    
    generator = TextGenerator(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    # Create multiple sets of questions
    questions_sets = [
        sample_questions,
        sample_questions[:2],  # Subset of questions
        sample_questions + ["What is our average response time?"]  # Extended set
    ]
    
    results = await generator.generate_required_attributes_batch(
        questions_sets=questions_sets,
        existing_attributes=existing_attributes,
        batch_size=2  # Process 2 sets at a time
    )
    
    # Verify results
    assert len(results) == len(questions_sets)
    for result in results:
        assert "attributes" in result
        assert isinstance(result["attributes"], list)
        assert len(result["attributes"]) > 0
        
        # Verify each attribute
        for attribute in result["attributes"]:
            assert "field_name" in attribute
            assert "title" in attribute
            assert "description" in attribute
            assert "rationale" in attribute
            assert "_" in attribute["field_name"] or attribute["field_name"].islower()

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_generate_attributes(sample_text, multiple_attributes, llm_debug):
    """Test generation of multiple attributes in a single call."""
    
    generator = TextGenerator(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    results = await generator.generate_attributes(
        text=sample_text,
        attributes=multiple_attributes
    )
    
    # Verify response structure
    assert isinstance(results, list)
    assert len(results) == len(multiple_attributes)
    
    # Verify each attribute result has required fields
    for result in results:
        assert "field_name" in result
        assert "value" in result
        assert "confidence" in result
        assert "explanation" in result
        
        # Verify field_name matches one of the input attributes
        assert any(attr["field_name"] == result["field_name"] for attr in multiple_attributes)
        
        # Verify confidence is a float between 0 and 1
        assert 0 <= result["confidence"] <= 1
        
        # Verify explanation is not empty
        assert len(result["explanation"]) > 0
