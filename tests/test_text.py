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
