import pytest
from contact_center_analysis import TextGenerator, AttributeMatcher
import os
import json

@pytest.fixture
def sample_questions():
    return [
        "How often do customers mention pricing concerns?",
        "What are the most common product features requested?",
        "How satisfied are customers with our support response times?"
    ]

@pytest.fixture
def sample_available_attributes():
    return [
        {
            "name": "price_feedback",
            "description": json.dumps({
                "title": "Price Related Comments",
                "description": "Customer feedback specifically about pricing and costs"
            })
        },
        {
            "name": "feature_requests",
            "description": json.dumps({
                "title": "Requested Features",
                "description": "Product features and improvements requested by customers"
            })
        },
        {
            "name": "support_satisfaction",
            "description": json.dumps({
                "title": "Support Satisfaction Score",
                "description": "Customer satisfaction rating for support interactions"
            })
        },
        {
            "name": "response_time",
            "description": json.dumps({
                "title": "Support Response Time",
                "description": "Time taken to respond to customer support requests"
            })
        }
    ]

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_basic_attribute_matching(sample_questions, sample_available_attributes, llm_debug):
    """Test basic matching between required and available attributes."""
    
    # First generate required attributes using TextGenerator
    text_gen = TextGenerator(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    required_attrs = await text_gen.generate_required_attributes(
        questions=sample_questions
    )
    
    # Now use those required attributes for matching
    matcher = AttributeMatcher(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    # Use a lower confidence threshold since semantic matches might not be exact
    matches, missing = await matcher.find_matches(
        required_attributes=required_attrs['attributes'],
        available_attributes=sample_available_attributes,
        confidence_threshold=0.6  # Lower threshold to catch semantic matches
    )
    
    # Verify we got some matches
    assert len(matches) > 0
    # Verify the structure of matches
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in matches.items())

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_partial_matches(sample_available_attributes, llm_debug):
    """Test matching when some required attributes aren't available."""
    
    # Generate required attributes for questions that won't all have matches
    text_gen = TextGenerator(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    required_attrs = await text_gen.generate_required_attributes(
        questions=[
            "How often do customers mention pricing concerns?",
            "What percentage of customers are using our mobile app?",  # No matching attribute
            "How satisfied are customers with our support response times?"
        ]
    )
    
    matcher = AttributeMatcher(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    matches, missing = await matcher.find_matches(
        required_attributes=required_attrs['attributes'],
        available_attributes=sample_available_attributes
    )
    
    # Verify we have both matches and missing attributes
    assert len(matches) > 0
    assert len(missing) > 0
    # Verify missing attributes contain the mobile app related field
    assert any('mobile' in attr['field_name'].lower() or 
              'mobile' in attr['description'].lower() 
              for attr in missing)

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_confidence_threshold(sample_questions, sample_available_attributes, llm_debug):
    """Test that matches respect the confidence threshold."""
    
    text_gen = TextGenerator(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    required_attrs = await text_gen.generate_required_attributes(
        questions=sample_questions
    )
    
    matcher = AttributeMatcher(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    # Test with a very high confidence threshold
    matches, missing = await matcher.find_matches(
        required_attributes=required_attrs['attributes'],
        available_attributes=sample_available_attributes,
        confidence_threshold=0.9  # Very high threshold
    )
    
    # Test with a lower confidence threshold
    matches_lenient, missing_lenient = await matcher.find_matches(
        required_attributes=required_attrs['attributes'],
        available_attributes=sample_available_attributes,
        confidence_threshold=0.5  # Lower threshold
    )
    
    # Verify we get more matches with lower threshold
    assert len(matches_lenient) >= len(matches)

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_find_matches_batch(sample_questions, sample_available_attributes, llm_debug):
    """Test batch processing of attribute matching."""
    
    text_gen = TextGenerator(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    required_attrs = await text_gen.generate_required_attributes(
        questions=sample_questions
    )
    
    matcher = AttributeMatcher(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    results = await matcher.find_matches_batch(
        required_attributes=required_attrs['attributes'],
        available_attributes=sample_available_attributes,
        batch_size=2,  # Test with small batch size
        confidence_threshold=0.7
    )
    
    # Verify results structure
    assert isinstance(results, list)
    assert len(results) > 0
    
    for match in results:
        assert "required_field" in match
        assert "matched_attribute" in match or match["matched_attribute"] is None
        assert "confidence" in match
        assert 0 <= match["confidence"] <= 1

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_find_matches_batch_empty(sample_available_attributes, llm_debug):
    """Test batch matching with empty required attributes."""
    
    matcher = AttributeMatcher(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    results = await matcher.find_matches_batch(
        required_attributes=[],
        available_attributes=sample_available_attributes
    )
    
    assert len(results) == 0 