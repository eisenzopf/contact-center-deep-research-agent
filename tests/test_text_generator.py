import pytest
from contact_center_analysis import TextGenerator
import os

@pytest.fixture
def sample_text():
    return """Customer called about their subscription. They mentioned that the monthly cost of $49.99 
    was too expensive compared to competitors. When offered our retention discount of 20% off for 6 months, 
    they still decided to cancel. Primary reason given was price, but they also mentioned not using all 
    the premium features."""

@pytest.fixture
def sample_attribute():
    return {
        "field_name": "cancellation_reason",
        "title": "Cancellation Reason",
        "description": "The primary reason given by the customer for cancellation",
        "rationale": "Needed to track common causes of customer churn"
    }

@pytest.fixture
def multiple_attributes():
    return [
        {
            "field_name": "price_sentiment",
            "title": "Price Sentiment",
            "description": "Customer's sentiment regarding the product/service pricing",
            "rationale": "To understand price perception impact on decisions"
        },
        {
            "field_name": "retention_offer_accepted",
            "title": "Retention Offer Accepted",
            "description": "Whether the customer accepted any retention offers",
            "rationale": "To track effectiveness of retention strategies"
        }
    ]

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_generate_single_attribute(sample_text, sample_attribute, llm_debug):
    """Test generation of a single attribute value from text."""
    
    generator = TextGenerator(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    result = await generator.generate_attribute(
        text=sample_text,
        attribute=sample_attribute
    )
    
    # Verify response structure
    assert "value" in result
    assert "confidence" in result
    assert "explanation" in result
    
    # Verify data types
    assert isinstance(result["value"], str)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["explanation"], str)
    
    # Verify confidence is between 0 and 1
    assert 0 <= result["confidence"] <= 1

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_generate_multiple_attributes(sample_text, multiple_attributes, llm_debug):
    """Test generation of multiple attribute values from the same text."""
    
    generator = TextGenerator(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    results = []
    for attribute in multiple_attributes:
        result = await generator.generate_attribute(
            text=sample_text,
            attribute=attribute
        )
        results.append(result)
    
    # Verify we got results for all attributes
    assert len(results) == len(multiple_attributes)
    
    # Verify each result
    for result in results:
        assert "value" in result
        assert "confidence" in result
        assert "explanation" in result
        assert 0 <= result["confidence"] <= 1

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_generate_attribute_empty_text(sample_attribute, llm_debug):
    """Test handling of empty text input."""
    
    generator = TextGenerator(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    result = await generator.generate_attribute(
        text="",  # Use actual empty string
        attribute=sample_attribute
    )
    
    # Verify we still get a valid response structure
    assert "value" in result
    assert "confidence" in result
    assert "explanation" in result
    
    # For empty text, we expect:
    # 1. Value should indicate no content/missing data
    assert result["value"] in ["No content", "Missing data", "Not available", "Unknown"]
    # 2. Confidence should be 0 for empty input
    assert result["confidence"] == 0.0
    # 3. Explanation should mention empty/missing content
    assert any(phrase in result["explanation"].lower() for phrase in ["empty", "no content", "missing", "not provided"]) 