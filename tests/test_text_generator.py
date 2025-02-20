import pytest
from contact_center_analysis import TextGenerator
import os

@pytest.fixture
def sample_text():
    return """Agent: Thank you for calling customer support. My name is Sarah. How can I help you today?

Customer: Hi Sarah, I need to cancel my premium subscription.

Agent: I'm sorry to hear that. Could you tell me more about why you'd like to cancel?

Customer: Well, it's just too expensive. I'm paying $49.99 a month and I've seen other companies offering similar services for like $29.99 or $39.99.

Agent: I understand price is a concern. As a valued customer of 2 years, I can offer you our retention discount of 20% off for the next 6 months.

Customer: I appreciate the offer, but even with that discount... I've been thinking about it and I'm really only using a small part of what I'm paying for.

Agent: Which features do you use most often?

Customer: Mostly just the basic reporting. I rarely touch the advanced analytics or those custom reporting features everyone said were so great during the sales pitch. Maybe using 40% of what I'm paying for.

Agent: I understand. Would you be interested in our basic tier? It includes the standard reporting features you're using.

Customer: No, I think I've made up my mind. I'd like to proceed with the cancellation.

Agent: I understand. I'll help you process that cancellation right away. Your service will remain active until the end of your current billing cycle."""

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

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_generate_labeled_attribute(sample_text, sample_attribute, llm_debug):
    """Test generation of attribute value with optional label generation in a single request."""
    
    generator = TextGenerator(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    # Test without label generation first
    result = await generator.generate_labeled_attribute(
        text=sample_text,
        attribute=sample_attribute,
        create_label=False
    )
    
    # Verify basic response structure
    assert "value" in result
    assert "confidence" in result
    assert "explanation" in result
    assert "label" not in result
    
    # Test with label generation
    result_with_label = await generator.generate_labeled_attribute(
        text=sample_text,
        attribute=sample_attribute,
        create_label=True
    )
    
    # Verify extended response structure
    assert "value" in result_with_label
    assert "confidence" in result_with_label
    assert "explanation" in result_with_label
    assert "label" in result_with_label
    assert "label_confidence" in result_with_label
    
    # Verify label characteristics
    assert len(result_with_label["label"].split()) <= 5, "Label should be 5 words or less"
    assert len(result_with_label["label"]) <= 50, "Label should be 50 characters or less"
    
    # Verify confidence values
    assert 0 <= result_with_label["confidence"] <= 1
    assert 0 <= result_with_label["label_confidence"] <= 1
    
    # Verify value contains detailed information
    assert len(result_with_label["value"]) > len(result_with_label["label"]), "Value should be more detailed than label"
    
    # Check for specific content indicators
    value_lower = result_with_label["value"].lower()
    assert any(word in value_lower for word in ["price", "cost", "expensive"]), "Value should mention price-related details"
    assert any(word in value_lower for word in ["feature", "premium", "subscription"]), "Value should mention product features"

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_generate_attributes_batch(sample_text, multiple_attributes, llm_debug):
    """Test batch processing of attribute generation."""
    
    generator = TextGenerator(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    # Create multiple conversations
    conversations = [
        {"id": "1", "text": sample_text},
        {"id": "2", "text": sample_text},
        {"id": "3", "text": sample_text}
    ]
    
    results = await generator.generate_attributes_batch(
        conversations=conversations,
        required_attributes=multiple_attributes,
        batch_size=2  # Test with small batch size
    )
    
    # Verify results structure
    assert len(results) == len(conversations)
    for result in results:
        assert "conversation_id" in result
        assert "attribute_values" in result
        assert len(result["attribute_values"]) == len(multiple_attributes)
        
        # Verify each attribute value
        for attr_value in result["attribute_values"]:
            assert "field_name" in attr_value
            assert "value" in attr_value
            assert "confidence" in attr_value
            assert 0 <= attr_value["confidence"] <= 1

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_generate_attributes_batch_empty(multiple_attributes, llm_debug):
    """Test batch processing with empty conversations."""
    
    generator = TextGenerator(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    results = await generator.generate_attributes_batch(
        conversations=[],
        required_attributes=multiple_attributes
    )
    
    assert len(results) == 0 