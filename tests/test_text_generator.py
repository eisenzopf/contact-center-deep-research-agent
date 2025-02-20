import pytest
from contact_center_analysis import TextGenerator
import os
import time
from tabulate import tabulate

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
    
    # Verify basic structure
    assert "field_name" in result
    assert "value" in result
    assert "confidence" in result
    assert "label" not in result
    
    # Verify data types
    assert isinstance(result["value"], str)
    assert isinstance(result["confidence"], float)
    assert 0 <= result["confidence"] <= 1
    
    # Test with label generation
    result_with_label = await generator.generate_labeled_attribute(
        text=sample_text,
        attribute=sample_attribute,
        create_label=True
    )
    
    # Verify label is included
    assert "field_name" in result_with_label
    assert "value" in result_with_label
    assert "confidence" in result_with_label
    assert "label" in result_with_label
    
    # Verify data types
    assert isinstance(result_with_label["value"], str)
    assert isinstance(result_with_label["confidence"], float)
    assert isinstance(result_with_label["label"], str)
    assert 0 <= result_with_label["confidence"] <= 1
    
    # Verify label characteristics
    assert len(result_with_label["label"].split()) <= 5, "Label should be 5 words or less"
    assert len(result_with_label["label"]) <= 50, "Label should be 50 characters or less"
    
    # Verify value contains detailed information
    assert len(result_with_label["value"]) > len(result_with_label["label"]), "Value should be more detailed than label"

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

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_generate_attributes_parallel_batch(sample_text, multiple_attributes, llm_debug):
    """Test parallel batch processing of attribute generation."""
    
    generator = TextGenerator(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    # Create larger set of conversations for parallel processing
    conversations = [
        {"id": str(i), "text": sample_text} for i in range(10)
    ]
    
    # Test batch processing
    results = await generator.generate_attributes_batch(
        conversations=conversations,
        required_attributes=multiple_attributes,
        batch_size=5  # Process 5 at a time
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
            assert "label" in attr_value  # Label is included by default
            assert 0 <= attr_value["confidence"] <= 1
            assert isinstance(attr_value["label"], str)
            assert len(attr_value["label"]) <= 50 

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_batch_vs_parallel_performance(sample_text, multiple_attributes, llm_debug, capsys):
    """Compare performance between batch and parallel batch processing."""
    
    generator = TextGenerator(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    # Create large set of conversations
    num_conversations = 20
    conversations = [
        {"id": str(i), "text": sample_text} for i in range(num_conversations)
    ]
    
    # Test sequential batch processing
    with capsys.disabled():
        print("\nStarting sequential batch processing...")
    start_time = time.time()
    batch_results = await generator.generate_attributes_batch(
        conversations=conversations,
        required_attributes=multiple_attributes,
        batch_size=1  # Process one at a time
    )
    batch_time = time.time() - start_time
    with capsys.disabled():
        print(f"Sequential batch processing took: {batch_time:.2f}s")
    
    # Test parallel batch processing
    with capsys.disabled():
        print("\nStarting parallel batch processing...")
    start_time = time.time()
    parallel_results = await generator.generate_attributes_batch(
        conversations=conversations,
        required_attributes=multiple_attributes,
        batch_size=20  # Process all conversations in parallel
    )
    parallel_time = time.time() - start_time
    with capsys.disabled():
        print(f"Parallel batch processing took: {parallel_time:.2f}s")
    
    # Verify both approaches give same results
    assert len(batch_results) == len(parallel_results)
    assert len(batch_results) == num_conversations
    
    # Verify structure matches in both results
    for batch_result, parallel_result in zip(batch_results, parallel_results):
        assert batch_result["conversation_id"] == parallel_result["conversation_id"]
        assert len(batch_result["attribute_values"]) == len(parallel_result["attribute_values"])
        
        # Compare attribute values
        for batch_attr, parallel_attr in zip(
            batch_result["attribute_values"], 
            parallel_result["attribute_values"]
        ):
            assert batch_attr["field_name"] == parallel_attr["field_name"]
            assert isinstance(batch_attr["value"], str)
            assert isinstance(parallel_attr["value"], str)
            assert isinstance(batch_attr["confidence"], float)
            assert isinstance(parallel_attr["confidence"], float)
            assert "label" in batch_attr
            assert "label" in parallel_attr
    
    # Generate performance report
    speedup = batch_time/parallel_time
    items_per_second_batch = num_conversations / batch_time
    items_per_second_parallel = num_conversations / parallel_time
    
    report_data = [
        ["Metric", "Sequential Batch", "Parallel Batch"],
        ["Total Time (s)", f"{batch_time:.2f}", f"{parallel_time:.2f}"],
        ["Items Processed", num_conversations, num_conversations],
        ["Items/Second", f"{items_per_second_batch:.2f}", f"{items_per_second_parallel:.2f}"],
        ["Batch Size", "1", "20"],
        ["Speedup", "-", f"{speedup:.2f}x"]
    ]
    
    with capsys.disabled():
        print("\n=== Performance Report ===")
        print(tabulate(report_data, headers="firstrow", tablefmt="grid"))

    with capsys.disabled():
        print("\n=== Parallelization Stats ===")
        print(f"Maximum Concurrent Requests: {generator.llm.max_concurrent_seen}") 