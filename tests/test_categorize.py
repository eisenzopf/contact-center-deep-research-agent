import pytest
from contact_center_analysis import Categorizer
import os

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_categorize_intents_cancellation(sample_intents, cancellation_examples, llm_debug):
    """Test categorization of cancellation intents."""
    
    categorizer = Categorizer(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    results = await categorizer.categorize_intents(
        intents=sample_intents,
        target_category="cancellation",
        examples=cancellation_examples
    )
    
    # Check that we got results for all intents
    assert len(results) == len(sample_intents)
    
    # Check that known cancellation intents are classified correctly
    cancellation_texts = [
        "cancel my subscription please",
        "need to terminate service",
        "want to end membership"
    ]
    
    for intent_text in cancellation_texts:
        assert results.get(intent_text) == True, f"Failed to identify cancellation intent: {intent_text}"

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_categorize_intents_billing(sample_intents, billing_examples, llm_debug):
    """Test categorization of billing intents."""
    
    categorizer = Categorizer(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    results = await categorizer.categorize_intents(
        intents=sample_intents,
        target_category="billing",
        examples=billing_examples
    )
    
    # Check that we got results for all intents
    assert len(results) == len(sample_intents)
    
    # Check that known billing intents are classified correctly
    billing_texts = [
        "billing question about invoice",
        "change payment method"
    ]
    
    for intent_text in billing_texts:
        assert results.get(intent_text) == True, f"Failed to identify billing intent: {intent_text}"

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_categorize_intents_error_handling(sample_intents, cancellation_examples, llm_debug):
    """Test error handling in intent categorization."""
    
    # Create categorizer with invalid API key
    categorizer = Categorizer(
        api_key="invalid_key",
        debug=llm_debug
    )
    
    # Should handle error gracefully and return False for all intents
    results = await categorizer.categorize_intents(
        intents=sample_intents,
        target_category="cancellation",
        examples=cancellation_examples
    )
    
    assert len(results) == len(sample_intents)
    assert all(not value for value in results.values())

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_categorize_intents_empty_inputs(llm_debug):
    """Test handling of empty inputs."""
    
    categorizer = Categorizer(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    # Test with empty intents list
    results = await categorizer.categorize_intents(
        intents=[],
        target_category="cancellation",
        examples=["cancel subscription"]
    )
    assert len(results) == 0
    
    # Test with empty examples list
    results = await categorizer.categorize_intents(
        intents=[{"name": "test intent"}],
        target_category="cancellation",
        examples=[]
    )
    assert len(results) == 1

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_debug_flag(llm_debug):
    """Test that debug flag is properly set."""
    print(f"\nDebug flag value: {llm_debug}")
    assert llm_debug is True, "Debug flag should be True when --llm-debug is used" 