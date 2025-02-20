import pytest
from contact_center_analysis import Categorizer
import os
import time
import asyncio

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_is_in_class_cancellation(sample_intents, cancellation_examples, llm_debug):
    """Test classification of cancellation intents."""
    
    categorizer = Categorizer(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    results = await categorizer.is_in_class(
        intents=sample_intents,
        target_class="cancellation",
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
async def test_is_in_class_billing(sample_intents, billing_examples, llm_debug):
    """Test classification of billing intents."""
    
    categorizer = Categorizer(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    results = await categorizer.is_in_class(
        intents=sample_intents,
        target_class="billing",
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
async def test_is_in_class_without_examples(sample_intents, llm_debug):
    """Test classification without providing examples."""
    
    categorizer = Categorizer(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    results = await categorizer.is_in_class(
        intents=sample_intents,
        target_class="cancellation"
    )
    
    assert len(results) == len(sample_intents)

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_is_in_class_error_handling(sample_intents, cancellation_examples, llm_debug):
    """Test error handling in intent classification."""
    
    # Create categorizer with invalid API key
    categorizer = Categorizer(
        api_key="invalid_key",
        debug=llm_debug
    )
    
    # Should handle error gracefully and return False for all intents
    results = await categorizer.is_in_class(
        intents=sample_intents,
        target_class="cancellation",
        examples=cancellation_examples
    )
    
    assert len(results) == len(sample_intents)
    assert all(not value for value in results.values())

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_is_in_class_empty_inputs(llm_debug):
    """Test handling of empty inputs."""
    
    categorizer = Categorizer(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    # Test with empty intents list
    results = await categorizer.is_in_class(
        intents=[],
        target_class="cancellation",
        examples=["cancel subscription"]
    )
    assert len(results) == 0
    
    # Test with empty examples list
    results = await categorizer.is_in_class(
        intents=[{"name": "test intent"}],
        target_class="cancellation",
        examples=[]
    )
    assert len(results) == 1

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_debug_flag(llm_debug):
    """Test that debug flag is properly set."""
    assert llm_debug is True, "Debug flag should be True when --llm-debug is used"

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_single_cancellation_intent(cancellation_examples, llm_debug):
    """Test categorization of a single cancellation intent."""
    
    categorizer = Categorizer(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    single_intent = [{"name": "cancel my subscription please"}]
    
    results = await categorizer.is_in_class(
        intents=single_intent,
        target_class="cancellation",
        examples=cancellation_examples
    )
    
    assert len(results) == 1
    assert results.get("cancel my subscription please") == True

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_single_billing_intent(billing_examples, llm_debug):
    """Test categorization of a single billing intent."""
    
    categorizer = Categorizer(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    single_intent = [{"name": "billing question about invoice"}]
    
    results = await categorizer.is_in_class(
        intents=single_intent,
        target_class="billing",
        examples=billing_examples
    )
    
    assert len(results) == 1
    assert results.get("billing question about invoice") == True

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_is_in_class_batch(sample_intents, cancellation_examples, llm_debug):
    """Test batch processing of intent classification."""
    
    categorizer = Categorizer(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    # Create multiple sets of intents
    all_intents = sample_intents * 3  # Triple the intents for batch processing
    
    results = await categorizer.is_in_class_batch(
        intents=all_intents,
        target_class="cancellation",
        examples=cancellation_examples,
        batch_size=5
    )
    
    assert len(results) == len(all_intents)
    assert isinstance(results, list)
    assert all(isinstance(r, dict) and len(r) == 1 for r in results)
