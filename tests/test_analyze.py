import pytest
from contact_center_analysis import DataAnalyzer
import os
import json

@pytest.fixture
def sample_attribute_values():
    return {
        "save_attempt": {
            "value": "Yes",
            "confidence": 0.95,
            "explanation": "The agent clearly attempted to save the customer by offering discounts."
        },
        "cancellation_reason": {
            "value": "Price too high",
            "confidence": 0.88,
            "explanation": "Customer explicitly mentioned that the price was too expensive."
        },
        "retention_offer_made": {
            "value": "20% discount",
            "confidence": 0.92,
            "explanation": "Agent offered a 20% discount on the monthly subscription."
        }
    }

@pytest.fixture
def sample_questions():
    return [
        "When customers call to cancel their service, how often do agents try to save the customer?",
        "What are the most common reasons for cancellation?",
        "How effective are our retention offers?"
    ]

@pytest.fixture
def sample_analysis_results():
    return {
        "answers": [
            {
                "question": "When customers call to cancel their service, how often do agents try to save the customer?",
                "answer": "Based on the data, agents attempt to save customers in approximately 85% of cancellation calls.",
                "confidence": "High",
                "supporting_data": "In 17 out of 20 analyzed conversations, agents made explicit save attempts."
            },
            {
                "question": "What are the most common reasons for cancellation?",
                "answer": "The most common reason for cancellation is pricing concerns, followed by service quality issues.",
                "confidence": "Medium",
                "supporting_data": "45% of customers cited price as their primary reason for cancellation."
            }
        ],
        "data_gaps": ["Limited information on retention offer effectiveness"]
    }

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_analyze_findings(sample_attribute_values, sample_questions, llm_debug):
    """Test analysis of findings from attribute extraction."""
    
    analyzer = DataAnalyzer(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    result = await analyzer.analyze_findings(
        attribute_values=sample_attribute_values,
        questions=sample_questions
    )
    
    # Verify response structure
    assert "answers" in result
    assert isinstance(result["answers"], list)
    assert len(result["answers"]) > 0
    
    # Verify each answer has required fields
    for answer in result["answers"]:
        assert "question" in answer
        assert "answer" in answer
        assert "confidence" in answer
        assert "supporting_data" in answer

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_analyze_retention_strategies(sample_analysis_results, llm_debug):
    """Test generation of retention strategy recommendations."""
    
    analyzer = DataAnalyzer(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    result = await analyzer.analyze_retention_strategies(
        analysis_results=sample_analysis_results
    )
    
    # Verify response structure
    assert "immediate_actions" in result
    assert isinstance(result["immediate_actions"], list)
    assert len(result["immediate_actions"]) > 0
    
    # Verify each action has required fields
    for action in result["immediate_actions"]:
        assert "action" in action
        assert "rationale" in action
        assert "expected_impact" in action
        assert "priority" in action
        assert isinstance(action["priority"], int)
    
    # Verify other required fields
    assert "implementation_notes" in result
    assert "success_metrics" in result