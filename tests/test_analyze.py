import pytest
from contact_center_analysis import DataAnalyzer
import os
import json

@pytest.fixture
def sample_attribute_values():
    return {
        "save_attempt": {
            "distribution": {
                "Yes": 25,
                "No": 75
            },
            "confidence": 0.95,
            "explanation": "In only 25% of cancellation calls, agents made explicit save attempts."
        },
        "cancellation_reason": {
            "distribution": {
                "Price too high": 45,
                "Service quality issues": 30,
                "Found better competitor": 15,
                "No longer need service": 10
            },
            "confidence": 0.88,
            "explanation": "Price concerns represent the largest segment (45%) of cancellation reasons."
        },
        "retention_offer_made": {
            "distribution": {
                "20% discount": 20,
                "Free month": 10,
                "Service upgrade": 5,
                "No offer": 65
            },
            "confidence": 0.92,
            "explanation": "Agents made retention offers in only 35% of calls, with most calls (65%) receiving no offer."
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
                "answer": "Based on the data, agents attempt to save customers in only 25% of cancellation calls.",
                "confidence": "High",
                "supporting_data": "In 75% of analyzed conversations, agents made no explicit save attempts."
            },
            {
                "question": "What are the most common reasons for cancellation?",
                "answer": "The most common reason for cancellation is pricing concerns (45%), followed by service quality issues (30%), competitor offers (15%), and no longer needing the service (10%).",
                "confidence": "Medium",
                "supporting_data": "45% of customers cited price as their primary reason for cancellation, while 30% mentioned service quality issues."
            },
            {
                "question": "How effective are our retention offers?",
                "answer": "Retention offers are only made in 35% of calls, with 65% receiving no offer at all. The most common offer is a 20% discount (20% of calls).",
                "confidence": "Medium",
                "supporting_data": "In 65% of calls, no retention offer was made to the customer."
            }
        ],
        "data_gaps": ["Limited information on retention offer effectiveness"],
        "key_metrics": [
            {
                "name": "Save attempt rate",
                "value": "25%",
                "trend": "Declining",
                "recommendation": "Implement mandatory save attempt protocols"
            },
            {
                "name": "Price-related cancellations",
                "value": "45%",
                "trend": "Increasing",
                "recommendation": "Review pricing strategy and competitive positioning"
            },
            {
                "name": "Retention offer rate",
                "value": "35%",
                "trend": "Stable",
                "recommendation": "Increase offer frequency and train agents on effective offer presentation"
            }
        ]
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