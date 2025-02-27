import pytest
from contact_center_analysis import RecommendationEngine
import os
import json

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
async def test_generate_retention_strategies(sample_analysis_results, llm_debug):
    """Test generation of retention strategy recommendations."""
    
    recommender = RecommendationEngine(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    result = await recommender.generate_retention_strategies(
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