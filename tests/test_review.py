import os
import pytest
from contact_center_analysis.review import Reviewer
from contact_center_analysis.analyze import DataAnalyzer
from contact_center_analysis.categorize import Categorizer
from contact_center_analysis.match import AttributeMatcher
from contact_center_analysis.questions import QuestionGenerator
from contact_center_analysis.recommend import RecommendationEngine
from contact_center_analysis.text import TextGenerator

@pytest.fixture
def reviewer():
    """Create a Reviewer instance for testing."""
    return Reviewer(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=True
    )

@pytest.fixture
def sample_analysis_results():
    """Sample analysis results for testing."""
    return {
        "answers": [
            {
                "question": "What are the most common reasons for customer cancellations?",
                "answer": "The most common reasons for customer cancellations are price concerns (45%), service quality issues (30%), and switching to competitors (15%).",
                "key_metrics": ["Price concerns: 45%", "Service quality: 30%", "Competitor switch: 15%"],
                "confidence": "High",
                "supporting_data": "Analysis of 500 cancellation calls shows consistent patterns across regions."
            }
        ],
        "data_gaps": [
            "Limited information on competitor offers that customers switched to",
            "No demographic breakdown of cancellation reasons"
        ]
    }

@pytest.fixture
def sample_categorization_results():
    """Sample categorization results for testing."""
    return {
        "classifications": [
            {
                "intent": "I want to cancel my subscription",
                "is_match": True
            },
            {
                "intent": "How do I update my payment method?",
                "is_match": False
            },
            {
                "intent": "I need to end my service",
                "is_match": True
            }
        ]
    }

@pytest.fixture
def sample_attribute_matching_results():
    """Sample attribute matching results for testing."""
    return {
        "matches": [
            {
                "required_field": "cancellation_reason",
                "matched_attribute": "reason_for_cancellation",
                "confidence": 0.92
            },
            {
                "required_field": "customer_sentiment",
                "matched_attribute": "sentiment_score",
                "confidence": 0.85
            }
        ]
    }

@pytest.fixture
def sample_question_generation_results():
    """Sample question generation results for testing."""
    return [
        {
            "question_id": "Q1",
            "question": "What are the most common reasons for customer cancellations?",
            "rationale": "Understanding cancellation drivers helps improve retention strategies."
        },
        {
            "question_id": "Q2",
            "question": "How effective are current retention offers?",
            "rationale": "Evaluating offer effectiveness helps optimize retention spending."
        }
    ]

@pytest.fixture
def sample_recommendation_results():
    """Sample recommendation results for testing."""
    return {
        "immediate_actions": [
            {
                "action": "Implement price-match guarantee for loyal customers",
                "rationale": "45% of cancellations are price-related",
                "expected_impact": "15-20% reduction in price-related cancellations",
                "priority": 5
            },
            {
                "action": "Retrain agents on empathy and active listening",
                "rationale": "30% of cancellations mention poor service experience",
                "expected_impact": "10-15% improvement in service quality perception",
                "priority": 4
            }
        ],
        "implementation_notes": [
            "Price-match should be limited to customers with 6+ months tenure",
            "Training should be completed within 30 days"
        ],
        "success_metrics": [
            "Reduction in price-related cancellation rate",
            "Improvement in service quality CSAT scores"
        ]
    }

@pytest.fixture
def sample_text_generation_results():
    """Sample text generation results for testing."""
    return {
        "attributes": [
            {
                "field_name": "cancellation_reason",
                "title": "Cancellation Reason",
                "description": "The primary reason the customer is cancelling their service",
                "rationale": "Understanding why customers cancel helps improve retention"
            },
            {
                "field_name": "retention_offer_accepted",
                "title": "Retention Offer Accepted",
                "description": "Whether the customer accepted a retention offer",
                "rationale": "Tracks effectiveness of retention strategies"
            }
        ]
    }

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_review_analysis_output(reviewer, sample_analysis_results):
    """Test reviewing analysis output from DataAnalyzer."""
    
    # Define review criteria
    criteria = [
        "Accuracy and factual correctness",
        "Completeness of the response",
        "Clarity and actionability of insights",
        "Proper identification of data gaps"
    ]
    
    # Create a sample prompt that would have generated this output
    original_prompt = """Based on the analysis of customer service conversations, help answer these questions:
    
Questions:
1. What are the most common reasons for customer cancellations?

Please provide:
1. Specific answers to each question, citing the data
2. Key metrics that quantify the answer when applicable
3. Confidence level for each answer
4. Identification of any data gaps"""
    
    # Review the analysis results
    review_results = await reviewer.review_analysis(
        analysis=sample_analysis_results,
        criteria=criteria,
        original_prompt=original_prompt
    )
    
    # Verify review structure
    assert "criteria_scores" in review_results
    assert "overall_quality" in review_results
    assert "prompt_effectiveness" in review_results
    
    # Verify criteria scores - don't check exact length since LLM may add additional criteria
    assert len(review_results["criteria_scores"]) > 0
    for score in review_results["criteria_scores"]:
        assert "criterion" in score
        assert "score" in score
        assert "assessment" in score
        assert "improvement_needed" in score
        assert isinstance(score["score"], float)
        assert 0 <= score["score"] <= 1
    
    # Verify overall quality
    assert "score" in review_results["overall_quality"]
    assert "strengths" in review_results["overall_quality"]
    assert "weaknesses" in review_results["overall_quality"]
    
    # Verify prompt effectiveness
    assert "assessment" in review_results["prompt_effectiveness"]
    assert "suggested_improvements" in review_results["prompt_effectiveness"]

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_review_categorization_output(reviewer, sample_categorization_results):
    """Test reviewing categorization output from Categorizer."""
    
    # Define review criteria
    criteria = [
        "Accuracy of classification",
        "Consistency across similar intents",
        "Handling of edge cases"
    ]
    
    # Create a sample prompt that would have generated this output
    original_prompt = """Analyze each intent text and classify if it represents a cancellation request.

Use this JSON schema:
IntentClassification = {"intent": str, "is_match": bool}
Return: list[IntentClassification]

Examples of cancellation intents:
- I want to cancel my account
- Please terminate my subscription
- I need to end my service

Classify the following intents:
- I want to cancel my subscription
- How do I update my payment method?
- I need to end my service"""
    
    # Review the categorization results
    review_results = await reviewer.review_analysis(
        analysis=sample_categorization_results,
        criteria=criteria,
        original_prompt=original_prompt
    )
    
    # Verify review structure
    assert "criteria_scores" in review_results
    assert "overall_quality" in review_results
    assert "prompt_effectiveness" in review_results

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_review_attribute_matching_output(reviewer, sample_attribute_matching_results):
    """Test reviewing attribute matching output from AttributeMatcher."""
    
    # Define review criteria
    criteria = [
        "Accuracy of field matching",
        "Appropriate confidence scores",
        "Handling of semantic similarity"
    ]
    
    # Review the attribute matching results
    review_results = await reviewer.review_analysis(
        analysis=sample_attribute_matching_results,
        criteria=criteria
    )
    
    # Verify review structure
    assert "criteria_scores" in review_results
    assert "overall_quality" in review_results

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_review_question_generation_output(reviewer, sample_question_generation_results):
    """Test reviewing question generation output from QuestionGenerator."""
    
    # Define review criteria
    criteria = [
        "Relevance to business needs",
        "Analytical depth",
        "Clarity and specificity",
        "Coverage of important topics"
    ]
    
    # Review the question generation results
    review_results = await reviewer.review_analysis(
        analysis=sample_question_generation_results,
        criteria=criteria
    )
    
    # Verify review structure
    assert "criteria_scores" in review_results
    assert "overall_quality" in review_results

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_review_recommendation_output(reviewer, sample_recommendation_results):
    """Test reviewing recommendation output from RecommendationEngine."""
    
    # Define review criteria
    criteria = [
        "Actionability of recommendations",
        "Alignment with business goals",
        "Appropriate prioritization",
        "Feasibility of implementation",
        "Expected impact"
    ]
    
    # Review the recommendation results
    review_results = await reviewer.review_analysis(
        analysis=sample_recommendation_results,
        criteria=criteria
    )
    
    # Verify review structure
    assert "criteria_scores" in review_results
    assert "overall_quality" in review_results

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_review_text_generation_output(reviewer, sample_text_generation_results):
    """Test reviewing text generation output from TextGenerator."""
    
    # Define review criteria
    criteria = [
        "Completeness of attribute definitions",
        "Clarity of descriptions",
        "Relevance to analytical needs",
        "Naming consistency"
    ]
    
    # Review the text generation results
    review_results = await reviewer.review_analysis(
        analysis=sample_text_generation_results,
        criteria=criteria
    )
    
    # Verify review structure
    assert "criteria_scores" in review_results
    assert "overall_quality" in review_results

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_suggest_improvements(reviewer, sample_recommendation_results):
    """Test suggesting improvements for recommendation output."""
    
    # Create a sample prompt that would have generated this output
    original_prompt = """Based on this analysis of customer cancellations and retention efforts, 
    recommend specific, actionable steps to improve customer retention."""
    
    # Get improvement suggestions
    improvement_results = await reviewer.suggest_improvements(
        current_results=sample_recommendation_results,
        original_prompt=original_prompt
    )
    
    # Verify improvement suggestions structure
    assert "content_improvements" in improvement_results
    assert "prompt_improvements" in improvement_results
    assert "revised_prompt" in improvement_results
    
    # Verify content improvements
    for improvement in improvement_results["content_improvements"]:
        assert "issue" in improvement
        assert "suggestion" in improvement
        assert "priority" in improvement
        assert isinstance(improvement["priority"], int)
        assert 1 <= improvement["priority"] <= 5
    
    # Verify prompt improvements
    for improvement in improvement_results["prompt_improvements"]:
        assert "issue" in improvement
        assert "suggested_prompt_modification" in improvement
        assert "rationale" in improvement
    
    # Verify revised prompt
    assert isinstance(improvement_results["revised_prompt"], str)
    assert len(improvement_results["revised_prompt"]) > 0

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_end_to_end_review_workflow(reviewer):
    """Test an end-to-end workflow with actual analyzer output and review."""
    
    # Create analyzer instance
    analyzer = DataAnalyzer(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=True
    )
    
    # Sample data for analysis
    sample_data = {
        "cancellation_reasons": {
            "price": 45,
            "service_quality": 30,
            "competitor": 15,
            "other": 10
        },
        "retention_offers": {
            "discount": {"acceptance_rate": 0.35, "count": 150},
            "free_month": {"acceptance_rate": 0.25, "count": 100},
            "upgrade": {"acceptance_rate": 0.40, "count": 75}
        }
    }
    
    # Questions to analyze
    questions = [
        "What are the most effective retention offers?",
        "What is the primary reason for customer cancellations?"
    ]
    
    # Run analysis
    analysis_results = await analyzer.analyze_findings(
        attribute_values=sample_data,
        questions=questions
    )
    
    # Define review criteria
    criteria = [
        "Accuracy and factual correctness",
        "Completeness of the response",
        "Clarity and actionability of insights"
    ]
    
    # Review the analysis results
    review_results = await reviewer.review_analysis(
        analysis=analysis_results,
        criteria=criteria
    )
    
    # Verify review structure
    assert "criteria_scores" in review_results
    assert "overall_quality" in review_results
    
    # If review indicates improvements needed, get suggestions
    if any(score.get("improvement_needed", False) for score in review_results["criteria_scores"]):
        improvement_results = await reviewer.suggest_improvements(
            current_results=analysis_results
        )
        
        # Verify improvement suggestions
        assert "content_improvements" in improvement_results
        assert "revised_prompt" in improvement_results 