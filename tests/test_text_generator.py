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

@pytest.fixture
def sample_conversation_csv():
    return """conversation_id,speaker_id,speaker,date_time,text
1265390061,EMPL_01147,agent,2024-10-29T13:00:00Z,"Thank you for calling Standard Charter Bank, this is Daniela Ortiz. How can I help you today?"
1265390061,1001387048,client,2024-10-29T13:00:05Z,"Yes, hello. I'm Michael Reed. I'm having some trouble with my debit card."
1265390061,EMPL_01147,agent,2024-10-29T13:00:10Z,"Okay, Michael. I can definitely help you with that. First, I need to verify your identity. Could you please provide your account number?"
1265390061,1001387048,client,2024-10-29T13:00:17Z,"Um... one moment. I think I have it written down here... Okay, it's 5db2121591ac."
1265390061,EMPL_01147,agent,2024-10-29T13:00:24Z,"Thank you. And can you please provide your date of birth?"
1265390061,1001387048,client,2024-10-29T13:00:30Z,"It's... hold on. I always get the day and month mixed up. It's February 20, 1979."
1265390061,EMPL_01147,agent,2024-10-29T13:00:38Z,"Okay, thank you, Michael. And for security purposes, can you confirm the address we have on file for you?"
1265390061,1001387048,client,2024-10-29T13:00:45Z,"That would be 1296 Cody Trace Apt. 322, Greenville, 62611."
1265390061,EMPL_01147,agent,2024-10-29T13:00:52Z,"Perfect, thank you. Okay Michael Reed, I have verified your information. Now, what seems to be the problem with your debit card?"
1265390061,1001387048,client,2024-10-29T13:00:59Z,"Well, I tried to use it at the grocery store earlier, and it was declined. I know I have money in my account, so I'm not sure what's going on. It's really frustrating."
1265390061,EMPL_01147,agent,2024-10-29T13:01:08Z,"I understand your frustration, Michael. Let me check the status of your card and your recent transactions. Just one moment, please. May I place you on hold for about two minutes while I look into that?"
1265390061,1001387048,client,2024-10-29T13:01:16Z,"Yeah, sure, go ahead."
1265390061,EMPL_01147,agent,2024-10-29T13:01:19Z,"Thank you for your patience."
1265390061,EMPL_01147,agent,2024-10-29T13:03:21Z,"Thank you for holding, Michael. I've checked your account, and it appears your debit card was temporarily blocked due to a series of unusual transactions."
1265390061,1001387048,client,2024-10-29T13:03:29Z,"Unusual transactions? What do you mean? I haven't done anything unusual."
1265390061,EMPL_01147,agent,2024-10-29T13:03:35Z,"Our system flagged a few small POS purchases in rapid succession, which triggered a security alert. It's a precautionary measure to protect you from potential fraud."
1265390061,1001387048,client,2024-10-29T13:03:44Z,"Oh, that must have been when I was at the farmer's market. I bought a few different things from different vendors. So, what do I do now?"
1265390061,EMPL_01147,agent,2024-10-29T13:03:53Z,"I understand. I can unblock your card right away. However, I would recommend that we review those transactions together just to make sure everything is correct. Would you like to do that?"
1265390061,1001387048,client,2024-10-29T13:04:02Z,"Yes, please. I want to make sure everything is okay."
1265390061,EMPL_01147,agent,2024-10-29T13:04:06Z,"Okay, great. I see a purchase for $42.89 at a place called \"Island Greens,\" is that correct?"
1265390061,1001387048,client,2024-10-29T13:04:14Z,"Yes, that's right. I bought some lettuce."
1265390061,EMPL_01147,agent,2024-10-29T13:04:17Z,"And then a purchase for $127.28 at \"Aloha Fruits?\""
1265390061,1001387048,client,2024-10-29T13:04:23Z,"Yes, that was for some mangoes and pineapples."
1265390061,EMPL_01147,agent,2024-10-29T13:04:27Z,"Okay, everything seems to be in order. I've removed the block on your card. You should be able to use it again immediately. I'm very sorry for the inconvenience, Michael."
1265390061,1001387048,client,2024-10-29T13:04:36Z,"Okay, great. Thank you. I appreciate that."
1265390061,EMPL_01147,agent,2024-10-29T13:04:39Z,"You're welcome. Is there anything else I can help you with today, Michael? Perhaps you would be interested in our cash back credit card? Given the amount of purchases you are making, you could be earning rewards."
1265390061,1001387048,client,2024-10-29T13:04:50Z,"No, thank you. I'm not really interested in a credit card right now. I am trying to keep my spending under control."
1265390061,EMPL_01147,agent,2024-10-29T13:04:57Z,"I understand completely. Well, thank you for calling Standard Charter Bank, Michael. Have a great day!"
1265390061,1001387048,client,2024-10-29T13:05:02Z,"You too. Goodbye."
1265390061,EMPL_01147,agent,2024-10-29T13:05:05Z,"Goodbye.\""""

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
async def test_generate_intent(sample_conversation_csv, llm_debug):
    """Test generation of customer intent from conversation text."""
    
    generator = TextGenerator(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    # Convert CSV to plain text for processing
    import csv
    from io import StringIO
    
    conversation_text = ""
    csv_reader = csv.DictReader(StringIO(sample_conversation_csv))
    for row in csv_reader:
        speaker_type = row['speaker']
        text = row['text']
        conversation_text += f"{speaker_type.capitalize()}: {text}\n"
    
    result = await generator.generate_intent(conversation_text)
    
    # Verify response structure
    assert "label_name" in result
    assert "label" in result
    assert "description" in result
    
    # Verify data types
    assert isinstance(result["label_name"], str)
    assert isinstance(result["label"], str)
    assert isinstance(result["description"], str)
    
    # Verify label format
    assert result["label"] == result["label_name"].lower().replace(" ", "_")
    
    # Verify label characteristics
    words_in_label = result["label_name"].split()
    assert 1 <= len(words_in_label) <= 3, "Label should be 1-3 words"
    
    # Verify the intent is related to card issues (based on the conversation)
    assert any(term in result["label"].lower() for term in ["card", "debit", "transaction", "block", "decline"])
    
    # Verify description is not empty and has reasonable length
    assert len(result["description"]) > 10
    assert len(result["description"]) < 200  # Description should be concise

@pytest.mark.llm_debug
@pytest.mark.asyncio
async def test_generate_intent_empty_text(llm_debug):
    """Test handling of empty text input for intent generation."""
    
    generator = TextGenerator(
        api_key=os.getenv('GEMINI_API_KEY'),
        debug=llm_debug
    )
    
    result = await generator.generate_intent("")
    
    # Verify we get the default response for empty input
    assert result["label_name"] == "Unclear Intent"
    assert result["label"] == "unclear_intent"
    assert "unclear" in result["description"].lower()
    assert "does not contain" in result["description"].lower() 