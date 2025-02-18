from typing import List, Dict, Any
from .base import BaseAnalyzer

class QuestionGenerator(BaseAnalyzer):
    """Generate and answer questions about conversation data."""
    
    async def generate_analysis_questions(self, context: str) -> List[Dict[str, str]]:
        """Generate relevant questions for analyzing the data."""
        prompt = f"""Given this context about the conversation data:
{context}

Generate a list of analytical questions that would provide valuable insights.
Return as JSON list of questions with fields:
- question_id: str
- question: str
- rationale: str"""
        
        response = await self._generate_content(prompt)
        return self._parse_json_response(response)
    
    async def answer_question(self, 
                            question: str, 
                            data: Dict[str, Any],
                            context: str = "") -> Dict[str, Any]:
        """Answer a specific analytical question using the data."""
        prompt = f"""Question: {question}

Available Data:
{self._format_data(data)}

Context:
{context}

Provide a detailed answer with:
1. Direct response to the question
2. Supporting evidence from the data
3. Confidence level and any caveats
4. Suggestions for additional data that could improve the answer"""
        
        response = await self._generate_content(prompt)
        return self._parse_json_response(response) 