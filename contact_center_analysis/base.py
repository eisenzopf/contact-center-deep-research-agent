from typing import Optional, Dict, Any
from .llm import LLMInterface

class BaseAnalyzer:
    """Base class with shared LLM functionality."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = 'gemini-2.0-flash',
                 debug: bool = False):
        """
        Initialize base analyzer.
        
        Args:
            api_key: Gemini API key (optional, will check env var if not provided)
            model_name: Name of Gemini model to use
            debug: Whether to enable LLM debugging
        """
        self.llm = LLMInterface(api_key=api_key, model_name=model_name, debug=debug)

    async def _generate_content(self, 
                              prompt: str, 
                              expected_format: Optional[Dict[str, Any]] = None,
                              **kwargs) -> Dict[str, Any]:
        """Generate content from LLM with validation."""
        print("\n=== LLM Input ===")
        print(prompt)
        print("\n=== Expected Format ===")
        print(expected_format)
        
        response = await self.llm.generate_response(
            prompt,
            expected_format=expected_format,
            **kwargs
        )
        
        print("\n=== LLM Response ===")
        print(response)
        print("================\n")
        
        return response 