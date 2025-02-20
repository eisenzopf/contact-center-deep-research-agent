from typing import Optional, Dict, Any, List, Callable, Awaitable, TypeVar
from .llm import LLMInterface
import asyncio
import time
import math

T = TypeVar('T')  # Generic type for input items
R = TypeVar('R')  # Generic type for results

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
        self.debug = debug
        self.llm = LLMInterface(
            api_key=api_key,
            model_name=model_name,
            debug=debug
        )

    async def _generate_content(self, 
                              prompt: str, 
                              expected_format: Optional[Dict[str, Any]] = None,
                              **kwargs) -> Dict[str, Any]:
        """Generate content from LLM with validation."""
        # Only print debug info if debug is enabled
        if self.debug:
            print("\n=== LLM Input ===")
            print(prompt)
            print("\n=== Expected Format ===")
            print(expected_format)
        
        response = await self.llm.generate_response(
            prompt,
            expected_format=expected_format,
            **kwargs
        )
        
        # Only print debug info if debug is enabled
        if self.debug:
            print("\n=== LLM Response ===")
            print(response)
            print("================\n")
        
        return response 

    async def process_in_batches(
        self,
        items: List[T],
        batch_size: Optional[int],
        process_func: Callable[[T], Awaitable[R]]
    ) -> List[R]:
        """Process items in batches with true parallelization."""
        if not items:
            return []
            
        effective_batch_size = batch_size or 10
        
        # Create all tasks at once
        tasks = [process_func(item) for item in items]
        
        # Process in parallel batches
        results = []
        for i in range(0, len(tasks), effective_batch_size):
            batch = tasks[i:i + effective_batch_size]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
        
        return results

class RateLimitedAnalyzer(BaseAnalyzer):
    """Base class for analyzers that need rate limiting."""
    
    async def process_in_batches(self, items, batch_size: Optional[int] = None, process_func=None):
        """Process items in batches with rate limiting."""
        if not items:
            return []
            
        effective_batch_size = batch_size or 10
        results = []
        
        for i in range(0, len(items), effective_batch_size):
            batch = items[i:i + effective_batch_size]
            batch_results = await asyncio.gather(
                *[process_func(item) for item in batch]
            )
            results.extend(batch_results)
            
        return results 