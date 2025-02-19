from typing import Dict, Any, Optional, List, Union
import google.generativeai as genai
import json
import os
import asyncio
from datetime import datetime
import logging

class LLMInterface:
    """Interface for handling LLM interactions."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = 'gemini-2.0-flash',
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 debug: bool = False):
        """
        Initialize LLM interface.
        
        Args:
            api_key: API key for the LLM service
            model_name: Name of the model to use
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            debug: Whether to print debug information
        """
        self.debug = debug
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided or set in environment")
            
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        # Track usage for rate limiting
        self.request_history = []
        self.requests_per_minute = 1900  # Gemini rate limit
        
    async def generate_response(self,
                              prompt: str,
                              expected_format: Optional[Dict[str, Any]] = None,
                              temperature: float = 0.0,
                              **kwargs) -> Dict[str, Any]:
        """
        Generate a response from the LLM with retry logic and validation.
        """
        await self._check_rate_limit()
        
        for attempt in range(self.max_retries):
            try:
                response = await self._make_request(
                    prompt,
                    expected_format=expected_format,
                    temperature=temperature,
                    **kwargs
                )
                
                parsed_response = self._parse_response(response.text)
                
                if expected_format:
                    if self.debug:
                        self.logger.debug("\n=== Validating Response ===")
                        self.logger.debug(f"Parsed Response: {parsed_response}")
                    self._validate_response(parsed_response, expected_format)
                    
                return parsed_response
                
            except Exception as e:
                if self.debug:
                    self.logger.debug(f"\n=== Error on attempt {attempt + 1}/{self.max_retries} ===")
                    self.logger.debug(f"Error: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (attempt + 1))
    
    async def _make_request(self,
                          prompt: str,
                          expected_format: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Any:
        """Make the actual request to the LLM."""
        generation_config = {
            'temperature': kwargs.pop('temperature', 0.0),
            'candidate_count': kwargs.pop('candidate_count', 1),
            'top_p': kwargs.pop('top_p', 0.95),
            'top_k': kwargs.pop('top_k', 40),
            **kwargs
        }
        
        if self.debug:
            self.logger.debug("\n=== LLM Request ===")
            self.logger.debug(f"Prompt: {prompt}")
            self.logger.debug(f"Expected Format: {expected_format}")
            self.logger.debug(f"Generation Config: {generation_config}")
        
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        if self.debug:
            self.logger.debug("\n=== Raw LLM Response ===")
            self.logger.debug(response.text)
            self.logger.debug("================\n")
        
        self._log_request()
        return response
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the response text into a dictionary."""
        # Clean the response text
        if response_text.startswith('```'):
            response_text = response_text.split('\n', 1)[1]
            response_text = response_text.rsplit('\n', 1)[0]
            response_text = response_text.replace('```json\n', '').replace('```', '')
        
        # Additional cleaning for potential JSON formatting
        response_text = response_text.strip()
        if response_text.startswith('['):
            response_text = '{"classifications": ' + response_text + '}'
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse text: {response_text}")
    
    def _validate_response(self,
                         response: Dict[str, Any],
                         expected_format: Dict[str, Any]) -> None:
        """Validate response against expected format."""
        def _validate_dict(data: Dict[str, Any], format_dict: Dict[str, Any]) -> None:
            for key, expected_type in format_dict.items():
                if key not in data:
                    raise ValueError(f"Missing required field: {key}")
                    
                if isinstance(expected_type, dict):
                    if not isinstance(data[key], dict):
                        raise ValueError(f"Field {key} should be a dictionary")
                    _validate_dict(data[key], expected_type)
                    
                elif isinstance(expected_type, list):
                    if not isinstance(data[key], list):
                        raise ValueError(f"Field {key} should be a list")
                    if expected_type and data[key]:
                        _validate_dict(data[key][0], expected_type[0])
                        
                else:
                    if not isinstance(data[key], expected_type):
                        raise ValueError(
                            f"Field {key} should be of type {expected_type.__name__}"
                        )
        
        _validate_dict(response, expected_format)
    
    def _log_request(self) -> None:
        """Log request for rate limiting."""
        current_time = datetime.now()
        self.request_history.append(current_time)
        
        # Clean up old requests
        cutoff = current_time.timestamp() - 60
        self.request_history = [
            req for req in self.request_history 
            if req.timestamp() > cutoff
        ]
    
    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limits."""
        current_time = datetime.now()
        recent_requests = len([
            req for req in self.request_history 
            if (current_time - req).total_seconds() < 60
        ])
        
        if recent_requests >= self.requests_per_minute:
            delay = 60 - (
                current_time - self.request_history[-self.requests_per_minute]
            ).total_seconds()
            if delay > 0:
                await asyncio.sleep(delay) 