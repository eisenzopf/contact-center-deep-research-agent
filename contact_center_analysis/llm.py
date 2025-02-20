from typing import Dict, Any, Optional, List, Union
import google.generativeai as genai
import json
import os
import asyncio
from datetime import datetime
import logging
import time

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
        
        # Adjust parallel capacity for better throughput
        self.requests_per_minute = 1900
        self.max_parallel = min(500, self.requests_per_minute // 2)
        self.semaphore = asyncio.Semaphore(self.max_parallel)
        
        # Use bucketed rate limiting
        self.bucket_size = 1  # seconds
        self.buckets = {}
        self.bucket_locks = {}
        
        # Concurrent request tracking
        self.active_requests = 0
        self.max_concurrent_seen = 0
        self._request_counter_lock = asyncio.Lock()
        
    async def generate_response(self,
                              prompt: str,
                              expected_format: Optional[Dict[str, Any]] = None,
                              temperature: float = 0.0,
                              **kwargs) -> Dict[str, Any]:
        """
        Generate a response from the LLM with retry logic and validation.
        """
        async with self.semaphore:
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
    
    async def generate_responses_batch(
        self,
        prompts: List[str],
        expected_format: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        temperature: float = 0.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate multiple responses in parallel while respecting rate limits."""
        # Let the semaphore handle concurrency
        all_tasks = [
            self.generate_response(
                prompt,
                expected_format=expected_format,
                temperature=temperature,
                **kwargs
            )
            for prompt in prompts
        ]
        
        # Process all at once, let semaphore handle rate limiting
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Handle exceptions
        return [
            None if isinstance(r, Exception) else r 
            for r in results
        ]
    
    async def _make_request(self,
                          prompt: str,
                          expected_format: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Any:
        """Make the actual request to the LLM."""
        async with self._request_counter_lock:
            self.active_requests += 1
            self.max_concurrent_seen = max(self.max_concurrent_seen, self.active_requests)
            if self.debug:
                print(f"\n=== Request Started [Concurrent: {self.active_requests}] ===")
                print(f"Time: {time.strftime('%H:%M:%S')}")

        try:
            generation_config = {
                'temperature': kwargs.pop('temperature', 0.0),
                'candidate_count': kwargs.pop('candidate_count', 1),
                'top_p': kwargs.pop('top_p', 0.95),
                'top_k': kwargs.pop('top_k', 40),
                **kwargs
            }
            
            # Run the synchronous API call in a thread pool to make it non-blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(prompt, generation_config=generation_config)
            )
            
            return response
            
        finally:
            async with self._request_counter_lock:
                self.active_requests -= 1
                if self.debug:
                    print(f"\n=== Request Completed [Concurrent: {self.active_requests}] ===")
                    print(f"Time: {time.strftime('%H:%M:%S')}")
    
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
    
    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limits using time buckets."""
        current_bucket = int(time.time() / self.bucket_size)
        
        if current_bucket not in self.bucket_locks:
            self.bucket_locks[current_bucket] = asyncio.Lock()
            self.buckets[current_bucket] = 0
            
        # Clean old buckets
        old_buckets = [b for b in self.buckets.keys() if b < current_bucket - 60]
        for b in old_buckets:
            del self.buckets[b]
            del self.bucket_locks[b]
            
        async with self.bucket_locks[current_bucket]:
            if self.buckets[current_bucket] >= self.requests_per_minute / 60:
                await asyncio.sleep(self.bucket_size)
                # Move to next bucket after waiting
                current_bucket = int(time.time() / self.bucket_size)
                if current_bucket not in self.buckets:
                    self.buckets[current_bucket] = 0
                    self.bucket_locks[current_bucket] = asyncio.Lock()
            
            self.buckets[current_bucket] = self.buckets.get(current_bucket, 0) + 1

class RateLimiter:
    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            current_time = time.time()
            self.requests = [req_time for req_time in self.requests 
                           if current_time - req_time < 60]
            
            if len(self.requests) >= self.max_requests:
                wait_time = 60 - (current_time - self.requests[0])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    current_time = time.time()
                    self.requests = [req_time for req_time in self.requests 
                                   if current_time - req_time < 60]
            
            self.requests.append(current_time) 