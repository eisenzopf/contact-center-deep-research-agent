import asyncio
import time
from datetime import datetime

class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, max_requests_per_minute: int = 1900):
        self.max_requests = max_requests_per_minute
        self.request_times = []
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """Acquire permission to make a request."""
        async with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            cutoff = now - 60
            self.request_times = [t for t in self.request_times if t > cutoff]
            
            # Wait if we're at the limit
            while len(self.request_times) >= self.max_requests:
                wait_time = self.request_times[0] - cutoff
                await asyncio.sleep(wait_time)
                now = time.time()
                self.request_times = [t for t in self.request_times if t > now - 60]
            
            # Add current request
            self.request_times.append(now) 