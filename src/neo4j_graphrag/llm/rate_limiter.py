#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import asyncio
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Dict

from ..exceptions import LLMGenerationError

logger = logging.getLogger(__name__)

# Type variable for function return types
T = TypeVar("T")


class BaseRateLimiter(ABC):
    """Base class for rate limiters."""

    @abstractmethod
    def acquire(self, slots: int = 1) -> bool:
        """Attempt to acquire slots from the rate limiter.
        
        Args:
            slots: Number of slots to acquire (default: 1)
            
        Returns:
            True if slots were acquired, False otherwise
        """

    @abstractmethod
    async def aacquire(self, slots: int = 1) -> bool:
        """Async version of acquire.
        
        Args:
            slots: Number of slots to acquire (default: 1)
            
        Returns:
            True if slots were acquired, False otherwise
        """


class TokenTrackingRateLimiter(BaseRateLimiter):
    """Base class for rate limiters that can track actual token usage after API calls."""
    
    def update_token_usage(self, tokens_used: int) -> None:
        """Update the rate limiter with actual token usage from API response.
        
        Args:
            tokens_used: Number of tokens actually consumed by the API call
        """
        pass
    
    async def aupdate_token_usage(self, tokens_used: int) -> None:
        """Async version of update_token_usage.
        
        Args:
            tokens_used: Number of tokens actually consumed by the API call
        """
        pass


class SlotBucketRateLimiter(BaseRateLimiter):
    """Slot bucket rate limiter implementation.
    
    This rate limiter uses a slot bucket algorithm to control the rate of requests.
    Slots are added to the bucket at a constant rate, and each request consumes slots.
    
    Args:
        requests_per_second: Maximum number of requests per second
        max_bucket_size: Maximum number of slots the bucket can hold
        initial_slots: Initial number of slots in the bucket
    """

    def __init__(
        self,
        requests_per_second: float = 1.0,
        max_bucket_size: Optional[int] = None,
        initial_slots: Optional[int] = None,
    ):
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
            
        self.requests_per_second = requests_per_second
        self.max_bucket_size = max_bucket_size or int(requests_per_second)
        self.slots = initial_slots or self.max_bucket_size
        self.last_update = time.time()
        
        # Synchronous lock for sync methods
        self._lock = threading.Lock()
        
        # Async lock for async methods  
        self._async_lock = asyncio.Lock()

    def _add_slots(self) -> None:
        """Add slots to the bucket based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        slots_to_add = elapsed * self.requests_per_second
        
        if slots_to_add > 0:
            self.slots = min(self.max_bucket_size, self.slots + slots_to_add)
            self.last_update = now

    def acquire(self, slots: int = 1) -> bool:
        """Attempt to acquire slots from the bucket."""
        with self._lock:
            self._add_slots()
            if self.slots >= slots:
                self.slots -= slots
                return True
            return False

    async def aacquire(self, slots: int = 1) -> bool:
        """Async version of acquire using proper async lock."""
        async with self._async_lock:
            self._add_slots()
            if self.slots >= slots:
                self.slots -= slots
                return True
            return False


class TokenBucketRateLimiter(TokenTrackingRateLimiter):
    """Token bucket rate limiter implementation.
    
    This rate limiter tracks actual token consumption from API responses
    and manages rate limits based on tokens per second/minute.
    
    Args:
        tokens_per_second: Maximum number of tokens per second
        max_bucket_size: Maximum number of tokens the bucket can hold
        initial_tokens: Initial number of tokens in the bucket
        estimated_tokens_per_request: Estimated tokens per request for pre-flight checks
    """

    def __init__(
        self,
        tokens_per_second: float = 1000.0,  # Default: 1000 tokens/second
        max_bucket_size: Optional[int] = None,
        initial_tokens: Optional[int] = None,
        estimated_tokens_per_request: int = 500,  # Conservative estimate
    ):
        if tokens_per_second <= 0:
            raise ValueError("tokens_per_second must be positive")
        if estimated_tokens_per_request <= 0:
            raise ValueError("estimated_tokens_per_request must be positive")
            
        self.tokens_per_second = tokens_per_second
        self.max_bucket_size = max_bucket_size or int(tokens_per_second * 60)  # 1 minute worth
        self.tokens = initial_tokens or self.max_bucket_size
        self.last_update = time.time()
        self.estimated_tokens_per_request = estimated_tokens_per_request
        
        # Synchronous lock for sync methods
        self._lock = threading.Lock()
        
        # Async lock for async methods  
        self._async_lock = asyncio.Lock()

    def _add_tokens(self) -> None:
        """Add tokens to the bucket based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        tokens_to_add = elapsed * self.tokens_per_second
        
        if tokens_to_add > 0:
            self.tokens = min(self.max_bucket_size, self.tokens + tokens_to_add)
            self.last_update = now

    def acquire(self, slots: int = 1) -> bool:
        """Attempt to acquire tokens for estimated usage.
        
        Args:
            slots: Number of requests (uses estimated_tokens_per_request)
            
        Returns:
            True if estimated tokens were acquired, False otherwise
        """
        tokens_needed = slots * self.estimated_tokens_per_request
        
        with self._lock:
            self._add_tokens()
            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                return True
            return False

    async def aacquire(self, slots: int = 1) -> bool:
        """Async version of acquire using proper async lock."""
        tokens_needed = slots * self.estimated_tokens_per_request
        
        async with self._async_lock:
            self._add_tokens()
            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                return True
            return False

    def update_token_usage(self, tokens_used: int) -> None:
        """Update with actual token usage and adjust bucket.
        
        This should be called after API response to correct for estimation errors.
        
        Args:
            tokens_used: Actual tokens consumed by the API call
        """
        with self._lock:
            # Calculate the difference between estimated and actual usage
            estimated_used = self.estimated_tokens_per_request
            token_diff = tokens_used - estimated_used
            
            # Adjust the bucket: if we used more than estimated, deduct more
            # if we used less than estimated, give some back
            self.tokens = max(0, min(self.max_bucket_size, self.tokens - token_diff))

    async def aupdate_token_usage(self, tokens_used: int) -> None:
        """Async version of update_token_usage."""
        async with self._async_lock:
            # Calculate the difference between estimated and actual usage
            estimated_used = self.estimated_tokens_per_request
            token_diff = tokens_used - estimated_used
            
            # Adjust the bucket: if we used more than estimated, deduct more
            # if we used less than estimated, give some back
            self.tokens = max(0, min(self.max_bucket_size, self.tokens - token_diff))


class RetryConfig:
    """Configuration for retry behavior when rate limits are hit.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add jitter: ±25% of the delay
            jitter_amount = delay * 0.25
            delay += random.uniform(-jitter_amount, jitter_amount)
            
        return max(0, delay)


def is_rate_limit_error(error: Exception) -> bool:
    """Check if an error is a rate limit error.
    
    Args:
        error: Exception to check
        
    Returns:
        True if the error appears to be a rate limit error
    """
    error_str = str(error).lower()
    rate_limit_indicators = [
        "rate limit",
        "too many requests",
        "429",
        "quota exceeded",
        "rate_limit_exceeded",
        "throttled",
        "rate limiting",
    ]
    
    return any(indicator in error_str for indicator in rate_limit_indicators)


class CompositeRateLimiter(TokenTrackingRateLimiter):
    """Composite rate limiter that enforces multiple rate limits simultaneously.
    
    This is useful for providers that have both request-based AND token-based limits.
    For example, OpenAI has both "requests per minute" and "tokens per minute" limits.
    
    A request is only allowed if ALL underlying rate limiters approve it.
    
    Args:
        rate_limiters: List of rate limiters that must all approve requests
    """

    def __init__(self, *rate_limiters: BaseRateLimiter):
        if not rate_limiters:
            raise ValueError("At least one rate limiter must be provided")
        self.rate_limiters = list(rate_limiters)

    def acquire(self, slots: int = 1) -> bool:
        """Attempt to acquire slots from ALL rate limiters.
        
        Returns True only if ALL rate limiters approve the request.
        If any rate limiter rejects, we need to "rollback" any that already approved.
        """
        approved_limiters = []
        
        # Try to acquire from each limiter
        for limiter in self.rate_limiters:
            if limiter.acquire(slots):
                approved_limiters.append(limiter)
            else:
                # Rollback: return tokens/slots to limiters that already approved
                self._rollback_acquisitions(approved_limiters, slots)
                return False
        
        # All limiters approved
        return True

    async def aacquire(self, slots: int = 1) -> bool:
        """Async version of acquire from ALL rate limiters."""
        approved_limiters = []
        
        # Try to acquire from each limiter
        for limiter in self.rate_limiters:
            if await limiter.aacquire(slots):
                approved_limiters.append(limiter)
            else:
                # Rollback: return tokens/slots to limiters that already approved
                await self._arollback_acquisitions(approved_limiters, slots)
                return False
        
        # All limiters approved
        return True

    def _rollback_acquisitions(self, approved_limiters: list, slots: int) -> None:
        """Rollback acquisitions from limiters that already approved."""
        for limiter in approved_limiters:
            # Add back the tokens/slots we took
            if hasattr(limiter, '_lock') and hasattr(limiter, 'slots'):
                # SlotBucketRateLimiter
                with limiter._lock:
                    limiter.slots += slots
            elif hasattr(limiter, '_lock') and hasattr(limiter, 'tokens'):
                # TokenBucketRateLimiter
                with limiter._lock:
                    # For token limiters, we need to add back the estimated tokens
                    if hasattr(limiter, 'estimated_tokens_per_request'):
                        limiter.tokens += slots * limiter.estimated_tokens_per_request
                    else:
                        limiter.tokens += slots

    async def _arollback_acquisitions(self, approved_limiters: list, slots: int) -> None:
        """Async rollback acquisitions from limiters that already approved."""
        for limiter in approved_limiters:
            # Add back the tokens/slots we took
            if hasattr(limiter, '_async_lock') and hasattr(limiter, 'slots'):
                # SlotBucketRateLimiter
                async with limiter._async_lock:
                    limiter.slots += slots
            elif hasattr(limiter, '_async_lock') and hasattr(limiter, 'tokens'):
                # TokenBucketRateLimiter
                async with limiter._async_lock:
                    # For token limiters, we need to add back the estimated tokens
                    if hasattr(limiter, 'estimated_tokens_per_request'):
                        limiter.tokens += slots * limiter.estimated_tokens_per_request
                    else:
                        limiter.tokens += slots

    def update_token_usage(self, tokens_used: int) -> None:
        """Update token usage for all underlying token-tracking rate limiters."""
        for limiter in self.rate_limiters:
            if isinstance(limiter, TokenTrackingRateLimiter):
                limiter.update_token_usage(tokens_used)

    async def aupdate_token_usage(self, tokens_used: int) -> None:
        """Async update token usage for all underlying token-tracking rate limiters."""
        for limiter in self.rate_limiters:
            if isinstance(limiter, TokenTrackingRateLimiter):
                await limiter.aupdate_token_usage(tokens_used)


class APIFeedbackRateLimiter(TokenTrackingRateLimiter):
    """Rate limiter that syncs with actual API response headers.
    
    This rate limiter can consume real-time rate limiting information from
    API providers (like OpenAI) to maintain accurate state about remaining
    quotas and reset times.
    
    Args:
        fallback_requests_per_second: Fallback rate if no API feedback available
        fallback_tokens_per_second: Fallback token rate if no API feedback available
        estimated_tokens_per_request: Estimated tokens per request for pre-flight
    """

    def __init__(
        self,
        fallback_requests_per_second: float = 8.0,
        fallback_tokens_per_second: float = 500.0,
        estimated_tokens_per_request: int = 400,
    ):
        self.fallback_requests_per_second = fallback_requests_per_second
        self.fallback_tokens_per_second = fallback_tokens_per_second
        self.estimated_tokens_per_request = estimated_tokens_per_request
        
        # API feedback state
        self.remaining_requests: Optional[int] = None
        self.remaining_tokens: Optional[int] = None
        self.request_reset_time: Optional[float] = None
        self.token_reset_time: Optional[float] = None
        self.limit_requests: Optional[int] = None
        self.limit_tokens: Optional[int] = None
        
        # Last API feedback timestamp
        self.last_feedback_time: Optional[float] = None
        
        # Fallback to regular bucket behavior when no API feedback
        self.request_slots = 10  # Conservative initial slots
        self.token_slots = 2000   # Conservative initial tokens
        
        # Locks
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()

    def _is_feedback_stale(self, max_age_seconds: float = 30.0) -> bool:
        """Check if API feedback is too old to be reliable."""
        if self.last_feedback_time is None:
            return True
        return time.time() - self.last_feedback_time > max_age_seconds

    def _fallback_to_bucket_logic(self) -> None:
        """Fallback to bucket logic when API feedback is unavailable/stale."""
        now = time.time()
        
        # Add request slots over time
        if hasattr(self, '_last_request_update'):
            elapsed = now - self._last_request_update
            slots_to_add = elapsed * self.fallback_requests_per_second
            self.request_slots = min(20, self.request_slots + slots_to_add)
        self._last_request_update = now
        
        # Add token slots over time
        if hasattr(self, '_last_token_update'):
            elapsed = now - self._last_token_update
            tokens_to_add = elapsed * self.fallback_tokens_per_second
            self.token_slots = min(5000, self.token_slots + tokens_to_add)
        self._last_token_update = now

    def acquire(self, slots: int = 1) -> bool:
        """Acquire slots using API feedback or fallback logic."""
        with self._lock:
            now = time.time()
            
            # Check if we have fresh API feedback
            if not self._is_feedback_stale() and self.remaining_requests is not None:
                # Use API feedback
                tokens_needed = slots * self.estimated_tokens_per_request
                
                # Check both request and token limits
                if (self.remaining_requests >= slots and 
                    (self.remaining_tokens is None or self.remaining_tokens >= tokens_needed)):
                    
                    # Optimistically decrease our tracking
                    self.remaining_requests -= slots
                    if self.remaining_tokens is not None:
                        self.remaining_tokens -= tokens_needed
                    
                    return True
                else:
                    return False
            else:
                # Fallback to bucket logic
                self._fallback_to_bucket_logic()
                
                tokens_needed = slots * self.estimated_tokens_per_request
                if self.request_slots >= slots and self.token_slots >= tokens_needed:
                    self.request_slots -= slots
                    self.token_slots -= tokens_needed
                    return True
                else:
                    return False

    async def aacquire(self, slots: int = 1) -> bool:
        """Async version of acquire."""
        async with self._async_lock:
            # For simplicity, delegate to sync version since it's mostly calculations
            return self.acquire(slots)

    def update_from_api_response(self, headers: Dict[str, str]) -> None:
        """Update rate limiter state from API response headers.
        
        Args:
            headers: Response headers from the API call
        """
        with self._lock:
            self.last_feedback_time = time.time()
            
            # Extract OpenAI-style headers
            self.remaining_requests = self._parse_header_int(headers, 'x-ratelimit-remaining-requests')
            self.remaining_tokens = self._parse_header_int(headers, 'x-ratelimit-remaining-tokens')
            self.limit_requests = self._parse_header_int(headers, 'x-ratelimit-limit-requests')
            self.limit_tokens = self._parse_header_int(headers, 'x-ratelimit-limit-tokens')
            
            # Parse reset times (Unix timestamps)
            self.request_reset_time = self._parse_header_float(headers, 'x-ratelimit-reset-requests')
            self.token_reset_time = self._parse_header_float(headers, 'x-ratelimit-reset-tokens')

    async def aupdate_from_api_response(self, headers: Dict[str, str]) -> None:
        """Async version of update_from_api_response."""
        async with self._async_lock:
            self.update_from_api_response(headers)

    def _parse_header_int(self, headers: Dict[str, str], key: str) -> Optional[int]:
        """Parse integer header value."""
        value = headers.get(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass
        return None

    def _parse_header_float(self, headers: Dict[str, str], key: str) -> Optional[float]:
        """Parse float header value."""
        value = headers.get(key)
        if value is not None:
            try:
                return float(value)
            except ValueError:
                pass
        return None

    def update_token_usage(self, tokens_used: int) -> None:
        """Update with actual token usage (for compatibility)."""
        # With API feedback, this is less important since we get real updates
        # But we can still adjust our optimistic tracking
        with self._lock:
            if self.remaining_tokens is not None:
                # Correct our optimistic estimate
                estimated_used = self.estimated_tokens_per_request
                correction = tokens_used - estimated_used
                self.remaining_tokens = max(0, self.remaining_tokens - correction)

    async def aupdate_token_usage(self, tokens_used: int) -> None:
        """Async version of update_token_usage."""
        async with self._async_lock:
            self.update_token_usage(tokens_used)

    def get_status(self) -> Dict[str, any]:
        """Get current rate limiter status for debugging."""
        now = time.time()
        return {
            'remaining_requests': self.remaining_requests,
            'remaining_tokens': self.remaining_tokens,
            'request_reset_time': self.request_reset_time,
            'token_reset_time': self.token_reset_time,
            'request_reset_in_seconds': (
                self.request_reset_time - now if self.request_reset_time else None
            ),
            'token_reset_in_seconds': (
                self.token_reset_time - now if self.token_reset_time else None
            ),
            'limit_requests': self.limit_requests,
            'limit_tokens': self.limit_tokens,
            'has_fresh_feedback': not self._is_feedback_stale(),
            'last_feedback_age_seconds': (
                now - self.last_feedback_time if self.last_feedback_time else None
            ),
        } 
