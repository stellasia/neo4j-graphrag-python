"""
Example demonstrating how to use rate limiting with neo4j-graphrag LLM providers.

This example shows different ways to configure rate limiting for LLM providers
to handle API rate limits and avoid hitting provider constraints.

Rate limiting is automatically applied to all LLM methods (invoke, ainvoke, 
invoke_with_tools, ainvoke_with_tools) when a rate_limiter is provided.
"""

import asyncio
import os
from typing import List

from neo4j_graphrag.llm import (
    OpenAILLM,
    AnthropicLLM,
    SlotBucketRateLimiter,
    TokenBucketRateLimiter,
    CompositeRateLimiter,
    APIFeedbackRateLimiter,
    RetryConfig,
)
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
)
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks


def basic_rate_limiting_example():
    """Basic example of using rate limiting with OpenAI.
    
    Rate limiting is automatically applied to all LLM methods when configured.
    """
    
    # Create a rate limiter that allows 2 requests per second
    rate_limiter = SlotBucketRateLimiter(
        requests_per_second=2.0,
        max_bucket_size=5,  # Allow bursts up to 5 requests
    )
    
    # Create retry configuration
    retry_config = RetryConfig(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        exponential_base=2.0,
        jitter=True,
    )
    
    # Create LLM with rate limiting - it's automatically applied to all methods
    llm = OpenAILLM(
        model_name="gpt-3.5-turbo",
        model_params={"temperature": 0.0},
        rate_limiter=rate_limiter,  # Automatic rate limiting!
        retry_config=retry_config,
    )
    
    # Use the LLM normally - rate limiting happens automatically
    response = llm.invoke("Hello, how are you?")
    print(f"Response: {response.content}")


def provider_specific_rate_limiting():
    """Example using different rate limits for different providers.
    
    All providers automatically get rate limiting when configured.
    """
    
    # OpenAI with conservative rate limiting (3 requests/second)
    openai_rate_limiter = SlotBucketRateLimiter(
        requests_per_second=3.0,
        max_bucket_size=10,
    )
    openai_llm = OpenAILLM(
        model_name="gpt-4",
        model_params={"temperature": 0.0},
        rate_limiter=openai_rate_limiter,  # Automatic for all methods
    )
    
    # Anthropic with conservative rate limiting (2 requests/second)
    anthropic_rate_limiter = SlotBucketRateLimiter(
        requests_per_second=2.0,
        max_bucket_size=5,
    )
    anthropic_llm = AnthropicLLM(
        model_name="claude-3-haiku-20240307",
        model_params={"temperature": 0.0},
        rate_limiter=anthropic_rate_limiter,  # Automatic for all methods
    )
    
    # Very conservative rate limiter for heavy workloads
    custom_openai_rate_limiter = SlotBucketRateLimiter(
        requests_per_second=1.0,
        max_bucket_size=3,
    )
    
    custom_openai_llm = OpenAILLM(
        model_name="gpt-4",
        rate_limiter=custom_openai_rate_limiter,  # Automatic for all methods
    )
    
    return openai_llm, anthropic_llm, custom_openai_llm


def shared_rate_limiter_example():
    """Example of sharing rate limiter between multiple LLM instances.
    
    Rate limiting is applied automatically to all methods for all instances.
    """
    
    # Create ONE rate limiter for OpenAI (since they share the same API key/limits)
    shared_openai_limiter = SlotBucketRateLimiter(
        requests_per_second=2.0,
        max_bucket_size=5,
    )
    
    # Multiple LLMs sharing the same rate limiter
    # Rate limiting is automatic for all methods on both instances
    summarizer = OpenAILLM(
        model_name="gpt-3.5-turbo",
        model_params={"temperature": 0.0},
        rate_limiter=shared_openai_limiter,  # Shared, automatic rate limiting!
    )
    
    reasoner = OpenAILLM(
        model_name="gpt-4",
        model_params={"temperature": 0.1},
        rate_limiter=shared_openai_limiter,  # Same instance, automatic!
    )
    
    return summarizer, reasoner


async def rate_limiting_with_entity_extraction():
    """Example of using rate limiting with EntityRelationExtractor.
    
    Rate limiting is automatically applied to all LLM methods.
    """
    
    # Create rate limiter for OpenAI
    rate_limiter = SlotBucketRateLimiter(
        requests_per_second=2.0,  # Conservative rate for entity extraction
        max_bucket_size=5,
    )
    
    # Create LLM with rate limiting - automatically applied to all methods
    llm = OpenAILLM(
        model_name="gpt-4",
        model_params={
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        },
        rate_limiter=rate_limiter,  # Automatic rate limiting!
    )
    
    # Create entity extractor
    extractor = LLMEntityRelationExtractor(
        llm=llm,
        max_concurrency=3,  # Process 3 chunks concurrently
        # The rate limiter will automatically ensure we don't exceed our limits
        # even with concurrent processing - no manual intervention needed!
    )
    
    # Prepare some sample text chunks
    sample_texts = [
        "Alice works at TechCorp as a software engineer.",
        "Bob is the CEO of StartupInc based in Silicon Valley.",
        "Charlie and Diana are researchers at MIT studying AI.",
        "Eve leads the marketing team at GlobalBrand.",
        "Frank is a doctor at City Hospital.",
    ]
    
    chunks = TextChunks(
        chunks=[
            TextChunk(text=text, index=i)
            for i, text in enumerate(sample_texts)
        ]
    )
    
    # Run extraction - rate limiting is applied automatically
    print("Running entity extraction with automatic rate limiting...")
    graph = await extractor.run(chunks=chunks)
    
    print(f"Extracted {len(graph.nodes)} nodes and {len(graph.relationships)} relationships")
    for node in graph.nodes[:3]:  # Show first 3 nodes
        print(f"Node: {node.label} - {node.properties}")


def custom_rate_limiting_configuration():
    """Example of custom rate limiting configurations for different scenarios.
    
    All configurations automatically apply to all LLM methods.
    """
    
    # High-frequency, light requests (e.g., simple questions)
    high_frequency_limiter = SlotBucketRateLimiter(
        requests_per_second=5.0,
        max_bucket_size=10,
    )
    
    # Low-frequency, heavy requests (e.g., document processing)
    low_frequency_limiter = SlotBucketRateLimiter(
        requests_per_second=0.5,  # 1 request every 2 seconds
        max_bucket_size=2,
    )
    
    # Burst-friendly configuration
    burst_friendly_limiter = SlotBucketRateLimiter(
        requests_per_second=2.0,
        max_bucket_size=20,  # Allow large bursts
    )
    
    # Aggressive retry configuration for critical operations
    aggressive_retry_config = RetryConfig(
        max_retries=5,
        base_delay=2.0,
        max_delay=120.0,  # Up to 2 minutes
        exponential_base=2.0,
        jitter=True,
    )
    
    # Conservative retry configuration for non-critical operations
    conservative_retry_config = RetryConfig(
        max_retries=2,
        base_delay=0.5,
        max_delay=10.0,
        exponential_base=1.5,
        jitter=False,
    )
    
    # Create LLMs for different use cases
    # Rate limiting is automatically applied to all methods
    light_requests_llm = OpenAILLM(
        model_name="gpt-3.5-turbo",
        rate_limiter=high_frequency_limiter,  # Automatic!
        retry_config=conservative_retry_config,
    )
    
    heavy_requests_llm = OpenAILLM(
        model_name="gpt-4",
        rate_limiter=low_frequency_limiter,  # Automatic!
        retry_config=aggressive_retry_config,
    )
    
    burst_llm = OpenAILLM(
        model_name="gpt-3.5-turbo",
        rate_limiter=burst_friendly_limiter,  # Automatic!
    )
    
    return light_requests_llm, heavy_requests_llm, burst_llm


def token_based_rate_limiting_example():
    """Example using token-based rate limiting for more accurate control.
    
    Token-based rate limiting tracks actual token consumption from API responses,
    providing much more accurate rate limiting than simple request counting.
    """
    
    # Create a token-based rate limiter
    # OpenAI GPT-4: typically 40,000 TPM (tokens per minute) = ~667 tokens/second
    token_rate_limiter = TokenBucketRateLimiter(
        tokens_per_second=600.0,  # Conservative: 600 tokens/second
        max_bucket_size=36000,     # 1 minute worth of tokens
        estimated_tokens_per_request=500,  # Conservative estimate for pre-flight
    )
    
    # Create LLM with token-based rate limiting
    llm = OpenAILLM(
        model_name="gpt-4",
        model_params={"temperature": 0.0},
        rate_limiter=token_rate_limiter,  # Automatic token tracking!
    )
    
    print("=== Token-Based Rate Limiting ===")
    print("Using actual token consumption from API responses!")
    
    # Test with different sized requests
    test_prompts = [
        "Hi",  # Small: ~50 tokens
        "Write a short story about a robot learning to paint.",  # Medium: ~200-400 tokens
        "Explain quantum computing, machine learning, and blockchain technology in detail with examples.",  # Large: ~800-1200 tokens
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nRequest {i}: {prompt[:50]}...")
        response = llm.invoke(prompt)
        print(f"Response length: {len(response.content)} characters")
        # The rate limiter automatically tracks actual token usage from the API response!


def comparing_slot_vs_token_rate_limiting():
    """Example comparing slot-based vs token-based rate limiting."""
    
    print("=== Comparing Slot vs Token Rate Limiting ===")
    
    # Slot-based: treats all requests equally
    slot_rate_limiter = SlotBucketRateLimiter(
        requests_per_second=2.0,  # 2 requests per second
        max_bucket_size=5,
    )
    
    slot_llm = OpenAILLM(
        model_name="gpt-3.5-turbo",
        rate_limiter=slot_rate_limiter,
    )
    
    # Token-based: tracks actual token consumption
    token_rate_limiter = TokenBucketRateLimiter(
        tokens_per_second=1000.0,  # 1000 tokens per second
        max_bucket_size=5000,      # 5 second buffer
        estimated_tokens_per_request=300,
    )
    
    token_llm = OpenAILLM(
        model_name="gpt-3.5-turbo", 
        rate_limiter=token_rate_limiter,
    )
    
    print("\nSlot-based limiter: Treats 'Hi' and long essays the same")
    print("Token-based limiter: Accurately tracks actual token usage")
    print("\nThis means:")
    print("- You can make more small requests with token-based limiting")
    print("- Large requests are properly throttled based on actual consumption")
    print("- Better utilization of your API quotas!")


def advanced_token_rate_limiting():
    """Example of advanced token-based rate limiting configurations."""
    
    # High-throughput configuration for GPT-3.5-turbo
    high_throughput_limiter = TokenBucketRateLimiter(
        tokens_per_second=3000.0,  # 3000 tokens/second (180k/minute)
        max_bucket_size=15000,     # 5 seconds worth
        estimated_tokens_per_request=200,  # Optimistic estimate
    )
    
    # Conservative configuration for GPT-4 
    conservative_limiter = TokenBucketRateLimiter(
        tokens_per_second=500.0,   # 500 tokens/second (30k/minute)
        max_bucket_size=2500,      # 5 seconds worth
        estimated_tokens_per_request=800,  # Conservative estimate
    )
    
    # Burst-friendly configuration
    burst_friendly_limiter = TokenBucketRateLimiter(
        tokens_per_second=1000.0,  # 1000 tokens/second
        max_bucket_size=60000,     # 1 minute of tokens for bursts
        estimated_tokens_per_request=400,
    )
    
    high_throughput_llm = OpenAILLM(
        model_name="gpt-3.5-turbo",
        rate_limiter=high_throughput_limiter,
    )
    
    conservative_llm = OpenAILLM(
        model_name="gpt-4",
        rate_limiter=conservative_limiter,
    )
    
    burst_llm = OpenAILLM(
        model_name="gpt-3.5-turbo",
        rate_limiter=burst_friendly_limiter,
    )
    
    print("=== Advanced Token Rate Limiting ===")
    print("✓ High-throughput: 3000 tokens/second for GPT-3.5-turbo")
    print("✓ Conservative: 500 tokens/second for GPT-4")
    print("✓ Burst-friendly: Large token buffer for burst requests")
    print("✓ All configurations automatically track actual token usage!")
    
    return high_throughput_llm, conservative_llm, burst_llm


def realistic_openai_rate_limiting():
    """Example using composite rate limiting that matches real OpenAI API limits.
    
    OpenAI has BOTH request-based AND token-based limits:
    - Requests per minute (RPM): e.g., 500 requests/minute 
    - Tokens per minute (TPM): e.g., 30,000 tokens/minute
    
    Depending on usage, you could hit either limit first!
    """
    
    print("=== Realistic OpenAI Rate Limiting ===")
    print("Enforcing BOTH request limits AND token limits simultaneously!")
    
    # Request-based limiter: 500 requests/minute = ~8.33 requests/second
    request_limiter = SlotBucketRateLimiter(
        requests_per_second=8.0,  # Slightly conservative
        max_bucket_size=20,       # Allow small bursts
    )
    
    # Token-based limiter: 30,000 tokens/minute = 500 tokens/second
    token_limiter = TokenBucketRateLimiter(
        tokens_per_second=500.0,
        max_bucket_size=2500,     # 5 seconds worth
        estimated_tokens_per_request=400,
    )
    
    # Composite limiter: request must pass BOTH limits
    composite_limiter = CompositeRateLimiter(
        request_limiter,  # Must have request slots available
        token_limiter,    # Must have token budget available
    )
    
    # Create LLM with realistic rate limiting
    llm = OpenAILLM(
        model_name="gpt-4",
        model_params={"temperature": 0.0},
        rate_limiter=composite_limiter,  # Enforces BOTH limits!
    )
    
    print("\nThis setup will:")
    print("✓ Limit to ~8 requests/second (RPM limit)")
    print("✓ Limit to ~500 tokens/second (TPM limit)")
    print("✓ Block requests if EITHER limit is exceeded")
    print("✓ Allow more small requests when token budget permits")
    print("✓ Throttle large requests even if request budget permits")
    
    # Test scenarios
    print("\n=== Test Scenarios ===")
    
    # Many small requests → will hit RPM limit first
    print("Scenario 1: Many small requests ('Hi' repeated)")
    print("→ Expected: Will hit REQUEST limit before token limit")
    
    # Few large requests → will hit TPM limit first  
    print("\nScenario 2: Few large requests (complex analysis)")
    print("→ Expected: Will hit TOKEN limit before request limit")
    
    return llm


def provider_specific_composite_limiting():
    """Examples of composite rate limiting for different providers."""
    
    print("=== Provider-Specific Composite Rate Limiting ===")
    
    # OpenAI GPT-4: 500 RPM, 30k TPM
    openai_request_limiter = SlotBucketRateLimiter(requests_per_second=8.0)
    openai_token_limiter = TokenBucketRateLimiter(
        tokens_per_second=500.0, 
        estimated_tokens_per_request=600
    )
    openai_composite = CompositeRateLimiter(openai_request_limiter, openai_token_limiter)
    
    openai_llm = OpenAILLM(
        model_name="gpt-4",
        rate_limiter=openai_composite,
    )
    
    # OpenAI GPT-3.5-turbo: 3500 RPM, 90k TPM (higher limits)
    openai_turbo_request_limiter = SlotBucketRateLimiter(requests_per_second=50.0)
    openai_turbo_token_limiter = TokenBucketRateLimiter(
        tokens_per_second=1500.0,
        estimated_tokens_per_request=300
    )
    openai_turbo_composite = CompositeRateLimiter(
        openai_turbo_request_limiter, 
        openai_turbo_token_limiter
    )
    
    openai_turbo_llm = OpenAILLM(
        model_name="gpt-3.5-turbo", 
        rate_limiter=openai_turbo_composite,
    )
    
    # Conservative setup for production workloads
    conservative_request_limiter = SlotBucketRateLimiter(requests_per_second=2.0)
    conservative_token_limiter = TokenBucketRateLimiter(
        tokens_per_second=200.0,
        estimated_tokens_per_request=800
    )
    conservative_composite = CompositeRateLimiter(
        conservative_request_limiter,
        conservative_token_limiter
    )
    
    conservative_llm = OpenAILLM(
        model_name="gpt-4",
        rate_limiter=conservative_composite,
    )
    
    print("✅ GPT-4: 8 req/s + 500 tokens/s")
    print("✅ GPT-3.5-turbo: 50 req/s + 1500 tokens/s") 
    print("✅ Conservative: 2 req/s + 200 tokens/s")
    print("✅ All enforce BOTH request AND token limits!")
    
    return openai_llm, openai_turbo_llm, conservative_llm


def api_feedback_rate_limiting_example():
    """Example using API feedback rate limiting that syncs with OpenAI's real-time headers.
    
    This is the most accurate rate limiting approach because it uses actual API feedback
    instead of guessing or estimating limits.
    """
    
    print("=== API Feedback Rate Limiting ===")
    print("Syncing with real-time OpenAI API headers!")
    
    # Create API feedback rate limiter
    # It will sync with actual OpenAI rate limit headers for perfect accuracy
    api_feedback_limiter = APIFeedbackRateLimiter(
        fallback_requests_per_second=8.0,  # Used when no API feedback available
        fallback_tokens_per_second=500.0,  # Used when no API feedback available
        estimated_tokens_per_request=400,   # For pre-flight estimation
    )
    
    # Create LLM with API feedback rate limiting
    llm = OpenAILLM(
        model_name="gpt-4",
        model_params={"temperature": 0.0},
        rate_limiter=api_feedback_limiter,  # Real-time API sync!
    )
    
    print("\nBenefits of API Feedback Rate Limiting:")
    print("✓ Syncs with actual API limits in real-time")
    print("✓ Knows exactly how many requests/tokens are remaining")
    print("✓ Knows exactly when limits reset")
    print("✓ No guesswork or estimation errors")
    print("✓ Maximum API quota utilization")
    
    # Make a test request
    print("\nMaking test request...")
    response = llm.invoke("What is the capital of France?")
    print(f"Response: {response.content[:100]}...")
    
    # Check rate limiter status (debugging info)
    status = api_feedback_limiter.get_status()
    print(f"\nRate Limiter Status:")
    print(f"Remaining requests: {status['remaining_requests']}")
    print(f"Remaining tokens: {status['remaining_tokens']}")
    print(f"Request reset in: {status['request_reset_in_seconds']:.1f}s" if status['request_reset_in_seconds'] else "Unknown")
    print(f"Token reset in: {status['token_reset_in_seconds']:.1f}s" if status['token_reset_in_seconds'] else "Unknown")
    print(f"Request limit: {status['limit_requests']}")
    print(f"Token limit: {status['limit_tokens']}")
    print(f"Has fresh feedback: {status['has_fresh_feedback']}")
    
    return llm


def comparing_rate_limiting_approaches():
    """Compare different rate limiting approaches to show the benefits of API feedback."""
    
    print("=== Comparing Rate Limiting Approaches ===")
    
    # 1. Basic slot-based (treats all requests equally)
    slot_limiter = SlotBucketRateLimiter(requests_per_second=2.0)
    slot_llm = OpenAILLM(model_name="gpt-3.5-turbo", rate_limiter=slot_limiter)
    
    # 2. Token-based (tracks actual usage)  
    token_limiter = TokenBucketRateLimiter(
        tokens_per_second=1000.0,
        estimated_tokens_per_request=400
    )
    token_llm = OpenAILLM(model_name="gpt-3.5-turbo", rate_limiter=token_limiter)
    
    # 3. API feedback (syncs with real API limits)
    api_feedback_limiter = APIFeedbackRateLimiter(
        fallback_requests_per_second=8.0,
        fallback_tokens_per_second=1000.0,
        estimated_tokens_per_request=400
    )
    api_feedback_llm = OpenAILLM(model_name="gpt-3.5-turbo", rate_limiter=api_feedback_limiter)
    
    print("\n🔹 Slot-based:")
    print("  → All requests treated equally")
    print("  → Simple but inefficient")
    print("  → May waste or exceed quotas")
    
    print("\n🔸 Token-based:")
    print("  → Tracks actual token usage") 
    print("  → Better quota utilization")
    print("  → Still based on estimates")
    
    print("\n🔶 API Feedback:")
    print("  → Syncs with real API limits")
    print("  → Perfect accuracy")
    print("  → Maximum quota utilization")
    print("  → Future-proof (adapts to API changes)")
    
    return slot_llm, token_llm, api_feedback_llm


def advanced_api_feedback_scenarios():
    """Advanced scenarios showing API feedback rate limiting capabilities."""
    
    print("=== Advanced API Feedback Scenarios ===")
    
    # Scenario 1: High-throughput with API feedback
    high_throughput_limiter = APIFeedbackRateLimiter(
        fallback_requests_per_second=50.0,  # High fallback rate
        fallback_tokens_per_second=3000.0,  # High fallback token rate
        estimated_tokens_per_request=200,   # Optimistic estimate
    )
    
    high_throughput_llm = OpenAILLM(
        model_name="gpt-3.5-turbo",
        rate_limiter=high_throughput_limiter,
    )
    
    # Scenario 2: Conservative with API feedback
    conservative_limiter = APIFeedbackRateLimiter(
        fallback_requests_per_second=2.0,   # Conservative fallback
        fallback_tokens_per_second=300.0,   # Conservative fallback
        estimated_tokens_per_request=800,   # Conservative estimate
    )
    
    conservative_llm = OpenAILLM(
        model_name="gpt-4",
        rate_limiter=conservative_limiter,
    )
    
    # Scenario 3: Multiple models sharing API feedback
    # Since they share the same API key, they should share rate limits
    shared_api_feedback_limiter = APIFeedbackRateLimiter(
        fallback_requests_per_second=8.0,
        fallback_tokens_per_second=500.0,
        estimated_tokens_per_request=500,
    )
    
    summarizer = OpenAILLM(
        model_name="gpt-3.5-turbo",
        rate_limiter=shared_api_feedback_limiter,
    )
    
    analyzer = OpenAILLM(
        model_name="gpt-4",
        rate_limiter=shared_api_feedback_limiter,  # Same instance!
    )
    
    print("✅ High-throughput: Optimistic fallbacks, syncs with real limits")
    print("✅ Conservative: Safe fallbacks, syncs with real limits")
    print("✅ Shared limits: Multiple models coordinate via API feedback")
    
    return high_throughput_llm, conservative_llm, summarizer, analyzer


def api_feedback_debugging_example():
    """Example showing how to debug and monitor API feedback rate limiting."""
    
    print("=== API Feedback Debugging ===")
    
    # Create rate limiter with debugging capabilities
    debug_limiter = APIFeedbackRateLimiter(
        fallback_requests_per_second=5.0,
        fallback_tokens_per_second=1000.0,
        estimated_tokens_per_request=300,
    )
    
    llm = OpenAILLM(
        model_name="gpt-3.5-turbo",
        rate_limiter=debug_limiter,
    )
    
    print("Debugging capabilities:")
    print("✓ View remaining requests/tokens in real-time")
    print("✓ See when limits reset")
    print("✓ Monitor API feedback freshness")
    print("✓ Track estimation accuracy")
    
    # Simulate some requests and show debugging info
    test_prompts = [
        "Hi there!",
        "Explain quantum computing briefly.",
        "Write a short poem about spring.",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Request {i}: {prompt[:30]}... ---")
        
        # Show status before request
        status_before = debug_limiter.get_status()
        print(f"Before: {status_before['remaining_requests'] or 'Unknown'} requests, "
              f"{status_before['remaining_tokens'] or 'Unknown'} tokens remaining")
        
        # Make request (rate limiter gets updated automatically)
        try:
            response = llm.invoke(prompt)
            print(f"✓ Success: {len(response.content)} chars")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        # Show status after request
        status_after = debug_limiter.get_status()
        print(f"After:  {status_after['remaining_requests'] or 'Unknown'} requests, "
              f"{status_after['remaining_tokens'] or 'Unknown'} tokens remaining")
        
        if status_after['has_fresh_feedback']:
            print("📡 Fresh API feedback available")
        else:
            print("⏳ Using fallback rate limiting")
    
    # Final status
    final_status = debug_limiter.get_status()
    print(f"\n=== Final Status ===")
    for key, value in final_status.items():
        if key.endswith('_seconds') and value is not None:
            print(f"{key}: {value:.1f}s")
        else:
            print(f"{key}: {value}")
    
    return llm, debug_limiter


async def api_feedback_with_entity_extraction():
    """Example of API feedback rate limiting with entity extraction."""
    
    print("=== API Feedback + Entity Extraction ===")
    
    # Create API feedback rate limiter
    api_feedback_limiter = APIFeedbackRateLimiter(
        fallback_requests_per_second=3.0,  # Conservative for entity extraction
        fallback_tokens_per_second=600.0,
        estimated_tokens_per_request=600,  # Entity extraction uses more tokens
    )
    
    # Create LLM with API feedback rate limiting
    llm = OpenAILLM(
        model_name="gpt-4",
        model_params={
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        },
        rate_limiter=api_feedback_limiter,
    )
    
    # Create entity extractor
    extractor = LLMEntityRelationExtractor(
        llm=llm,
        max_concurrency=2,  # Conservative concurrency
    )
    
    # Sample texts
    sample_texts = [
        "Dr. Alice Johnson works at TechCorp's research division in Boston.",
        "Bob Chen, the CEO of StartupInc, announced a partnership with GlobalTech.",
        "Professor Diana Martinez from MIT published research on quantum computing."
    ]
    
    chunks = TextChunks(
        chunks=[
            TextChunk(text=text, index=i)
            for i, text in enumerate(sample_texts)
        ]
    )
    
    print("Running entity extraction with API feedback rate limiting...")
    
    # Check status before
    status_before = api_feedback_limiter.get_status()
    print(f"Before extraction: {status_before['remaining_requests'] or 'Unknown'} requests available")
    
    # Run extraction (rate limiting happens automatically)
    graph = await extractor.run(chunks=chunks)
    
    # Check status after
    status_after = api_feedback_limiter.get_status()
    print(f"After extraction: {status_after['remaining_requests'] or 'Unknown'} requests available")
    
    print(f"Extracted {len(graph.nodes)} nodes and {len(graph.relationships)} relationships")
    print("✓ Rate limiting was applied automatically and accurately!")
    
    return graph, api_feedback_limiter


async def main():
    """Main function demonstrating different rate limiting approaches."""
    
    print("=== Basic Rate Limiting Example ===")
    print("Rate limiting is automatically applied to all LLM methods!")
    try:
        basic_rate_limiting_example()
    except Exception as e:
        print(f"Error in basic example: {e}")
    
    print("\n=== Provider-Specific Rate Limiting ===")
    print("All providers get automatic rate limiting when configured!")
    try:
        openai_llm, anthropic_llm, custom_llm = provider_specific_rate_limiting()
        print("Successfully created rate-limited LLMs for different providers")
    except Exception as e:
        print(f"Error in provider-specific example: {e}")
    
    print("\n=== Shared Rate Limiter Example ===")
    print("Multiple LLMs can share the same rate limiter automatically!")
    try:
        summarizer, reasoner = shared_rate_limiter_example()
        print("Successfully created LLMs with shared rate limiter")
    except Exception as e:
        print(f"Error in shared rate limiter example: {e}")
    
    print("\n=== Rate Limiting with Entity Extraction ===")
    print("Rate limiting works automatically with all neo4j-graphrag components!")
    try:
        await rate_limiting_with_entity_extraction()
    except Exception as e:
        print(f"Error in entity extraction example: {e}")
    
    print("\n=== Custom Rate Limiting Configurations ===")
    print("Different rate limiting strategies are automatically applied!")
    try:
        light_llm, heavy_llm, burst_llm = custom_rate_limiting_configuration()
        print("Successfully created custom rate-limited LLM configurations")
    except Exception as e:
        print(f"Error in custom configuration example: {e}")
    
    print("\n=== Token-Based Rate Limiting ===")
    print("Using actual token consumption from API responses!")
    try:
        token_based_rate_limiting_example()
    except Exception as e:
        print(f"Error in token-based rate limiting example: {e}")
    
    print("\n=== Comparing Slot vs Token Rate Limiting ===")
    try:
        comparing_slot_vs_token_rate_limiting()
    except Exception as e:
        print(f"Error in comparing slot vs token rate limiting example: {e}")
    
    print("\n=== Advanced Token Rate Limiting ===")
    try:
        high_throughput_llm, conservative_llm, burst_llm = advanced_token_rate_limiting()
        print("Successfully created advanced token-based rate-limited LLM configurations")
    except Exception as e:
        print(f"Error in advanced token rate limiting example: {e}")
    
    print("\n=== Realistic OpenAI Rate Limiting ===")
    try:
        llm = realistic_openai_rate_limiting()
        print("Successfully created realistic rate-limited LLM")
    except Exception as e:
        print(f"Error in realistic OpenAI rate limiting example: {e}")
    
    print("\n=== Provider-Specific Composite Rate Limiting ===")
    try:
        openai_llm, openai_turbo_llm, conservative_llm = provider_specific_composite_limiting()
        print("Successfully created composite rate-limited LLM configurations")
    except Exception as e:
        print(f"Error in provider-specific composite rate limiting example: {e}")
    
    try:
        llm = api_feedback_rate_limiting_example()
        print("Successfully created API feedback rate-limited LLM")
    except Exception as e:
        print(f"Error in API feedback rate limiting example: {e}")
    
    print("\n=== Comparing Rate Limiting Approaches ===")
    try:
        slot_llm, token_llm, api_feedback_llm = comparing_rate_limiting_approaches()
        print("Successfully created rate-limited LLM configurations")
    except Exception as e:
        print(f"Error in comparing rate limiting approaches example: {e}")
    
    print("\n=== Advanced API Feedback Scenarios ===")
    try:
        high_throughput_llm, conservative_llm, summarizer, analyzer = advanced_api_feedback_scenarios()
        print("Successfully created advanced API feedback rate-limited LLM configurations")
    except Exception as e:
        print(f"Error in advanced API feedback scenarios example: {e}")
    
    print("\n=== API Feedback Debugging ===")
    try:
        llm, debug_limiter = api_feedback_debugging_example()
        print("Successfully created API feedback rate-limited LLM")
    except Exception as e:
        print(f"Error in API feedback debugging example: {e}")
    
    try:
        graph, api_feedback_limiter = api_feedback_with_entity_extraction()
        print("Successfully created API feedback rate-limited LLM")
    except Exception as e:
        print(f"Error in API feedback with entity extraction example: {e}")
    
    print("\n=== Summary ===")
    print("✓ Rate limiting is now AUTOMATIC for all LLM providers!")
    print("✓ No need to manually apply rate limiting in implementations")
    print("✓ Works with invoke, ainvoke, invoke_with_tools, ainvoke_with_tools")
    print("✓ All existing and future LLM providers get rate limiting for free")


if __name__ == "__main__":
    # Note: Make sure to set your API keys:
    # export OPENAI_API_KEY="your-openai-api-key"
    # export ANTHROPIC_API_KEY="your-anthropic-api-key"
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Some examples may fail.")
    
    asyncio.run(main()) 