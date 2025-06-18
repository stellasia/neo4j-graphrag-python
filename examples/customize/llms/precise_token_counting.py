"""
Precise Token Counting with OpenAI

This example demonstrates how the OpenAI LLM implementation uses tiktoken
to count exact input tokens for precise rate limiting, rather than relying
on rough estimates.

This makes rate limiting much more accurate and efficient.
"""

import asyncio
import os
from typing import Dict, Any

from neo4j_graphrag.llm import (
    OpenAILLM,
    TokenBucketRateLimiter,
    CompositeRateLimiter,
    SlotBucketRateLimiter,
    APIFeedbackRateLimiter,
    RetryConfig,
)
from neo4j_graphrag.tool import Tool


def demonstrate_precise_token_counting():
    """Show how OpenAI uses precise token counting for rate limiting."""
    
    print("=== Precise Token Counting with OpenAI ===")
    
    # Create a token-based rate limiter
    token_limiter = TokenBucketRateLimiter(
        tokens_per_second=500.0,  # 30k tokens per minute
        max_tokens=2000,          # Burst capacity
    )
    
    # Create OpenAI LLM with token-based rate limiting
    llm = OpenAILLM(
        model_name="gpt-3.5-turbo",
        rate_limiter=token_limiter,
    )
    
    print(f"Created OpenAI LLM with precise token counting")
    print(f"Tiktoken available: {hasattr(llm, '_tokenizer') and llm._tokenizer is not None}")
    
    # Test with different input sizes
    test_inputs = [
        "Hello world!",  # Short input
        "Explain quantum computing in simple terms with examples and applications.",  # Medium input
        "Write a detailed explanation of machine learning, including supervised learning, unsupervised learning, reinforcement learning, neural networks, deep learning, and provide examples of each with real-world applications. Include the mathematical foundations and key algorithms." * 2,  # Long input
    ]
    
    for i, input_text in enumerate(test_inputs, 1):
        print(f"\\nTest {i}: Input length = {len(input_text)} characters")
        
        # Show precise token count if available
        if hasattr(llm, '_estimate_input_tokens'):
            estimated_tokens = llm._estimate_input_tokens(input_text)
            print(f"Precise token estimate: {estimated_tokens} tokens")
        
        # Check rate limiter status before request
        status_before = token_limiter.get_status()
        print(f"Tokens available before: {status_before.get('available_tokens', 'N/A')}")
        
        try:
            response = llm.invoke(input_text)
            print(f"Response: {response.content[:100]}...")
            
            # Check status after request
            status_after = token_limiter.get_status()
            print(f"Tokens available after: {status_after.get('available_tokens', 'N/A')}")
            
        except Exception as e:
            print(f"Error: {e}")


def compare_estimation_vs_actual():
    """Compare estimated tokens vs actual tokens used."""
    
    print("\\n=== Estimation vs Actual Token Usage ===")
    
    # Create rate limiter that tracks both estimates and actual usage
    token_limiter = TokenBucketRateLimiter(
        tokens_per_second=1000.0,
        max_tokens=5000,
    )
    
    llm = OpenAILLM(
        model_name="gpt-3.5-turbo",
        rate_limiter=token_limiter,
    )
    
    test_cases = [
        ("Simple question", "What is 2+2?"),
        ("Complex question", "Explain the differences between machine learning and artificial intelligence, including their applications and limitations."),
        ("With context", "Based on the previous discussion about AI, what are the ethical implications?"),
    ]
    
    for name, input_text in test_cases:
        print(f"\\n{name}:")
        
        # Get precise estimate
        if hasattr(llm, '_estimate_input_tokens'):
            estimated = llm._estimate_input_tokens(input_text)
            print(f"  Estimated input tokens: {estimated}")
        
        try:
            response = llm.invoke(input_text)
            print(f"  Response: {response.content[:80]}...")
            
            # Note: Actual token usage would be updated in the rate limiter
            # from the API response if using a TokenTrackingRateLimiter
            
        except Exception as e:
            print(f"  Error: {e}")


def demonstrate_tool_calling_token_counting():
    """Show token counting with tool calling (includes tool definitions)."""
    
    print("\\n=== Token Counting with Tool Calling ===")
    
    # Create a simple tool for demonstration
    class WeatherTool(Tool):
        def get_name(self) -> str:
            return "get_weather"
        
        def get_description(self) -> str:
            return "Get current weather information for a location"
        
        def get_parameters(self) -> Dict[str, Any]:
            return {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        
        def invoke(self, **kwargs) -> str:
            return f"Weather in {kwargs.get('location', 'unknown')}: 22°C, sunny"
    
    # Create rate limiter
    composite_limiter = CompositeRateLimiter([
        SlotBucketRateLimiter(slots_per_second=2.0, max_slots=5),  # Request limiting
        TokenBucketRateLimiter(tokens_per_second=800.0, max_tokens=3000),  # Token limiting
    ])
    
    llm = OpenAILLM(
        model_name="gpt-3.5-turbo",
        rate_limiter=composite_limiter,
    )
    
    tools = [WeatherTool()]
    
    print("Testing token counting with tool definitions...")
    
    # Estimate tokens including tool definitions
    if hasattr(llm, '_estimate_input_tokens'):
        input_text = "What's the weather like in San Francisco?"
        
        # Tokens without tools
        tokens_without_tools = llm._estimate_input_tokens(input_text)
        print(f"Tokens without tools: {tokens_without_tools}")
        
        # Tokens with tools (includes tool definitions)
        tokens_with_tools = llm._estimate_input_tokens(input_text, tools=tools)
        print(f"Tokens with tools: {tokens_with_tools}")
        
        tool_overhead = tokens_with_tools - tokens_without_tools
        print(f"Tool definition overhead: {tool_overhead} tokens")
    
    try:
        response = llm.invoke_with_tools(
            "What's the weather like in San Francisco?",
            tools=tools
        )
        print(f"Tool response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")


def demonstrate_api_feedback_with_precise_counting():
    """Show combination of precise counting + API feedback for ultimate accuracy."""
    
    print("\\n=== API Feedback + Precise Counting ===")
    
    # Use API feedback rate limiter for real-time sync with OpenAI headers
    api_feedback_limiter = APIFeedbackRateLimiter(
        fallback_requests_per_second=8.0,
        fallback_tokens_per_second=500.0,
        estimated_tokens_per_request=400,  # This will be overridden by precise counting
    )
    
    llm = OpenAILLM(
        model_name="gpt-3.5-turbo",
        rate_limiter=api_feedback_limiter,
    )
    
    print("Using API feedback limiter with precise token counting")
    print("This provides the most accurate rate limiting possible!")
    
    # Test with a few requests
    test_prompts = [
        "Explain photosynthesis briefly.",
        "What are the main principles of quantum mechanics?",
        "Describe the process of machine learning model training.",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\\nRequest {i}: {prompt[:50]}...")
        
        # Show precise estimate
        if hasattr(llm, '_estimate_input_tokens'):
            estimated = llm._estimate_input_tokens(prompt)
            print(f"  Precise estimate: {estimated} tokens")
        
        # Show rate limiter status
        status = api_feedback_limiter.get_status()
        print(f"  API feedback available: {status.get('has_fresh_feedback', False)}")
        
        try:
            response = llm.invoke(prompt)
            print(f"  Response: {response.content[:80]}...")
            
            # Show updated status after API response
            updated_status = api_feedback_limiter.get_status()
            print(f"  Updated feedback: {updated_status.get('has_fresh_feedback', False)}")
            
        except Exception as e:
            print(f"  Error: {e}")


async def async_precise_token_counting():
    """Demonstrate async precise token counting."""
    
    print("\\n=== Async Precise Token Counting ===")
    
    token_limiter = TokenBucketRateLimiter(
        tokens_per_second=600.0,
        max_tokens=2000,
    )
    
    llm = OpenAILLM(
        model_name="gpt-3.5-turbo",
        rate_limiter=token_limiter,
    )
    
    # Create multiple concurrent requests with different token requirements
    tasks = []
    prompts = [
        "Short question?",  # Low tokens
        "Medium length question about artificial intelligence and its applications?",  # Medium tokens
        "Very long and detailed question about the implications of machine learning, artificial intelligence, deep learning, neural networks, and their impact on society, economy, and future technological development?",  # High tokens
    ]
    
    print("Creating concurrent requests with different token requirements...")
    
    for i, prompt in enumerate(prompts):
        if hasattr(llm, '_estimate_input_tokens'):
            estimated = llm._estimate_input_tokens(prompt)
            print(f"Task {i+1} estimated tokens: {estimated}")
        
        task = asyncio.create_task(llm.ainvoke(prompt))
        tasks.append(task)
    
    # Wait for all requests
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i+1} failed: {result}")
        else:
            print(f"Task {i+1} succeeded: {result.content[:50]}...")
    
    print(f"Final rate limiter status: {token_limiter.get_status()}")


def main():
    """Run all precise token counting examples."""
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Some examples may not work.")
        print("Set it with: export OPENAI_API_KEY=your-api-key")
        return
    
    # Run sync examples
    demonstrate_precise_token_counting()
    compare_estimation_vs_actual()
    demonstrate_tool_calling_token_counting()
    demonstrate_api_feedback_with_precise_counting()
    
    # Run async example
    asyncio.run(async_precise_token_counting())
    
    print("\\n=== All Precise Token Counting Examples Complete! ===")
    print("\\nKey Benefits:")
    print("✅ Exact token counting with tiktoken")
    print("✅ No more rough estimates or guessing")
    print("✅ Efficient rate limiting - no wasted quota")
    print("✅ Works with tools, context, and system instructions")
    print("✅ Combines with API feedback for ultimate accuracy")


if __name__ == "__main__":
    main() 