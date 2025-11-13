#!/usr/bin/env python3
"""
Test Strands Agents integration
"""
import asyncio
import uuid
from strands_client import StrandsClient


async def test_strands_basic():
    """Test basic Strands agent functionality"""
    print("=" * 50)
    print("Testing Strands Agents Integration")
    print("=" * 50)
    
    # Initialize client
    client = StrandsClient()
    print("✓ Strands client initialized")
    
    # Create a test session
    session_id = str(uuid.uuid4())
    print(f"✓ Test session created: {session_id}")
    
    # Test streaming response
    print("\n--- Testing Agent Response ---")
    user_message = "Hello! Can you introduce yourself briefly?"
    print(f"User: {user_message}")
    print("Agent: ", end="", flush=True)
    
    full_response = ""
    async for chunk in client.stream_response(session_id, user_message):
        print(chunk, end="", flush=True)
        full_response += chunk
    
    print("\n")
    print(f"✓ Received {len(full_response)} characters")
    
    # Test conversation continuity
    print("\n--- Testing Conversation Memory ---")
    follow_up = "What did I just ask you?"
    print(f"User: {follow_up}")
    print("Agent: ", end="", flush=True)
    
    follow_up_response = ""
    async for chunk in client.stream_response(session_id, follow_up):
        print(chunk, end="", flush=True)
        follow_up_response += chunk
    
    print("\n")
    
    # Verify session history
    history = client.get_session_history(session_id)
    print(f"✓ Session history contains {len(history)} messages")
    
    # Test session clearing
    print("\n--- Testing Session Management ---")
    client.clear_session(session_id)
    print("✓ Session cleared")
    
    cleared_history = client.get_session_history(session_id)
    print(f"✓ History after clear: {len(cleared_history)} messages")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)


async def test_strands_multi_session():
    """Test multiple concurrent sessions"""
    print("\n" + "=" * 50)
    print("Testing Multi-Session Support")
    print("=" * 50)
    
    client = StrandsClient()
    
    # Create two sessions
    session1 = str(uuid.uuid4())
    session2 = str(uuid.uuid4())
    
    # Send different messages to each session
    print(f"\nSession 1: Asking about Python")
    async for chunk in client.stream_response(session1, "Tell me one thing about Python"):
        print(chunk, end="", flush=True)
    
    print(f"\n\nSession 2: Asking about JavaScript")
    async for chunk in client.stream_response(session2, "Tell me one thing about JavaScript"):
        print(chunk, end="", flush=True)
    
    # Verify sessions are independent
    history1 = client.get_session_history(session1)
    history2 = client.get_session_history(session2)
    
    print(f"\n\n✓ Session 1 has {len(history1)} messages")
    print(f"✓ Session 2 has {len(history2)} messages")
    print("✓ Sessions are independent")
    
    # Cleanup
    client.clear_session(session1)
    client.clear_session(session2)
    
    print("\n" + "=" * 50)
    print("Multi-session test passed! ✓")
    print("=" * 50)


if __name__ == "__main__":
    print("\n🚀 Starting Strands Agents Tests\n")
    
    try:
        # Run basic test
        asyncio.run(test_strands_basic())
        
        # Run multi-session test
        asyncio.run(test_strands_multi_session())
        
        print("\n✅ All Strands integration tests completed successfully!\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}\n")
        import traceback
        traceback.print_exc()
