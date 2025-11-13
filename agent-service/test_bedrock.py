#!/usr/bin/env python3
"""
Simple test script to verify Bedrock connectivity
Run this to test your AWS credentials and Bedrock access
"""
import boto3
import sys
from config import AWS_REGION, BEDROCK_MODEL_ID

def test_bedrock_connection():
    """Test Bedrock connection and model access"""
    print("=" * 50)
    print("Testing AWS Bedrock Connection")
    print("=" * 50)
    print()
    
    print(f"Region: {AWS_REGION}")
    print(f"Model: {BEDROCK_MODEL_ID}")
    print()
    
    try:
        # Create Bedrock client
        print("Creating Bedrock client...")
        client = boto3.client('bedrock-runtime', region_name=AWS_REGION)
        print("✓ Client created successfully")
        print()
        
        # Test simple inference
        print("Testing model inference...")
        response = client.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": "Say 'Hello, I am working!' and nothing else."}]
                }
            ],
            inferenceConfig={
                "maxTokens": 50,
                "temperature": 0.7
            }
        )
        
        # Extract response
        output_text = response['output']['message']['content'][0]['text']
        print(f"✓ Model response: {output_text}")
        print()
        
        # Test streaming
        print("Testing streaming inference...")
        stream_response = client.converse_stream(
            modelId=BEDROCK_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": "Count from 1 to 5."}]
                }
            ],
            inferenceConfig={
                "maxTokens": 50,
                "temperature": 0.7
            }
        )
        
        print("✓ Streaming response: ", end="")
        for event in stream_response['stream']:
            if 'contentBlockDelta' in event:
                delta = event['contentBlockDelta']['delta']
                if 'text' in delta:
                    print(delta['text'], end="", flush=True)
        print()
        print()
        
        print("=" * 50)
        print("✓ All tests passed!")
        print("=" * 50)
        print()
        print("Your Bedrock setup is working correctly.")
        print("You can now start the voice agent service.")
        return True
        
    except Exception as e:
        print()
        print("=" * 50)
        print("✗ Test failed!")
        print("=" * 50)
        print()
        print(f"Error: {str(e)}")
        print()
        print("Common issues:")
        print("1. AWS credentials not set or invalid")
        print("2. Bedrock model access not enabled")
        print("3. Wrong AWS region")
        print("4. IAM permissions missing")
        print()
        print("Please check:")
        print("- .env file has correct AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        print("- Bedrock model access is enabled in AWS Console")
        print("- IAM user has bedrock:InvokeModel permission")
        print()
        return False

if __name__ == "__main__":
    success = test_bedrock_connection()
    sys.exit(0 if success else 1)
