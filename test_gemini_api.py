"""
Quick test script to verify Gemini API with new SDK
"""
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gemini_api():
    """Test the new Gemini API"""
    print("=" * 50)
    print("Testing Gemini API with New SDK")
    print("=" * 50)
    
    # Get API key
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not found in environment variables")
        print("Please add it to your .env file")
        return False
    
    print(f"✓ API Key found: {api_key[:10]}...")
    
    try:
        # Initialize client
        print("\nInitializing Gemini client...")
        client = genai.Client(api_key=api_key)
        print("✓ Client initialized successfully")
        
        # Test generation
        print("\nTesting content generation...")
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents='Say hello and confirm you are working with the new SDK!',
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=100,
            )
        )
        
        print(f"✓ Response received: {response.text[:100]}...")
        
        print("\n" + "=" * 50)
        print("✅ SUCCESS! Gemini API is working correctly!")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print("\nTroubleshooting:")
        print("1. Check your API key is valid")
        print("2. Ensure google-genai is installed: pip install google-genai")
        print("3. Check your internet connection")
        print("4. Verify API quota is not exceeded")
        return False

if __name__ == '__main__':
    success = test_gemini_api()
    exit(0 if success else 1)