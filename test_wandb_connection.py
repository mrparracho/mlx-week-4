#!/usr/bin/env python3
"""
Test wandb connection and permissions before running the sweep.
"""

import wandb
import os

def test_wandb_connection():
    """Test wandb connection and permissions"""
    
    print("🔍 Testing Wandb Connection...")
    
    # Check if API key is set
    wandb_token = os.getenv('WANDB_API_KEY')
    if not wandb_token:
        print("❌ WANDB_API_KEY environment variable not set!")
        print("Please set your wandb API key:")
        print("export WANDB_API_KEY=your_api_key_here")
        return False
    
    print(f"✅ API key found: {wandb_token[:8]}...")
    
    try:
        # Try to login
        wandb.login(key=wandb_token)
        print("✅ Login successful")
        
        # Try to create a simple test run
        with wandb.init(
            project="test-connection",
            entity="mrparracho",
            name="connection-test",
            config={"test": True}
        ) as run:
            print(f"✅ Test run created: {run.id}")
            wandb.log({"test_metric": 1.0})
            print("✅ Test metric logged successfully")
        
        print("🎉 All tests passed! Wandb connection is working.")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Check if your API key is correct")
        print("2. Verify you have access to the 'mrparracho' entity")
        print("3. Try creating the project manually on wandb.ai first")
        print("4. Check if your API key has the right permissions")
        return False

if __name__ == "__main__":
    test_wandb_connection() 