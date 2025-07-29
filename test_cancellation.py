# Create a test script: test_cancellation.py
import time
import threading
from src.demo.demo_datasets import DemoDatasets
from src.ml.automl import AutoMLEngine

def test_long_operation():
    """Test that mimics a long AutoML operation"""
    print("🧪 Starting long operation test...")
    
    # Load large dataset
    df = DemoDatasets.load_dataset("🏠 California Housing")  # 20k+ rows
    
    # Start AutoML with thorough search (should take 10+ minutes)
    automl = AutoMLEngine()
    
    print("⏳ Starting AutoML (this should take 10+ minutes)...")
    print("🚨 TESTING: Close your browser or navigate away after 30 seconds")
    print("💻 MONITORING: Watch system resources (CPU, memory)")
    
    # This should timeout/cancel properly
    try:
        automl.render_automl_tab(df)
    except Exception as e:
        print(f"Operation ended with: {e}")

if __name__ == "__main__":
    test_long_operation()