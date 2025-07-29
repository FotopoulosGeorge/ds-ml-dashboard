# test_memory_monitoring.py
import psutil
import time
import streamlit as st
from src.ml.ml_utils import MLUtils

def monitor_system_resources():
    """Monitor system resources during testing"""
    process = psutil.Process()
    
    print("📊 Starting memory monitoring...")
    print("Timestamp | Memory (MB) | CPU % | Active Threads")
    print("-" * 55)
    
    for i in range(60):  # Monitor for 60 seconds
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        # Count active threads (if available)
        thread_count = len(getattr(st.session_state, 'active_threads', []))
        
        print(f"{time.strftime('%H:%M:%S')} | {memory_mb:8.1f} | {cpu_percent:5.1f} | {thread_count:6d}")
        
        time.sleep(1)

if __name__ == "__main__":
    monitor_system_resources()