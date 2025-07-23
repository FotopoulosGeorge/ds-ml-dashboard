# src/ml/performance_decorator.py
import streamlit as st
import functools
import inspect
import time
import psutil
import threading
import queue
from datetime import datetime
from typing import Callable, Any, Optional, Dict, List

class MLPerformanceTracker:
    """
    Enhanced core class for tracking ML operation performance with memory monitoring
    """
    
    def __init__(self):
        # Initialize process for memory tracking
        try:
            self.process = psutil.Process()
            self.memory_tracking_available = True
        except ImportError:
            self.memory_tracking_available = False
            st.warning("⚠️ psutil not available - memory tracking disabled")
        
        self.operation_estimates = {
            "supervised_learning": {
                "base_time": 30,
                "row_factor": 0.001,
                "col_factor": 0.5,
                "memory_factor": 2.0,  # MB per 1000 rows
                "model_complexity": {
                    "simple": 1.0,
                    "medium": 1.5, 
                    "complex": 2.5
                }
            },
            "ensemble": {
                "base_time": 45,
                "row_factor": 0.002,
                "col_factor": 0.3,
                "memory_factor": 5.0,
                "model_factor": 20,
                "method_complexity": {
                    "voting": 1.0,
                    "bagging": 1.5,
                    "stacking": 3.0
                }
            },
            "automl": {
                "base_time": 120,
                "row_factor": 0.005,
                "col_factor": 0.8,
                "memory_factor": 10.0,  # Higher memory usage
                "search_complexity": {
                    "quick": 1.0,
                    "thorough": 3.0
                }
            },
            "clustering": {
                "base_time": 20,
                "row_factor": 0.001,
                "col_factor": 0.2,
                "memory_factor": 3.0,
                "n_clusters_factor": 5
            },
            "time_series": {
                "base_time": 60,
                "row_factor": 0.002,
                "memory_factor": 4.0,
                "seasonality_factor": 1.5
            },
            "anomaly_detection": {
                "base_time": 30,
                "row_factor": 0.002,
                "memory_factor": 3.5,
                "algorithm_complexity": {
                    "isolation_forest": 1.0,
                    "lof": 1.8,
                    "one_class_svm": 2.5
                }
            }
        }
    
    def get_current_memory_usage(self):
        """Get current memory usage in MB"""
        if not self.memory_tracking_available:
            return 0
        
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0
    
    def get_system_memory_info(self):
        """Get system memory information"""
        if not self.memory_tracking_available:
            return {"available": 8192, "total": 16384, "percent": 50}
        
        try:
            memory = psutil.virtual_memory()
            return {
                "available": memory.available / 1024 / 1024,  # MB
                "total": memory.total / 1024 / 1024,  # MB
                "percent": memory.percent
            }
        except Exception:
            return {"available": 8192, "total": 16384, "percent": 50}
    
    def estimate_memory_usage(self, operation_type: str, dataset_size: tuple, config: Dict = None) -> Dict:
        """Estimate memory usage for an ML operation"""
        if operation_type not in self.operation_estimates:
            return {"estimated_memory": 100, "warnings": []}
        
        estimates = self.operation_estimates[operation_type]
        rows, cols = dataset_size
        config = config or {}
        
        # Base memory calculation
        base_memory = estimates.get("memory_factor", 2.0) * (rows / 1000)  # MB per 1000 rows
        feature_memory = cols * 0.1  # Small memory per feature
        
        estimated_memory = base_memory + feature_memory
        
        warnings = []
        
        # Operation-specific memory adjustments
        if operation_type == "automl":
            search_type = config.get("search_strategy", "thorough").lower()
            if search_type == "thorough":
                estimated_memory *= 3  # AutoML uses much more memory
        
        elif operation_type == "ensemble":
            n_models = config.get("n_models", 3)
            estimated_memory *= n_models  # Each model uses memory
        
        # System memory check
        system_memory = self.get_system_memory_info()
        available_memory = system_memory["available"]
        
        if estimated_memory > available_memory * 0.8:
            warnings.append(f"High memory usage expected ({estimated_memory:.0f}MB) - may cause system slowdown")
        
        if estimated_memory > available_memory:
            warnings.append("⚠️ CRITICAL: Estimated memory exceeds available memory - operation may fail")
        
        return {
            "estimated_memory": int(estimated_memory),
            "available_memory": int(available_memory),
            "memory_warnings": warnings
        }
    
    def estimate_time(self, operation_type: str, dataset_size: tuple, config: Dict = None) -> Dict:
        """
        Enhanced time estimation with memory considerations
        """
        if operation_type not in self.operation_estimates:
            return {
                "min_time": 30,
                "max_time": 300,
                "estimated_time": 120,
                "confidence": "Low",
                "warnings": [f"Unknown operation type: {operation_type}"]
            }
        
        estimates = self.operation_estimates[operation_type]
        rows, cols = dataset_size
        config = config or {}
        
        # Base calculation (same as before)
        estimated_time = estimates["base_time"]
        estimated_time += rows * estimates.get("row_factor", 0)
        estimated_time += cols * estimates.get("col_factor", 0)
        
        warnings = []
        
        # Memory-based time adjustments
        memory_estimate = self.estimate_memory_usage(operation_type, dataset_size, config)
        if memory_estimate["estimated_memory"] > memory_estimate["available_memory"] * 0.8:
            estimated_time *= 1.5  # Slower when memory constrained
            warnings.append("Performance may be degraded due to high memory usage")
        
        # Rest of the original logic...
        if operation_type == "ensemble":
            n_models = config.get("n_models", 3)
            estimated_time += n_models * estimates.get("model_factor", 20)
            
            method = config.get("method", "voting").lower()
            if method in estimates.get("method_complexity", {}):
                estimated_time *= estimates["method_complexity"][method]
        
        # Data size warnings
        if rows > 50000:
            warnings.append(f"Large dataset ({rows:,} rows) - execution may be slow")
            estimated_time *= 1.3
            
        if rows > 100000:
            warnings.append("Very large dataset - consider sampling")
            estimated_time *= 1.8
        
        # Calculate range
        min_time = max(10, int(estimated_time * 0.7))
        max_time = int(estimated_time * 1.4)
        
        confidence = "High" if len(warnings) == 0 else "Medium" if len(warnings) <= 2 else "Low"
        
        return {
            "min_time": min_time,
            "max_time": max_time,
            "estimated_time": int(estimated_time),
            "confidence": confidence,
            "warnings": warnings,
            "memory_estimate": memory_estimate
        }


def ml_performance(
    operation_type: str,
    dataset_param: Optional[str] = None,
    config_params: Optional[List[str]] = None,
    show_estimate: bool = True,
    min_time_threshold: int = 60,
    track_memory: bool = True
):
    """
    Enhanced decorator with memory tracking
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Skip performance tracking if disabled
            if not show_estimate or not _should_show_performance_ui():
                return func(*args, **kwargs)
            
            try:
                # Extract parameters for performance estimation
                dataset_size, config = _extract_parameters_enhanced(
                    func, args, kwargs, dataset_param, config_params
                )
                
                # Get time and memory estimates
                tracker = MLPerformanceTracker()
                estimate = tracker.estimate_time(operation_type, dataset_size, config)
                
                # Show performance UI if estimate exceeds threshold
                if estimate["estimated_time"] >= min_time_threshold:
                    should_proceed = _show_enhanced_performance_ui(estimate, operation_type, tracker)
                    if not should_proceed:
                        st.warning("⚠️ Operation cancelled by user")
                        return None
                
                # Execute with enhanced tracking
                return _execute_with_enhanced_tracking(
                    func, args, kwargs, estimate, operation_type, track_memory, tracker
                )
                
            except Exception as e:
                st.warning(f"Performance tracking failed: {str(e)}")
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def _extract_parameters_enhanced(func, args, kwargs, dataset_param, config_params):
    """Enhanced parameter extraction with better fallbacks"""
    
    # Get function signature
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    
    # Multiple strategies to find dataset
    dataset_size = (1000, 10)  # Default fallback
    
    # Strategy 1: Explicit dataset parameter
    if dataset_param and dataset_param in bound_args.arguments:
        dataset = bound_args.arguments[dataset_param]
        if hasattr(dataset, 'shape'):
            dataset_size = dataset.shape
    
    # Strategy 2: Look for common dataset parameter names
    elif not dataset_param:
        dataset_candidates = ['df', 'data', 'X', 'dataset', 'dataframe', 'current_data']
        for candidate in dataset_candidates:
            if candidate in bound_args.arguments:
                dataset = bound_args.arguments[candidate]
                if hasattr(dataset, 'shape'):
                    dataset_size = dataset.shape
                    break
    
    # Strategy 3: Check session state
    if dataset_size == (1000, 10) and hasattr(st, 'session_state') and 'working_df' in st.session_state:
        df = st.session_state.working_df
        if hasattr(df, 'shape'):
            dataset_size = df.shape
    
    # Extract configuration with better error handling
    config = {}
    if config_params:
        for param in config_params:
            if param in bound_args.arguments:
                config[param] = bound_args.arguments[param]
    
    return dataset_size, config


def _show_enhanced_performance_ui(estimate: Dict, operation_type: str, tracker: MLPerformanceTracker) -> bool:
    """Enhanced performance UI with memory information"""
    
    # Format time display (same as before)
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            if remaining_seconds > 0:
                return f"{minutes}m {remaining_seconds}s"
            else:
                return f"{minutes}m"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    min_time_str = format_time(estimate['min_time'])
    max_time_str = format_time(estimate['max_time'])
    
    # Enhanced UI with memory info
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        # Time estimates
        if estimate['max_time'] < 120:
            st.info(f"⚡ **Estimated Time:** {min_time_str} - {max_time_str}")
        elif estimate['max_time'] < 600:
            st.warning(f"⏳ **Estimated Time:** {min_time_str} - {max_time_str}")
        else:
            st.error(f"🐌 **Long Execution:** {min_time_str} - {max_time_str}")
    
    with perf_col2:
        # Memory estimates
        memory_info = estimate.get('memory_estimate', {})
        est_memory = memory_info.get('estimated_memory', 0)
        avail_memory = memory_info.get('available_memory', 0)
        
        if est_memory > 0:
            memory_pct = (est_memory / avail_memory * 100) if avail_memory > 0 else 0
            
            if memory_pct < 50:
                st.info(f"🧠 **Memory:** ~{est_memory:.0f}MB ({memory_pct:.0f}%)")
            elif memory_pct < 80:
                st.warning(f"🧠 **Memory:** ~{est_memory:.0f}MB ({memory_pct:.0f}%)")
            else:
                st.error(f"🧠 **High Memory:** ~{est_memory:.0f}MB ({memory_pct:.0f}%)")
    
    # Show warnings
    all_warnings = estimate.get('warnings', []) + memory_info.get('memory_warnings', [])
    if all_warnings:
        with st.expander("⚠️ Performance Warnings", expanded=True):
            for warning in all_warnings:
                st.write(f"• {warning}")
    
    # System info
    with st.expander("💻 System Information"):
        system_info = tracker.get_system_memory_info()
        current_memory = tracker.get_current_memory_usage()
        
        sys_col1, sys_col2, sys_col3 = st.columns(3)
        with sys_col1:
            st.metric("Available Memory", f"{system_info['available']:.0f} MB")
        with sys_col2:
            st.metric("Current Usage", f"{current_memory:.0f} MB")
        with sys_col3:
            st.metric("System Memory", f"{system_info['percent']:.0f}% used")
    
    # Get confirmation for very long operations or high memory usage
    memory_pct = (est_memory / avail_memory * 100) if avail_memory > 0 else 0
    needs_confirmation = estimate['max_time'] > 600 or memory_pct > 80
    
    if needs_confirmation:
        st.error("⚠️ **Resource Intensive Operation Detected**")
        
        confirm_key = f"confirm_{operation_type}_{int(time.time())}"
        
        if estimate['max_time'] > 600:
            st.write(f"• **Long execution time:** May take {estimate['max_time']//60}+ minutes")
        if memory_pct > 80:
            st.write(f"• **High memory usage:** May use {memory_pct:.0f}% of available memory")
        
        confirm = st.checkbox(
            "✅ **I understand the resource requirements and want to proceed**",
            key=confirm_key
        )
        return confirm
    
    return True


def _execute_with_enhanced_tracking(func: Callable, args: tuple, kwargs: dict, 
                                   estimate: Dict, operation_type: str, 
                                   track_memory: bool, tracker: MLPerformanceTracker):
    """Execute function with enhanced progress and memory tracking"""
    
    operation_name = operation_type.replace("_", " ").title()
    
    # Get initial memory usage
    initial_memory = tracker.get_current_memory_usage() if track_memory else 0
    start_time = time.time()
    # Progress tracking for long operations
    if estimate["estimated_time"] > 120:  # 2+ minutes
        progress_bar = st.progress(0)
        status_text = st.empty()
        memory_text = st.empty()
        
        # Use threading for progress monitoring
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def worker():
            try:
                result = func(*args, **kwargs)
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)
        
        def monitor_progress():
            start_time = time.time()
            estimated_duration = estimate["estimated_time"]
            
            thread = threading.Thread(target=worker)
            thread.start()
            
            while thread.is_alive():
                elapsed = time.time() - start_time
                progress = min(elapsed / estimated_duration, 0.95)  # Cap at 95%
                
                progress_bar.progress(progress)
                status_text.text(f'⏳ {operation_name} running... ({elapsed:.0f}s elapsed)')
                
                if track_memory:
                    current_memory = tracker.get_current_memory_usage()
                    memory_used = current_memory - initial_memory
                    memory_text.text(f'🧠 Memory: {current_memory:.0f}MB (+{memory_used:.0f}MB)')
                
                time.sleep(1)
            
            thread.join()
            progress_bar.progress(1.0)
            
            if not result_queue.empty():
                return result_queue.get()
            elif not exception_queue.empty():
                raise exception_queue.get()
        
        result = monitor_progress()
    
    else:
        # Regular execution with spinner
        with st.spinner(f'Executing {operation_name}...'):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                execution_time = time.time() - start_time
                final_memory = tracker.get_current_memory_usage() if track_memory else 0
                memory_used = final_memory - initial_memory
                
                st.error(f"❌ {operation_name} failed after {execution_time:.1f}s (Memory: +{memory_used:.1f}MB)")
                raise
    
    # Show completion metrics
    execution_time = time.time() - start_time
    final_memory = tracker.get_current_memory_usage() if track_memory else 0
    memory_used = final_memory - initial_memory if track_memory else 0
    
    if execution_time > 5:
        if track_memory and memory_used > 10:
            st.success(f"✅ {operation_name} completed in {execution_time:.1f}s (Memory: +{memory_used:.1f}MB)")
        else:
            st.success(f"✅ {operation_name} completed in {execution_time:.1f}s")
    
    # Store performance data for learning
    _store_performance_data(operation_type, estimate, execution_time, memory_used)
    
    return result


def _store_performance_data(operation_type: str, estimate: Dict, actual_time: float, memory_used: float):
    """Store performance data for future estimate improvements"""
    if not hasattr(st.session_state, 'performance_history'):
        st.session_state.performance_history = []
    
    performance_record = {
        'operation_type': operation_type,
        'estimated_time': estimate.get('estimated_time', 0),
        'actual_time': actual_time,
        'memory_used': memory_used,
        'timestamp': datetime.now().isoformat()
    }
    
    st.session_state.performance_history.append(performance_record)
    
    # Keep only recent records (last 50)
    if len(st.session_state.performance_history) > 50:
        st.session_state.performance_history = st.session_state.performance_history[-50:]


def _should_show_performance_ui() -> bool:
    """Check if we should show performance UI"""
    try:
        if hasattr(st.session_state, 'disable_performance_tracking'):
            return not st.session_state.disable_performance_tracking
        return True
    except:
        return True


# Convenience decorators for your specific ML operations
def supervised_learning_performance(dataset_param="df", config_params=None):
    """Decorator for supervised learning operations"""
    return ml_performance(
        "supervised_learning", 
        dataset_param=dataset_param, 
        config_params=config_params or ["algorithm", "cv_folds"]
    )

def automl_performance(dataset_param="X", config_params=None):
    """Decorator for AutoML operations"""
    return ml_performance(
        "automl",
        dataset_param=dataset_param,
        config_params=config_params or ["search_strategy", "max_time"]
    )

def time_series_performance(dataset_param="df", config_params=None):
    """Decorator for time series operations"""
    return ml_performance(
        "time_series",
        dataset_param=dataset_param,
        config_params=config_params or ["forecast_periods", "freq"]
    )

def clustering_performance(dataset_param="df", config_params=None):
    """Decorator for clustering operations"""
    return ml_performance(
        "clustering",
        dataset_param=dataset_param,
        config_params=config_params or ["n_clusters", "algorithm"]
    )

def anomaly_detection_performance(dataset_param="df", config_params=None):
    """Decorator for anomaly detection operations"""
    return ml_performance(
        "anomaly_detection",
        dataset_param=dataset_param,
        config_params=config_params or ["algorithm", "contamination"]
    )