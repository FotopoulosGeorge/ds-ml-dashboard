# src/ml/performance_utils.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psutil
import platform

class PerformanceEstimator:
    """
    Estimate execution times and provide performance warnings
    """
    
    @staticmethod
    def get_system_info():
        """Get basic system information"""
        try:
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else None
            
            return {
                'cpu_cores': cpu_count,
                'memory_gb': round(memory_gb, 1),
                'cpu_freq_mhz': round(cpu_freq) if cpu_freq else None,
                'platform': platform.system()
            }
        except:
            return {
                'cpu_cores': 'Unknown',
                'memory_gb': 'Unknown', 
                'cpu_freq_mhz': None,
                'platform': 'Unknown'
            }
    
    @staticmethod
    def estimate_execution_time(method_type, dataset_size, config=None):
        """
        Estimate execution time based on method and data size
        
        Returns: dict with min_time, max_time, confidence, warnings
        """
        rows, cols = dataset_size
        
        # Base time factors (in seconds)
        base_times = {
            'Sequential Chaining': {
                'base': 10,
                'row_factor': 0.001,
                'col_factor': 0.5,
                'model_factor': 15
            },
            'Voting Ensemble': {
                'base': 15,
                'row_factor': 0.002,
                'col_factor': 0.3,
                'model_factor': 20
            },
            'Bagging Ensemble': {
                'base': 20,
                'row_factor': 0.003,
                'col_factor': 0.4,
                'model_factor': 10  # Per estimator
            },
            'Stacking Ensemble': {
                'base': 30,
                'row_factor': 0.005,
                'col_factor': 0.8,
                'model_factor': 25,
                'cv_factor': 10  # Per CV fold
            },
            'Pipeline Building': {
                'base': 20,
                'row_factor': 0.004,
                'col_factor': 0.6,
                'tuning_factor': 60  # If hyperparameter tuning enabled
            }
        }
        
        if method_type not in base_times:
            return {
                'min_time': 30,
                'max_time': 300,
                'confidence': 'Low',
                'warnings': ['Unknown method - time estimate unavailable']
            }
        
        factors = base_times[method_type]
        
        # Calculate base time
        estimated_time = factors['base']
        estimated_time += rows * factors['row_factor']
        estimated_time += cols * factors['col_factor']
        
        # Method-specific adjustments
        warnings = []
        
        if method_type == 'Sequential Chaining':
            n_models = config.get('n_models', 3) if config else 3
            estimated_time += n_models * factors['model_factor']
            
        elif method_type == 'Voting Ensemble':
            n_models = config.get('n_models', 3) if config else 3
            estimated_time += n_models * factors['model_factor']
            
        elif method_type == 'Bagging Ensemble':
            n_estimators = config.get('n_estimators', 10) if config else 10
            estimated_time += n_estimators * factors['model_factor']
            if n_estimators > 50:
                warnings.append(f"High number of estimators ({n_estimators}) - may take longer")
                
        elif method_type == 'Stacking Ensemble':
            n_models = config.get('n_models', 3) if config else 3
            cv_folds = config.get('cv_folds', 5) if config else 5
            estimated_time += n_models * factors['model_factor']
            estimated_time += cv_folds * factors['cv_factor']
            if cv_folds > 5:
                warnings.append(f"High CV folds ({cv_folds}) - will increase time significantly")
                
        elif method_type == 'Pipeline Building':
            if config and config.get('hyperparameter_tuning', False):
                estimated_time += factors['tuning_factor']
                tuning_iterations = config.get('tuning_iterations', 20)
                estimated_time += tuning_iterations * 2
                warnings.append("Hyperparameter tuning enabled - expect longer execution time")
        
        # Hardware adjustments
        system_info = PerformanceEstimator.get_system_info()
        if isinstance(system_info['cpu_cores'], int) and system_info['cpu_cores'] > 4:
            estimated_time *= 0.7  # Faster with more cores
        
        # Data size warnings
        if rows > 50000:
            warnings.append(f"Large dataset ({rows:,} rows) - execution may be slow")
            estimated_time *= 1.5
            
        if rows > 100000:
            warnings.append("Very large dataset - consider sampling for faster results")
            estimated_time *= 2.0
            
        if cols > 50:
            warnings.append(f"Many features ({cols}) - consider feature selection")
            estimated_time *= 1.2
        
        # Calculate range (±30%)
        min_time = max(10, estimated_time * 0.7)
        max_time = estimated_time * 1.3
        
        # Confidence based on factors
        confidence = 'High' if len(warnings) == 0 else 'Medium' if len(warnings) <= 2 else 'Low'
        
        return {
            'min_time': int(min_time),
            'max_time': int(max_time),
            'confidence': confidence,
            'warnings': warnings,
            'estimated_time': int(estimated_time)
        }
    
    @staticmethod
    def display_performance_warning(method_type, dataset_size, config=None):
        """Display performance warning UI component"""
        estimate = PerformanceEstimator.estimate_execution_time(method_type, dataset_size, config)
        
        # Format time display
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds}s"
            elif seconds < 3600:
                return f"{seconds//60}m {seconds%60}s"
            else:
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                return f"{hours}h {minutes}m"
        
        min_time_str = format_time(estimate['min_time'])
        max_time_str = format_time(estimate['max_time'])
        
        # Color-coded warning based on estimated time
        if estimate['max_time'] < 120:  # < 2 minutes
            alert_type = "info"
            icon = "⚡"
        elif estimate['max_time'] < 600:  # < 10 minutes
            alert_type = "warning" 
            icon = "⏳"
        else:  # > 10 minutes
            alert_type = "error"
            icon = "🐌"
        
        # Display warning
        if alert_type == "info":
            st.info(f"{icon} **Estimated Time:** {min_time_str} - {max_time_str} (Confidence: {estimate['confidence']})")
        elif alert_type == "warning":
            st.warning(f"{icon} **Estimated Time:** {min_time_str} - {max_time_str} (Confidence: {estimate['confidence']})")
        else:
            st.error(f"{icon} **Long Execution Expected:** {min_time_str} - {max_time_str} (Confidence: {estimate['confidence']})")
        
        # Display specific warnings
        if estimate['warnings']:
            with st.expander("⚠️ Performance Warnings"):
                for warning in estimate['warnings']:
                    st.write(f"• {warning}")
        
        # System info
        system_info = PerformanceEstimator.get_system_info()
        with st.expander("💻 System Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**CPU Cores:** {system_info['cpu_cores']}")
                st.write(f"**Memory:** {system_info['memory_gb']} GB")
            with col2:
                st.write(f"**Platform:** {system_info['platform']}")
                if system_info['cpu_freq_mhz']:
                    st.write(f"**CPU Freq:** {system_info['cpu_freq_mhz']} MHz")
        
        return estimate
    
    @staticmethod
    def show_refresh_warning():
        """Show warning about page refresh during processing"""
        st.warning("""
        ⚠️ **Important:** 
        - Don't refresh the page during execution
        - Progress will be lost if browser is closed
        - Keep this tab active for best performance
        """)
        
        st.info("""
        💡 **Tips for Long Executions:**
        - Start with smaller datasets to test
        - Use fewer models for initial experiments  
        - Consider reducing cross-validation folds
        - Enable hyperparameter tuning only when needed
        """)

class ProgressTracker:
    """
    Track progress of long-running operations
    """
    
    def __init__(self, total_steps, operation_name="Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.operation_name = operation_name
        self.start_time = datetime.now()
        
        # Initialize progress display
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.time_text = st.empty()
        
    def update(self, step_name, additional_info=""):
        """Update progress"""
        self.current_step += 1
        progress = self.current_step / self.total_steps
        
        # Update progress bar
        self.progress_bar.progress(progress)
        
        # Update status
        self.status_text.text(f"Step {self.current_step}/{self.total_steps}: {step_name}")
        
        # Calculate ETA
        if self.current_step > 1:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            avg_time_per_step = elapsed / (self.current_step - 1)
            remaining_steps = self.total_steps - self.current_step
            eta_seconds = remaining_steps * avg_time_per_step
            
            if eta_seconds < 60:
                eta_str = f"~{int(eta_seconds)}s remaining"
            else:
                eta_str = f"~{int(eta_seconds/60)}m remaining"
                
            self.time_text.text(f"⏱️ {eta_str} {additional_info}")
    
    def finish(self, success_message="✅ Complete!"):
        """Finish progress tracking"""
        self.progress_bar.progress(1.0)
        self.status_text.text(success_message)
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        if total_time < 60:
            time_str = f"Total time: {total_time:.1f}s"
        else:
            time_str = f"Total time: {int(total_time/60)}m {int(total_time%60)}s"
            
        self.time_text.text(f"🎉 {time_str}")

class SessionStateBackup:
    """
    Provide options for backing up critical session state
    """
    
    @staticmethod
    def backup_trained_models():
        """Create backup of trained models info"""
        if 'trained_models' not in st.session_state:
            return None
            
        backup_data = {}
        for model_id, model_info in st.session_state.trained_models.items():
            # Store only serializable info (not the actual model object)
            backup_data[model_id] = {
                'algorithm': model_info.get('algorithm', 'Unknown'),
                'target': model_info.get('target', 'Unknown'),
                'features': model_info.get('features', []),
                'problem_type': model_info.get('problem_type', 'Unknown'),
                'model_id': model_info.get('model_id', model_id),
                'created_at': datetime.now().isoformat()
            }
        
        return backup_data
    
    @staticmethod
    def show_backup_options():
        """Show backup options to user"""
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔄 Session Backup")
        
        if st.sidebar.button("📥 Backup Models Info"):
            backup = SessionStateBackup.backup_trained_models()
            if backup:
                import json
                backup_json = json.dumps(backup, indent=2)
                
                st.sidebar.download_button(
                    label="💾 Download Backup",
                    data=backup_json,
                    file_name=f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
                st.sidebar.success("✅ Backup ready for download")
            else:
                st.sidebar.info("No models to backup")