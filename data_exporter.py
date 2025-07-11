# data_exporter.py - Independent export module
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io

class DataExporter:
    """
    Independent export class that handles data downloads from session state
    """
    
    def __init__(self):
        pass
    
    def get_current_data(self):
        """Get current working data from session state"""
        if 'working_df' in st.session_state and st.session_state.working_df is not None:
            return st.session_state.working_df.copy()
        elif 'base_df' in st.session_state and st.session_state.base_df is not None:
            return st.session_state.base_df.copy()
        else:
            st.error("No data available for export")
            return pd.DataFrame()
    
    def get_original_data(self):
        """Get original data from session state"""
        if 'base_df' in st.session_state and st.session_state.base_df is not None:
            return st.session_state.base_df.copy()
        else:
            return pd.DataFrame()
    
    def render_export_tab(self):
        """
        Render the complete export tab
        """
        st.header("💾 **Data Export**")
        
        # Get current data
        current_data = self.get_current_data()
        original_data = self.get_original_data()
        
        if current_data.empty:
            st.warning("⚠️ No data available for export")
            return
        
        # Export summary
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            st.metric("Current Data", f"{len(current_data):,} rows")
        with export_col2:
            st.metric("File Size", f"{self._estimate_file_size(current_data)}")
        with export_col3:
            filters_applied = len(current_data) != len(original_data) if not original_data.empty else False
            st.metric("Status", "Filtered" if filters_applied else "Original")
        
        st.markdown("---")
        
        # Export options
        export_type = st.selectbox(
            "**What would you like to export?**",
            [
                "📄 Current Data (CSV)",
                "📊 Summary Statistics",
                "📋 Data Report",
                "🔍 Filter Information",
                "📈 All Analytics"
            ],
            key="export_type"
        )
        
        st.markdown("---")
        
        try:
            if export_type == "📄 Current Data (CSV)":
                self._export_data_csv(current_data)
            elif export_type == "📊 Summary Statistics":
                self._export_statistics(current_data)
            elif export_type == "📋 Data Report":
                self._export_data_report(current_data, original_data)
            elif export_type == "🔍 Filter Information":
                self._export_filter_info(current_data, original_data)
            elif export_type == "📈 All Analytics":
                self._export_all_analytics(current_data, original_data)
                
        except Exception as e:
            st.error(f"❌ Error during export: {str(e)}")
    
    def _estimate_file_size(self, df):
        """Estimate the file size of the dataframe"""
        memory_usage = df.memory_usage(deep=True).sum()
        
        if memory_usage < 1024:
            return f"{memory_usage} B"
        elif memory_usage < 1024**2:
            return f"{memory_usage/1024:.1f} KB"
        elif memory_usage < 1024**3:
            return f"{memory_usage/(1024**2):.1f} MB"
        else:
            return f"{memory_usage/(1024**3):.1f} GB"
    
    def _export_data_csv(self, df):
        """Export main data as CSV"""
        st.subheader("📄 Export Current Data")
        
        # Preview options
        preview_col1, preview_col2 = st.columns(2)
        
        with preview_col1:
            include_index = st.checkbox("Include row numbers", value=False, key="csv_index")
        with preview_col2:
            encoding = st.selectbox("Encoding", ["utf-8", "utf-16", "latin1"], key="csv_encoding")
        
        # Generate CSV
        csv_data = df.to_csv(index=include_index, encoding=encoding)
        
        # Download button
        st.download_button(
            label=f"📥 **Download Data ({len(df):,} rows)**",
            data=csv_data,
            file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Preview
        with st.expander("👁️ Preview Export Data"):
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Showing first 10 of {len(df):,} rows")
    
    def _export_statistics(self, df):
        """Export summary statistics"""
        st.subheader("📊 Export Summary Statistics")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numeric columns available for statistics")
            return
        
        # Generate statistics
        stats_df = df[numeric_cols].describe()
        
        # Additional statistics
        additional_stats = pd.DataFrame({
            'Column': numeric_cols,
            'Skewness': [df[col].skew() for col in numeric_cols],
            'Kurtosis': [df[col].kurtosis() for col in numeric_cols],
            'Missing_Count': [df[col].isnull().sum() for col in numeric_cols]
        })
        
        stats_csv = stats_df.to_csv()
        additional_csv = additional_stats.to_csv(index=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📊 **Download Basic Statistics**",
                data=stats_csv,
                file_name=f"statistics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label="📈 **Download Advanced Statistics**",
                data=additional_csv,
                file_name=f"advanced_stats_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Preview
        with st.expander("👁️ Preview Statistics"):
            st.dataframe(stats_df, use_container_width=True)
    
    def _export_data_report(self, current_df, original_df):
        """Export comprehensive data report"""
        st.subheader("📋 Export Data Report")
        
        # Generate report
        report_data = {
            'Report_Generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Original_Rows': len(original_df) if not original_df.empty else 0,
            'Current_Rows': len(current_df),
            'Total_Columns': len(current_df.columns),
            'Numeric_Columns': len(current_df.select_dtypes(include=[np.number]).columns),
            'Text_Columns': len(current_df.select_dtypes(include=['object', 'category']).columns),
            'Date_Columns': len(current_df.select_dtypes(include=['datetime64']).columns),
            'Missing_Values': current_df.isnull().sum().sum(),
            'Duplicate_Rows': current_df.duplicated().sum(),
            'Memory_Usage_MB': round(current_df.memory_usage(deep=True).sum() / (1024**2), 2)
        }
        
        # Column details
        column_details = pd.DataFrame({
            'Column_Name': current_df.columns,
            'Data_Type': current_df.dtypes.astype(str),
            'Non_Null_Count': current_df.count(),
            'Null_Count': current_df.isnull().sum(),
            'Unique_Values': [current_df[col].nunique() for col in current_df.columns],
            'Sample_Value': [str(current_df[col].iloc[0]) if len(current_df) > 0 else 'N/A' for col in current_df.columns]
        })
        
        # Create report dataframes
        report_summary = pd.DataFrame([report_data])
        
        # Combine into one CSV
        output = io.StringIO()
        
        output.write("=== DATA REPORT SUMMARY ===\n")
        report_summary.to_csv(output, index=False)
        output.write("\n=== COLUMN DETAILS ===\n")
        column_details.to_csv(output, index=False)
        
        report_csv = output.getvalue()
        output.close()
        
        st.download_button(
            label="📋 **Download Complete Report**",
            data=report_csv,
            file_name=f"data_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Preview
        with st.expander("👁️ Preview Report Summary"):
            st.dataframe(report_summary, use_container_width=True)
        
        with st.expander("👁️ Preview Column Details"):
            st.dataframe(column_details, use_container_width=True)
    
    def _export_filter_info(self, current_df, original_df):
        """Export filter information"""
        st.subheader("🔍 Export Filter Information")
        
        # Get filter info from session state
        filter_info = st.session_state.get('active_filters', [])
        
        filter_data = {
            'Export_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Original_Rows': len(original_df) if not original_df.empty else 0,
            'Filtered_Rows': len(current_df),
            'Rows_Removed': (len(original_df) - len(current_df)) if not original_df.empty else 0,
            'Filters_Applied': len(filter_info),
            'Filter_Status': 'Active' if len(filter_info) > 0 else 'None'
        }
        
        if len(original_df) > 0:
            filter_data['Reduction_Percentage'] = round(((len(original_df) - len(current_df)) / len(original_df)) * 100, 2)
        else:
            filter_data['Reduction_Percentage'] = 0
        
        filter_df = pd.DataFrame([filter_data])
        filter_csv = filter_df.to_csv(index=False)
        
        st.download_button(
            label="🔍 **Download Filter Info**",
            data=filter_csv,
            file_name=f"filter_info_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Preview
        with st.expander("👁️ Preview Filter Info"):
            st.dataframe(filter_df, use_container_width=True)
    
    def _export_all_analytics(self, current_df, original_df):
        """Export all analytics combined"""
        st.subheader("📈 Export All Analytics")
        
        st.info("This will create a ZIP file with all analytics exports")
        
        # For now, provide individual downloads
        col1, col2 = st.columns(2)
        
        with col1:
            # Data CSV
            csv_data = current_df.to_csv(index=False)
            st.download_button(
                label="📄 Data CSV",
                data=csv_data,
                file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Statistics
            numeric_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                stats_csv = current_df[numeric_cols].describe().to_csv()
                st.download_button(
                    label="📊 Statistics",
                    data=stats_csv,
                    file_name=f"stats_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            # Report
            report_data = {
                'Generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Rows': len(current_df),
                'Columns': len(current_df.columns),
                'Missing': current_df.isnull().sum().sum()
            }
            report_csv = pd.DataFrame([report_data]).to_csv(index=False)
            
            st.download_button(
                label="📋 Report",
                data=report_csv,
                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.caption("💡 Tip: Download each file individually for now")