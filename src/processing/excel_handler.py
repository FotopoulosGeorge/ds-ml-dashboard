# src/processing/excel_handler.py

import streamlit as st
import pandas as pd
from datetime import datetime

class ExcelHandler:
    """
    Advanced Excel file handling with sheet selection and preview
    """
    
    @staticmethod
    def handle_excel_upload(uploaded_file):
        """
        Handle Excel file with multiple sheet support
        """
        try:
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            if len(sheet_names) == 1:
                # Single sheet - load directly
                df = pd.read_excel(uploaded_file, sheet_name=0)
                return df, sheet_names[0]
            
            else:
                # Multiple sheets - let user choose
                st.info(f"📑 **Excel file contains {len(sheet_names)} sheets**")
                
                selected_sheet = st.selectbox(
                    "Select sheet to load:",
                    sheet_names,
                    key=f"sheet_selector_{uploaded_file.name}"
                )
                
                if st.button(f"📊 Load Sheet: {selected_sheet}", key=f"load_sheet_{uploaded_file.name}"):
                    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                    
                    # Show preview
                    st.subheader(f"📋 Preview: {selected_sheet}")
                    st.dataframe(df.head(), use_container_width=True)
                    st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
                    
                    return df, selected_sheet
                
                # Show preview of all sheets
                with st.expander("👁️ Preview All Sheets"):
                    for sheet in sheet_names:
                        st.write(f"**📄 {sheet}**")
                        try:
                            preview_df = pd.read_excel(uploaded_file, sheet_name=sheet, nrows=3)
                            st.dataframe(preview_df, use_container_width=True)
                            st.caption(f"Shape: {pd.read_excel(uploaded_file, sheet_name=sheet).shape}")
                        except Exception as e:
                            st.error(f"Error previewing {sheet}: {str(e)}")
                        st.markdown("---")
                
                return None, None
        
        except Exception as e:
            st.error(f"❌ Error processing Excel file: {str(e)}")
            return None, None
    
    @staticmethod
    def auto_detect_data_types(df):
        """
        Enhanced data type detection for Excel data
        """
        for col in df.columns:
            # Handle Excel date detection
            if df[col].dtype == 'object':
                # Try to convert to datetime
                try:
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
                except:
                    # Try to convert to numeric
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except:
                        pass  # Keep as object
        
        return df
    
    @staticmethod
    def get_excel_info(uploaded_file):
        """
        Get comprehensive Excel file information
        """
        try:
            excel_file = pd.ExcelFile(uploaded_file)
            
            info = {
                'filename': uploaded_file.name,
                'sheets': [],
                'total_size': 0
            }
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                sheet_info = {
                    'name': sheet_name,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'size_mb': df.memory_usage(deep=True).sum() / (1024**2),
                    'column_types': df.dtypes.value_counts().to_dict()
                }
                info['sheets'].append(sheet_info)
                info['total_size'] += sheet_info['size_mb']
            
            return info
            
        except Exception as e:
            return {'error': str(e)}