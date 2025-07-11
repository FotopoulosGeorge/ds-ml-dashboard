# feature_engineering.py
import pandas as pd
import numpy as np
from datetime import datetime
from functools import cached_property

class FeatureEngineer:
    """Handles all feature engineering transformations"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_columns = list(df.columns)
        self.new_features = []
        self._column_cache = {}
    
    # Numerical transformations
    def apply_numerical_transform(self, column, transform_type, new_col_name=None, **kwargs):
        """
        Apply numerical transformations to a specified column
        
        Parameters:
        -----------
        column : str
            Name of the column to transform
        transform_type : str
            Type of transformation ('Log Transform', 'Square Root', etc.)
        new_col_name : str, optional
            Name for the new column. If None, auto-generates name
        **kwargs : dict
            Additional parameters (e.g., n_bins for quantile binning)
        
        Returns:
        --------
        str : Name of the created column
        
        Raises:
        -------
        ValueError : If column doesn't exist or isn't numeric
        """
    
        # Validation
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError(f"Column '{column}' is not numeric")
        
        # Auto-generate column name if not provided
        if new_col_name is None:
            new_col_name = f"{column}_{transform_type.lower().replace(' ', '_')}"
        
        # Get the data
        data = self.df[column].copy()
        
        try:
            if transform_type == "Log Transform":
                # Handle negative values by adding constant
                min_val = data.min()
                if min_val <= 0:
                    # Add constant to make all values positive
                    transformed = np.log(data + abs(min_val) + 1)
                else:
                    transformed = np.log(data)
            
            elif transform_type == "Square Root":
                # Take absolute value to handle negative numbers
                transformed = np.sqrt(data.abs())
            
            elif transform_type == "Square":
                transformed = data ** 2
            
            elif transform_type == "Reciprocal":
                # Avoid division by zero by replacing 0 with NaN
                transformed = 1 / data.replace(0, np.nan)
            
            elif transform_type == "StandardScaler":
                # Z-score normalization: (x - mean) / std
                mean_val = data.mean()
                std_val = data.std()
                if std_val == 0:
                    raise ValueError("Cannot standardize: standard deviation is zero")
                transformed = (data - mean_val) / std_val
            
            elif transform_type == "MinMaxScaler":
                # Scale to [0, 1]: (x - min) / (max - min)
                min_val = data.min()
                max_val = data.max()
                if min_val == max_val:
                    raise ValueError("Cannot scale: all values are identical")
                transformed = (data - min_val) / (max_val - min_val)
            
            elif transform_type == "RobustScaler":
                # Scale using median and IQR: (x - median) / IQR
                median_val = data.median()
                q75 = data.quantile(0.75)
                q25 = data.quantile(0.25)
                iqr = q75 - q25
                if iqr == 0:
                    raise ValueError("Cannot robust scale: IQR is zero")
                transformed = (data - median_val) / iqr
            
            elif transform_type == "Quantile Binning":
                # Convert to discrete bins based on quantiles
                n_bins = kwargs.get('n_bins', 5)
                transformed = pd.qcut(data, q=n_bins, labels=False, duplicates='drop')
            
            else:
                raise ValueError(f"Unknown transformation type: {transform_type}")
            
            # Add the new column to dataframe
            self.df[new_col_name] = transformed
            
            # Track the new feature
            if new_col_name not in self.new_features:
                self.new_features.append(new_col_name)
            
            return new_col_name
        
        except Exception as e:
            raise ValueError(f"Transformation failed: {str(e)}")


    # Helper method to get available transformations
    def get_available_numerical_transforms(self):
        """Returns list of available numerical transformations"""
        return [
            "Log Transform",
            "Square Root", 
            "Square",
            "Reciprocal",
            "StandardScaler",
            "MinMaxScaler", 
            "RobustScaler",
            "Quantile Binning"
        ]


    # Helper method to get transformation info
    def get_transform_info(self, transform_type):
        """Returns description and requirements for each transformation"""
        info = {
            "Log Transform": {
                "description": "Natural logarithm transformation. Handles negative values automatically.",
                "use_case": "Reduce skewness, compress large ranges",
                "requirements": "Numeric data"
            },
            "Square Root": {
                "description": "Square root transformation. Uses absolute values for negative numbers.",
                "use_case": "Reduce right skewness, moderate compression", 
                "requirements": "Numeric data"
            },
            "StandardScaler": {
                "description": "Z-score normalization (mean=0, std=1)",
                "use_case": "When features have different scales",
                "requirements": "Numeric data, std > 0"
            },
            "MinMaxScaler": {
                "description": "Scale to range [0,1]",
                "use_case": "When you need bounded values",
                "requirements": "Numeric data, min ≠ max"
            },
            "Quantile Binning": {
                "description": "Convert to discrete bins based on quantiles",
                "use_case": "Create categorical features from continuous",
                "requirements": "Numeric data",
                "parameters": "n_bins (default: 5)"
            }
        }
        return info.get(transform_type, {"description": "No info available"})
       
        

    
    # Text feature extraction  
    def extract_text_features(self, column, features_list):
        """Extract text-based features"""
        # Implementation here
        pass
    
    # Date/time features
    def extract_date_features(self, column, features_list):
        """Extract temporal features"""
        # Implementation here
        pass
    
    # Categorical encoding
    def encode_categorical(self, column, encoding_type, target_col=None):
        """Various categorical encoding methods"""
        # Implementation here
        pass
    
    # Advanced features
    def create_interaction_features(self, col1, col2):
        """Create interaction terms"""
        # Implementation here
        pass
    
    def create_ratio_features(self, numerator, denominator):
        """Create ratio features"""
        # Implementation here
        pass
    
    # Statistical features
    def create_statistical_features(self, group_col, agg_col, functions):
        """Create group-wise statistical features"""
        # Implementation here
        pass
    
    def _update_column_cache(self):
        """Update column type cache when dataframe changes"""
        self._column_cache = {
            'numeric': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical': self.df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime': self.df.select_dtypes(include=['datetime64']).columns.tolist(),
            'all': self.df.columns.tolist()
        }

    def get_column_info(self):
        """Get all column types at once - computed only when needed"""
        if not self._column_cache or len(self._column_cache.get('all', [])) != len(self.df.columns):
            self._update_column_cache()
        return self._column_cache

    # Utility methods
    def get_feature_summary(self):
        """Return summary of created features"""
        return {
            'original_features': len(self.original_columns),
            'new_features': len(self.new_features), 
            'total_features': len(self.df.columns),
            'feature_list': self.new_features
        }
    
    def reset_to_original(self, original_df):
        """Reset to original dataset"""
        self.df = original_df.copy()
        self.new_features = []
        return self.df