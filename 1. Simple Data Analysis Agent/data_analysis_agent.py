import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Any, Optional
import io
import base64

class DataAnalysisAgent:

    def __init__(self, name: str = 'Data Assistant'):
        self.name = name
        self.data = None
        self.file_path = None
        self.history = []
        self.log(f'{self.name} initialized and ready')
    
    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.history.append(f'[{timestamp}] {message}')

    def load_data(self, file_path: str=None, file_content = None) -> bool:
        try:
            if file_content is not None:
                if isinstance(file_content, bytes):
                    self.data = pd.read_csv(io.BytersIO(file_content))
                else:
                    self.data = file_content
                self.file_path = 'uploaded file'
            
            elif file_path is not None:
                if file_path.endswith('.csv'):
                    self.data = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx','.xls')):
                    self.data = pd.read_excel(file_path)
                else:
                    self.log(f'Unsupported file format: {file_path}')
                    return False
                self.file_path = file_path
            else:
                self.log("No data source provided")
                return False
            
            self.log(f"Successfully loaded data from {self.file_path}")
            return True
        
        except Exception as e:
            self.log(f"Error Loading data: {str(e)}")
            return False

    def describe_data(self) -> Dict[str, Any]:
        if self.data is None:
            self.log('No data loaded to describe')
            return {'Error':"No data Loaded"}
        
        summary = {
            'rows': len(self.data),
            'columns': len(self.data.columns),
            'column_names': list(self.data.columns),
            'data_types': {col: str(dtype) for col, dtype in self.data.dtypes.items()},
            'missing_values': self.data.isna().sum().to_dict(),
            'numeric_summary': self.data.describe().to_dict() if not self.data.select_dtypes(include=[np.number]).empty else {}
        }

        self.log('Generated data summary')
        return summary
    
    def filter_data(self, conditions: Dict[str, Any]) -> pd.DataFrame:
        if self.data is None:
            self.log('No data loaded to filter')
            return pd.DataFrame()
        
        filtered_data = self.data.copy()
        for col, val in conditions.items():
            if col in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[col] == val]
        
        self.log(f'filtered data using conditions: {conditions}')
        return filtered_data
    
    def calculate_statistics(self, columns: List[str] = None)->Dict[str, Dict[str, float]]:
        if self.data is None:
            self.log('No data Loaded for statistics')
            return{'error': 'No data loaded'}
        
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        stats = {}
        for col in columns:
            if col in self.data.columns:
                if pd.api.is_numeric_dtypes(self.data[col]):
                    stats[col] = {
                        'mean':self.data[col].mean(),
                        'median':self.data[col].median(),
                        'std':self.data[col].std(),
                        'min':self.data[col].min(),
                        'max':self.data[col].max(),
                    }
                else:
                    stats[col] = {
                        'unique_values': self.data[col].nunique(),
                        'most_common': self.data[col].value_counts().index[0] if not self.data[col].value_counts().empty else None
                    }

        self.log(f"Calculated statistics for columns: {columns}")
        return stats
    
    def generate_correlation_matrix(self) -> pd.DataFrame:
        """Generate correlation matrix for numeric columns"""
        if self.data is None:
            self.log("No data loaded for correlation analysis")
            return pd.DataFrame()
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        corr_matrix = self.data[numeric_cols].corr()
        self.log("Generated correlation matrix")
        return corr_matrix
    def generate_plots(self):
        """Generate common plots for data visualization"""
        if self.data is None:
            self.log("No data loaded for plotting")
            return []

        plots = []

        # Create histograms for numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:5]  # Limit to first 5
        if not numeric_cols.empty:
            fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 3 * len(numeric_cols)))
            if len(numeric_cols) == 1:
                axes = [axes]  # Make it iterable if only one subplot

            for i, col in enumerate(numeric_cols):
                axes[i].hist(self.data[col].dropna(), bins=20, edgecolor='black')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')

            plt.tight_layout()
            plots.append(("histograms", fig))

        # Create correlation heatmap
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = self.data[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Matrix')
            plots.append(("correlation", fig))

        # Create bar chart for categorical columns (top 5 values)
        cat_cols = self.data.select_dtypes(exclude=[np.number]).columns[:3]  # Limit to first 3
        for col in cat_cols:
            if self.data[col].nunique() < 15:  # Only for columns with reasonable number of categories
                fig, ax = plt.subplots(figsize=(10, 6))
                value_counts = self.data[col].value_counts().nlargest(10)
                value_counts.plot(kind='bar', ax=ax)
                ax.set_title(f'Top Values in {col}')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plots.append((f"barchart_{col}", fig))

        self.log(f"Generated {len(plots)} plots for data visualization")
        return plots
    
    def generate_report_html(self) -> str:
        """Generate an HTML report for streamlit"""
        if self.data is None:
            self.log("No data loaded for report generation")
            return "<h2>Error: No data loaded</h2>"

        summary = self.describe_data()

        html = f"""
        <h1>Data Analysis Report</h1>
        <p>Generated by: {self.name}</p>
        <p>Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Source: {self.file_path}</p>

        <h2>Dataset Overview</h2>
        <p>Rows: {summary['rows']}</p>
        <p>Columns: {summary['columns']}</p>

        <h2>Column Information</h2>
        <table border="1">
            <tr>
                <th>Column Name</th>
                <th>Data Type</th>
                <th>Missing Values</th>
            </tr>
        """
        