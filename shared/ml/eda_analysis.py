"""
Exploratory Data Analysis (EDA) Module
=====================================

Comprehensive EDA for India Environmental Monitoring System.
Generates plots, reports, and insights for all 8+ datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime
import structlog

logger = structlog.get_logger()


class EDAGenerator:
    """Generate comprehensive EDA reports and visualizations."""
    
    def __init__(self, output_path: str = "ml/eda"):
        self.output_path = Path(output_path)
        self.plots_path = self.output_path / "plots"
        self.reports_path = self.output_path / "reports"
        
        # Create directories
        self.output_path.mkdir(exist_ok=True)
        self.plots_path.mkdir(exist_ok=True)
        self.reports_path.mkdir(exist_ok=True)
        
        # Configure plotting
        plt.style.use('default')
        sns.set_palette("husl")
        
        logger.info("EDA Generator initialized", output_path=str(self.output_path))
    
    def generate_complete_eda(self, datasets: Dict[str, pd.DataFrame], 
                            integrated_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate complete EDA for all datasets."""
        logger.info("Generating complete EDA...")
        
        eda_results = {
            'timestamp': datetime.now().isoformat(),
            'datasets_analyzed': list(datasets.keys()),
            'plots_generated': [],
            'insights': {}
        }
        
        # 1. Individual dataset analysis
        for name, df in datasets.items():
            logger.info(f"Analyzing dataset: {name}")
            analysis = self.analyze_individual_dataset(name, df)
            eda_results['insights'][name] = analysis
        
        # 2. Integrated dataset analysis
        logger.info("Analyzing integrated dataset")
        integrated_analysis = self.analyze_integrated_dataset(integrated_data)
        eda_results['insights']['integrated'] = integrated_analysis
        
        # 3. Cross-dataset correlations
        logger.info("Analyzing cross-dataset correlations")
        correlation_analysis = self.analyze_cross_dataset_correlations(datasets, integrated_data)
        eda_results['insights']['correlations'] = correlation_analysis
        
        # 4. Generate summary plots
        summary_plots = self.generate_summary_visualizations(datasets, integrated_data)
        eda_results['plots_generated'].extend(summary_plots)
        
        # 5. Generate EDA report
        self.generate_eda_report(eda_results)
        
        logger.info("Complete EDA generation finished", 
                   plots=len(eda_results['plots_generated']),
                   insights=len(eda_results['insights']))
        
        return eda_results
    
    def analyze_individual_dataset(self, name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze individual dataset and generate insights."""
        analysis = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_data': {},
            'data_types': {},
            'summary_stats': {},
            'unique_counts': {},
            'insights': []
        }
        
        # Missing data analysis
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        analysis['missing_data'] = {
            'counts': missing_counts[missing_counts > 0].to_dict(),
            'percentages': missing_percentages[missing_percentages > 0].to_dict()
        }
        
        # Data types
        analysis['data_types'] = df.dtypes.astype(str).to_dict()
        
        # Summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis['summary_stats'] = df[numeric_cols].describe().to_dict()
        
        # Unique value counts for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
            if df[col].nunique() < 50:  # Only for columns with reasonable number of unique values
                analysis['unique_counts'][col] = df[col].value_counts().head(10).to_dict()
        
        # Generate insights
        analysis['insights'] = self.generate_dataset_insights(name, df, analysis)
        
        # Generate individual plots
        plot_files = self.generate_individual_plots(name, df)
        analysis['plot_files'] = plot_files
        
        return analysis
    
    def generate_dataset_insights(self, name: str, df: pd.DataFrame, analysis: Dict[str, Any]) -> List[str]:
        """Generate human-readable insights for a dataset."""
        insights = []
        
        # Data quality insights
        missing_pct = sum(analysis['missing_data']['percentages'].values()) / len(df.columns)
        if missing_pct > 10:
            insights.append(f"High missing data: {missing_pct:.1f}% of values are missing")
        elif missing_pct < 1:
            insights.append(f"Excellent data quality: Only {missing_pct:.1f}% missing values")
        
        # Size insights
        if df.shape[0] > 10000:
            insights.append(f"Large dataset with {df.shape[0]:,} records")
        elif df.shape[0] < 100:
            insights.append(f"Small dataset with only {df.shape[0]} records")
        
        # Feature insights
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object']).columns)
        insights.append(f"Feature mix: {numeric_cols} numeric, {categorical_cols} categorical columns")
        
        # Dataset-specific insights
        if name == 'factories':
            if 'emission_level' in df.columns:
                high_emission = (df['emission_level'] == 'High').sum()
                total = len(df)
                insights.append(f"High-emission factories: {high_emission}/{total} ({high_emission/total*100:.1f}%)")
        
        elif name == 'vehicles':
            if 'vehicle_density' in df.columns:
                avg_density = df['vehicle_density'].mean()
                insights.append(f"Average vehicle density: {avg_density:.1f} vehicles per 1000 people")
        
        elif name == 'forest':
            if 'forest_cover_percent' in df.columns:
                avg_cover = df['forest_cover_percent'].mean()
                insights.append(f"Average forest cover: {avg_cover:.1f}%")
        
        elif name == 'population':
            if 'urbanization_rate' in df.columns:
                avg_urbanization = df['urbanization_rate'].mean()
                insights.append(f"Average urbanization rate: {avg_urbanization:.1f}%")
        
        return insights
    
    def generate_individual_plots(self, name: str, df: pd.DataFrame) -> List[str]:
        """Generate plots for individual dataset."""
        plot_files = []
        
        # 1. Missing data heatmap
        if df.isnull().sum().sum() > 0:
            plt.figure(figsize=(12, 8))
            sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
            plt.title(f'Missing Data Pattern - {name.title()}')
            plt.tight_layout()
            
            plot_file = self.plots_path / f'{name}_missing_data.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(str(plot_file))
        
        # 2. Numeric distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            n_cols = min(4, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols[:16]):  # Limit to 16 plots
                if i < len(axes):
                    df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(f'Numeric Distributions - {name.title()}')
            plt.tight_layout()
            
            plot_file = self.plots_path / f'{name}_distributions.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(str(plot_file))
        
        # 3. Categorical value counts
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols[:4]:  # Limit to 4 categorical columns
                if df[col].nunique() < 20:  # Only plot if reasonable number of categories
                    plt.figure(figsize=(12, 6))
                    value_counts = df[col].value_counts().head(15)
                    value_counts.plot(kind='bar')
                    plt.title(f'Value Counts - {col} ({name.title()})')
                    plt.xlabel(col)
                    plt.ylabel('Count')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    plot_file = self.plots_path / f'{name}_{col}_counts.png'
                    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_files.append(str(plot_file))
        
        return plot_files
    
    def analyze_integrated_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the integrated dataset for ML readiness."""
        analysis = {
            'shape': df.shape,
            'feature_summary': {},
            'correlation_analysis': {},
            'ml_readiness': {},
            'feature_importance_proxy': {},
            'insights': []
        }
        
        # Feature type summary
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        analysis['feature_summary'] = {
            'total_features': len(df.columns),
            'numeric_features': len(numeric_features),
            'categorical_features': len(categorical_features),
            'numeric_feature_names': numeric_features,
            'categorical_feature_names': categorical_features
        }
        
        # Correlation analysis for numeric features
        if len(numeric_features) > 1:
            correlation_matrix = df[numeric_features].corr()
            
            # Find highly correlated pairs
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # High correlation threshold
                        high_correlations.append({
                            'feature_1': correlation_matrix.columns[i],
                            'feature_2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            analysis['correlation_analysis'] = {
                'high_correlations': high_correlations,
                'correlation_matrix_shape': correlation_matrix.shape
            }
            
            # Generate correlation heatmap
            self.generate_correlation_heatmap(correlation_matrix, 'integrated_correlation')
        
        # ML readiness assessment
        missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        
        analysis['ml_readiness'] = {
            'missing_data_percentage': missing_percentage,
            'data_quality_score': max(0, 100 - missing_percentage),
            'feature_count': len(df.columns),
            'sample_count': len(df),
            'recommendations': []
        }
        
        # Add ML readiness recommendations
        if missing_percentage > 10:
            analysis['ml_readiness']['recommendations'].append("Address missing data before ML modeling")
        if len(df) < 1000:
            analysis['ml_readiness']['recommendations'].append("Consider data augmentation - low sample count")
        if len(numeric_features) < 5:
            analysis['ml_readiness']['recommendations'].append("Consider feature engineering to create more numeric features")
        
        # Feature importance proxy (based on variance for numeric features)
        if len(numeric_features) > 0:
            feature_variance = df[numeric_features].var().sort_values(ascending=False)
            analysis['feature_importance_proxy'] = feature_variance.head(10).to_dict()
        
        # Generate insights
        analysis['insights'] = self.generate_integrated_insights(df, analysis)
        
        return analysis
    
    def generate_integrated_insights(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> List[str]:
        """Generate insights for the integrated dataset."""
        insights = []
        
        # Data quality insights
        missing_pct = analysis['ml_readiness']['missing_data_percentage']
        if missing_pct < 5:
            insights.append(f"Excellent data quality: Only {missing_pct:.1f}% missing values")
        elif missing_pct > 20:
            insights.append(f"Data quality concern: {missing_pct:.1f}% missing values")
        
        # Feature insights
        feature_summary = analysis['feature_summary']
        insights.append(f"Feature composition: {feature_summary['numeric_features']} numeric, {feature_summary['categorical_features']} categorical")
        
        # Correlation insights
        if 'correlation_analysis' in analysis:
            high_corr_count = len(analysis['correlation_analysis']['high_correlations'])
            if high_corr_count > 0:
                insights.append(f"Found {high_corr_count} highly correlated feature pairs (|r| > 0.7)")
        
        # Sample size insights
        if len(df) > 10000:
            insights.append(f"Large sample size: {len(df):,} records suitable for complex models")
        elif len(df) < 1000:
            insights.append(f"Small sample size: {len(df)} records - consider simpler models")
        
        # Feature diversity insights
        if 'feature_importance_proxy' in analysis:
            top_varying_feature = list(analysis['feature_importance_proxy'].keys())[0]
            insights.append(f"Most varying feature: {top_varying_feature}")
        
        return insights
    
    def analyze_cross_dataset_correlations(self, datasets: Dict[str, pd.DataFrame], 
                                         integrated_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between different datasets."""
        analysis = {
            'dataset_relationships': {},
            'geographic_analysis': {},
            'environmental_factors': {},
            'insights': []
        }
        
        # Geographic clustering analysis
        if 'latitude' in integrated_data.columns and 'longitude' in integrated_data.columns:
            geo_analysis = self.analyze_geographic_patterns(integrated_data)
            analysis['geographic_analysis'] = geo_analysis
        
        # Environmental factor correlations
        env_factors = []
        for col in integrated_data.columns:
            if any(keyword in col.lower() for keyword in ['emission', 'forest', 'vehicle', 'pollution']):
                env_factors.append(col)
        
        if len(env_factors) > 1:
            env_corr = integrated_data[env_factors].corr()
            analysis['environmental_factors'] = {
                'factors_analyzed': env_factors,
                'strong_correlations': self.find_strong_correlations(env_corr)
            }
        
        # Generate cross-dataset insights
        analysis['insights'] = self.generate_cross_dataset_insights(datasets, analysis)
        
        return analysis
    
    def analyze_geographic_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze geographic distribution patterns."""
        geo_analysis = {}
        
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Geographic bounds
            geo_analysis['bounds'] = {
                'lat_min': df['latitude'].min(),
                'lat_max': df['latitude'].max(),
                'lon_min': df['longitude'].min(),
                'lon_max': df['longitude'].max(),
                'coverage_area_km2': self.calculate_coverage_area(df)
            }
            
            # Geographic clustering
            from sklearn.cluster import KMeans
            coords = df[['latitude', 'longitude']].dropna()
            if len(coords) > 10:
                kmeans = KMeans(n_clusters=min(5, len(coords)//2), random_state=42)
                clusters = kmeans.fit_predict(coords)
                geo_analysis['clusters'] = {
                    'n_clusters': len(np.unique(clusters)),
                    'cluster_centers': kmeans.cluster_centers_.tolist()
                }
        
        return geo_analysis
    
    def calculate_coverage_area(self, df: pd.DataFrame) -> float:
        """Calculate approximate coverage area in km²."""
        if 'latitude' in df.columns and 'longitude' in df.columns:
            lat_range = df['latitude'].max() - df['latitude'].min()
            lon_range = df['longitude'].max() - df['longitude'].min()
            
            # Rough approximation (1 degree ≈ 111 km)
            area_km2 = lat_range * lon_range * 111 * 111
            return area_km2
        return 0
    
    def find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.5) -> List[Dict]:
        """Find strong correlations in correlation matrix."""
        strong_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    strong_correlations.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                    })
        
        return sorted(strong_correlations, key=lambda x: abs(x['correlation']), reverse=True)
    
    def generate_cross_dataset_insights(self, datasets: Dict[str, pd.DataFrame], 
                                      analysis: Dict[str, Any]) -> List[str]:
        """Generate insights from cross-dataset analysis."""
        insights = []
        
        # Dataset size comparisons
        dataset_sizes = {name: len(df) for name, df in datasets.items()}
        largest_dataset = max(dataset_sizes, key=dataset_sizes.get)
        smallest_dataset = min(dataset_sizes, key=dataset_sizes.get)
        
        insights.append(f"Largest dataset: {largest_dataset} ({dataset_sizes[largest_dataset]:,} records)")
        insights.append(f"Smallest dataset: {smallest_dataset} ({dataset_sizes[smallest_dataset]:,} records)")
        
        # Geographic coverage
        if 'geographic_analysis' in analysis and 'bounds' in analysis['geographic_analysis']:
            coverage = analysis['geographic_analysis']['bounds']['coverage_area_km2']
            insights.append(f"Geographic coverage: ~{coverage:,.0f} km²")
        
        # Environmental correlations
        if 'environmental_factors' in analysis:
            strong_env_corr = len(analysis['environmental_factors']['strong_correlations'])
            if strong_env_corr > 0:
                insights.append(f"Found {strong_env_corr} strong environmental factor correlations")
        
        return insights
    
    def generate_correlation_heatmap(self, corr_matrix: pd.DataFrame, name: str):
        """Generate correlation heatmap plot."""
        plt.figure(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        
        plt.title(f'Correlation Matrix - {name.title()}')
        plt.tight_layout()
        
        plot_file = self.plots_path / f'{name}_heatmap.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)
    
    def generate_summary_visualizations(self, datasets: Dict[str, pd.DataFrame], 
                                      integrated_data: pd.DataFrame) -> List[str]:
        """Generate high-level summary visualizations."""
        plot_files = []
        
        # 1. Dataset size comparison
        plt.figure(figsize=(12, 6))
        dataset_sizes = {name: len(df) for name, df in datasets.items()}
        
        plt.subplot(1, 2, 1)
        bars = plt.bar(dataset_sizes.keys(), dataset_sizes.values())
        plt.title('Dataset Sizes (Record Count)')
        plt.xlabel('Dataset')
        plt.ylabel('Number of Records')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom')
        
        # 2. Missing data summary
        plt.subplot(1, 2, 2)
        missing_percentages = {}
        for name, df in datasets.items():
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            missing_percentages[name] = missing_pct
        
        bars = plt.bar(missing_percentages.keys(), missing_percentages.values())
        plt.title('Missing Data Percentage by Dataset')
        plt.xlabel('Dataset')
        plt.ylabel('Missing Data %')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_file = self.plots_path / 'dataset_summary.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(plot_file))
        
        # 3. Feature type distribution in integrated dataset
        numeric_count = len(integrated_data.select_dtypes(include=[np.number]).columns)
        categorical_count = len(integrated_data.select_dtypes(include=['object']).columns)
        
        plt.figure(figsize=(8, 6))
        plt.pie([numeric_count, categorical_count], 
               labels=['Numeric Features', 'Categorical Features'],
               autopct='%1.1f%%', startangle=90)
        plt.title('Feature Type Distribution - Integrated Dataset')
        
        plot_file = self.plots_path / 'feature_type_distribution.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(plot_file))
        
        # 4. Geographic distribution (if coordinates available)
        if 'latitude' in integrated_data.columns and 'longitude' in integrated_data.columns:
            plt.figure(figsize=(12, 8))
            
            coords = integrated_data[['latitude', 'longitude']].dropna()
            if len(coords) > 0:
                plt.scatter(coords['longitude'], coords['latitude'], alpha=0.6, s=10)
                plt.title('Geographic Distribution of Data Points')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.grid(True, alpha=0.3)
                
                plot_file = self.plots_path / 'geographic_distribution.png'
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(str(plot_file))
        
        return plot_files
    
    def generate_eda_report(self, eda_results: Dict[str, Any]):
        """Generate comprehensive EDA report in Markdown format."""
        report_content = self.create_eda_markdown_report(eda_results)
        
        report_file = self.reports_path / 'eda_report.md'
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        # Also save as JSON for programmatic access
        json_file = self.reports_path / 'eda_results.json'
        with open(json_file, 'w') as f:
            json.dump(eda_results, f, indent=2, default=str)
        
        logger.info("EDA report generated", 
                   markdown_report=str(report_file),
                   json_report=str(json_file))
    
    def create_eda_markdown_report(self, eda_results: Dict[str, Any]) -> str:
        """Create comprehensive EDA report in Markdown format."""
        
        report = f"""# Exploratory Data Analysis Report
## India Environmental Monitoring System

**Generated:** {eda_results['timestamp']}  
**Datasets Analyzed:** {len(eda_results['datasets_analyzed'])}  
**Plots Generated:** {len(eda_results['plots_generated'])}

---

## Executive Summary

This comprehensive EDA analyzes {len(eda_results['datasets_analyzed'])} datasets for the India Environmental Monitoring System, covering industrial emissions, vehicle data, population statistics, forest cover, and more.

### Key Findings:

"""
        
        # Add integrated dataset insights
        if 'integrated' in eda_results['insights']:
            integrated_insights = eda_results['insights']['integrated']['insights']
            for insight in integrated_insights[:5]:  # Top 5 insights
                report += f"- {insight}\n"
        
        report += "\n---\n\n## Dataset Analysis\n\n"
        
        # Individual dataset summaries
        for dataset_name in eda_results['datasets_analyzed']:
            if dataset_name in eda_results['insights']:
                dataset_analysis = eda_results['insights'][dataset_name]
                
                report += f"### {dataset_name.title()} Dataset\n\n"
                report += f"**Shape:** {dataset_analysis['shape'][0]:,} records × {dataset_analysis['shape'][1]} features  \n"
                report += f"**Memory Usage:** {dataset_analysis['memory_usage_mb']:.1f} MB  \n"
                
                # Missing data summary
                if dataset_analysis['missing_data']['percentages']:
                    missing_fields = len(dataset_analysis['missing_data']['percentages'])
                    total_missing = sum(dataset_analysis['missing_data']['percentages'].values())
                    report += f"**Missing Data:** {missing_fields} fields with missing values ({total_missing:.1f}% total)  \n"
                else:
                    report += f"**Missing Data:** None - Complete dataset  \n"
                
                # Key insights
                if dataset_analysis['insights']:
                    report += "\n**Key Insights:**\n"
                    for insight in dataset_analysis['insights']:
                        report += f"- {insight}\n"
                
                report += "\n"
        
        # Correlation analysis
        if 'correlations' in eda_results['insights']:
            corr_analysis = eda_results['insights']['correlations']
            
            report += "## Cross-Dataset Correlation Analysis\n\n"
            
            if 'environmental_factors' in corr_analysis:
                env_factors = corr_analysis['environmental_factors']
                report += f"**Environmental Factors Analyzed:** {len(env_factors['factors_analyzed'])}  \n"
                
                if env_factors['strong_correlations']:
                    report += "\n**Strong Environmental Correlations:**\n"
                    for corr in env_factors['strong_correlations'][:5]:
                        report += f"- {corr['feature_1']} ↔ {corr['feature_2']}: {corr['correlation']:.3f} ({corr['strength']})\n"
            
            if corr_analysis['insights']:
                report += "\n**Cross-Dataset Insights:**\n"
                for insight in corr_analysis['insights']:
                    report += f"- {insight}\n"
        
        # ML Readiness Assessment
        if 'integrated' in eda_results['insights']:
            ml_readiness = eda_results['insights']['integrated']['ml_readiness']
            
            report += f"\n## ML Readiness Assessment\n\n"
            report += f"**Data Quality Score:** {ml_readiness['data_quality_score']:.1f}/100  \n"
            report += f"**Missing Data:** {ml_readiness['missing_data_percentage']:.1f}%  \n"
            report += f"**Sample Count:** {ml_readiness['sample_count']:,}  \n"
            report += f"**Feature Count:** {ml_readiness['feature_count']}  \n"
            
            if ml_readiness['recommendations']:
                report += "\n**Recommendations:**\n"
                for rec in ml_readiness['recommendations']:
                    report += f"- {rec}\n"
        
        # Feature Analysis
        if 'integrated' in eda_results['insights']:
            feature_summary = eda_results['insights']['integrated']['feature_summary']
            
            report += f"\n## Feature Engineering Insights\n\n"
            report += f"**Total Features:** {feature_summary['total_features']}  \n"
            report += f"**Numeric Features:** {feature_summary['numeric_features']}  \n"
            report += f"**Categorical Features:** {feature_summary['categorical_features']}  \n"
            
            # Feature importance proxy
            if 'feature_importance_proxy' in eda_results['insights']['integrated']:
                feature_importance = eda_results['insights']['integrated']['feature_importance_proxy']
                report += f"\n**Top Varying Features (Variance-based):**\n"
                for i, (feature, variance) in enumerate(list(feature_importance.items())[:5]):
                    report += f"{i+1}. {feature}: {variance:.2f}\n"
        
        # Data Sources
        report += f"\n## Data Sources & Quality\n\n"
        report += "| Dataset | Records | Features | Missing Data | Quality |\n"
        report += "|---------|---------|----------|--------------|----------|\n"
        
        for dataset_name in eda_results['datasets_analyzed']:
            if dataset_name in eda_results['insights']:
                analysis = eda_results['insights'][dataset_name]
                records = analysis['shape'][0]
                features = analysis['shape'][1]
                missing_pct = sum(analysis['missing_data']['percentages'].values()) if analysis['missing_data']['percentages'] else 0
                quality = "Excellent" if missing_pct < 5 else "Good" if missing_pct < 15 else "Fair"
                
                report += f"| {dataset_name.title()} | {records:,} | {features} | {missing_pct:.1f}% | {quality} |\n"
        
        # Plots generated
        report += f"\n## Generated Visualizations\n\n"
        report += f"Total plots generated: {len(eda_results['plots_generated'])}\n\n"
        
        plot_categories = {
            'missing_data': 'Missing Data Analysis',
            'distributions': 'Feature Distributions',
            'correlation': 'Correlation Analysis',
            'summary': 'Summary Visualizations',
            'geographic': 'Geographic Analysis'
        }
        
        for category, title in plot_categories.items():
            category_plots = [p for p in eda_results['plots_generated'] if category in p]
            if category_plots:
                report += f"### {title}\n"
                for plot in category_plots:
                    plot_name = Path(plot).name
                    report += f"- `{plot_name}`\n"
                report += "\n"
        
        # Footer
        report += f"\n---\n\n"
        report += f"**Report Generated:** {eda_results['timestamp']}  \n"
        report += f"**Analysis Framework:** India Environmental Monitoring System EDA Pipeline  \n"
        report += f"**Total Datasets:** {len(eda_results['datasets_analyzed'])}  \n"
        report += f"**Total Visualizations:** {len(eda_results['plots_generated'])}  \n"
        
        return report


# Global EDA generator
eda_generator = EDAGenerator()


def run_complete_eda(datasets: Dict[str, pd.DataFrame], 
                    integrated_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Run complete EDA pipeline.
    
    Args:
        datasets: Dictionary of individual datasets
        integrated_data: Integrated ML-ready dataset
    
    Returns:
        EDA results dictionary
    """
    logger.info("Starting complete EDA pipeline...")
    
    results = eda_generator.generate_complete_eda(datasets, integrated_data)
    
    logger.info("EDA pipeline completed successfully",
               datasets_analyzed=len(results['datasets_analyzed']),
               plots_generated=len(results['plots_generated']))
    
    return results


if __name__ == "__main__":
    # Demo run - requires data to be loaded first
    from data_processing import load_and_prepare_data
    
    datasets, integrated = load_and_prepare_data()
    eda_results = run_complete_eda(datasets, integrated)
    
    print(f"EDA completed!")
    print(f"Datasets analyzed: {len(eda_results['datasets_analyzed'])}")
    print(f"Plots generated: {len(eda_results['plots_generated'])}")
    print(f"Report saved to: ml/eda/reports/eda_report.md")
