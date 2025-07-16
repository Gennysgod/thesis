import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
import warnings
import os
warnings.filterwarnings('ignore')

class ExperimentVisualizer:
    """Visualizer for experiment results and drift detection"""
    
    def __init__(self, figsize: tuple = (12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
        
        # Color palettes
        self.detector_colors = {
            'ADWIN': '#1f77b4',
            'DDM': '#ff7f0e',
            'QuadCDD': '#2ca02c'
        }
        
    def plot_detection_timeline(self, detected_drifts: List[int],
                               true_drift_start: int = 1000,
                               true_drift_end: int = 1500,
                               stream_length: int = 2000,
                               detector_name: str = "Detector",
                               save_path: str = None) -> plt.Figure:
        """Plot detection timeline with true drift points"""
        fig, ax = plt.subplots(1, 1, figsize=(15, 4))
        
        # Timeline
        timeline = np.arange(stream_length)
        baseline = np.zeros(stream_length)
        
        # Plot baseline
        ax.plot(timeline, baseline, 'k-', alpha=0.3, linewidth=1)
        
        # Mark true drift period
        ax.axvspan(true_drift_start, true_drift_end, alpha=0.2, color='red',
                  label=f'True Drift Period ({true_drift_start}-{true_drift_end})')
        
        # Mark true drift start
        ax.axvline(true_drift_start, color='red', linestyle='--', linewidth=2,
                  label=f'True Drift Start ({true_drift_start})')
        
        # Mark detections
        for i, detection in enumerate(detected_drifts):
            ax.axvline(detection, color='blue', linestyle='-', alpha=0.7,
                      label='Detection' if i == 0 else "", linewidth=1.5)
            ax.plot(detection, 0, 'bo', markersize=8, alpha=0.7)
            ax.text(detection, 0.1, f'D{i+1}', ha='center', va='bottom',
                   fontsize=9, color='blue')
        
        # Formatting
        ax.set_xlabel('Time Step')
        ax.set_ylabel('')
        ax.set_title(f'{detector_name} - Detection Timeline')
        ax.set_ylim(-0.2, 0.3)
        ax.set_xlim(0, stream_length)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add detection count
        ax.text(0.02, 0.95, f'Total Detections: {len(detected_drifts)}',
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        return fig
    
    def plot_metric_comparison(self, results_df: pd.DataFrame,
                              metric: str = 'drift_detection_delay',
                              save_path: str = None) -> plt.Figure:
        """Plot metric comparison across detectors and data groups"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{metric.replace("_", " ").title()} Analysis', fontsize=16)
        
        # Use the correct column name format
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        
        # Plot 1: Bar plot by detector
        ax1 = axes[0, 0]
        detector_means = results_df.groupby('detector')[mean_col].mean()
        bars = ax1.bar(detector_means.index, detector_means.values,
                       color=[self.detector_colors.get(d, 'gray') for d in detector_means.index],
                       alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, detector_means.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        ax1.set_title('Average by Detector')
        ax1.set_ylabel(metric.replace("_", " ").title())
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Bar plot by data group
        ax2 = axes[0, 1]
        group_means = results_df.groupby('data_group')[mean_col].mean().sort_values()
        bars = ax2.bar(range(len(group_means)), group_means.values, alpha=0.7)
        
        for bar, value in zip(bars, group_means.values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        ax2.set_title('Average by Data Group')
        ax2.set_ylabel(metric.replace("_", " ").title())
        ax2.set_xticks(range(len(group_means)))
        ax2.set_xticklabels(group_means.index, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Heatmap
        ax3 = axes[1, 0]
        pivot_data = results_df.pivot_table(values=mean_col,
                                           index='detector',
                                           columns='data_group',
                                           aggfunc='mean')
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlBu_r',
                   ax=ax3, cbar_kws={'label': metric.replace("_", " ").title()})
        ax3.set_title('Heatmap: Detector vs Data Group')
        
        # Plot 4: Box plot for distribution
        ax4 = axes[1, 1]
        box_data = []
        box_labels = []
        for detector in results_df['detector'].unique():
            detector_data = results_df[results_df['detector'] == detector][mean_col]
            box_data.append(detector_data.values)
            box_labels.append(detector)
        
        bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
        
        for patch, detector in zip(bp['boxes'], box_labels):
            patch.set_facecolor(self.detector_colors.get(detector, 'gray'))
            patch.set_alpha(0.7)
        
        ax4.set_title('Distribution by Detector')
        ax4.set_ylabel(metric.replace("_", " ").title())
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        return fig
    
    def plot_comprehensive_results(self, results_df: pd.DataFrame,
                                  save_path: str = None) -> plt.Figure:
        """Plot comprehensive results overview"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Experiment Results', fontsize=16)
        
        metrics = ['drift_detection_delay_mean', 'false_positive_rate_mean',
                   'missed_drift_count_mean', 'drift_detection_recall_mean']
        titles = ['Detection Delay', 'False Positive Rate',
                  'Missed Drift Count', 'Detection Recall']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            if metric not in results_df.columns:
                ax.text(0.5, 0.5, f'{metric} not available',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                continue
            
            # Group data by detector
            grouped_data = []
            detector_names = []
            for detector in results_df['detector'].unique():
                detector_data = results_df[results_df['detector'] == detector][metric]
                grouped_data.append(detector_data.values)
                detector_names.append(detector)
            
            # Box plot
            bp = ax.boxplot(grouped_data, labels=detector_names, patch_artist=True)
            
            for patch, detector in zip(bp['boxes'], detector_names):
                patch.set_facecolor(self.detector_colors.get(detector, 'gray'))
                patch.set_alpha(0.7)
            
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Add mean values as text
            for i, detector in enumerate(detector_names):
                mean_val = np.mean(grouped_data[i])
                ax.text(i+1, ax.get_ylim()[1]*0.9, f'μ={mean_val:.3f}',
                       ha='center', va='center', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        return fig
    
    def create_experiment_report(self, results_df: pd.DataFrame,
                                output_dir: str = "results/figures/") -> List[plt.Figure]:
        """Create comprehensive experiment report with all visualizations"""
        print("Generating comprehensive experiment report...")
        figures = []
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Comprehensive results overview
        fig1 = self.plot_comprehensive_results(
            results_df, 
            save_path=os.path.join(output_dir, 'comprehensive_results.png')
        )
        figures.append(('comprehensive_results', fig1))
        
        # 2. Individual metric comparisons
        metrics = ['drift_detection_delay', 'false_positive_rate',
                   'missed_drift_count', 'drift_detection_recall']
        
        for metric in metrics:
            if f'{metric}_mean' in results_df.columns:
                fig = self.plot_metric_comparison(
                    results_df, metric,
                    save_path=os.path.join(output_dir, f'metric_{metric}.png')
                )
                figures.append((f'metric_{metric}', fig))
        
        return [fig for _, fig in figures]