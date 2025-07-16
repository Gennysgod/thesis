# data_streams/data_visualizer.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class DataStreamVisualizer:
    """Visualizer for data streams and concept drift"""
    
    def __init__(self, figsize: tuple = (12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')

    def plot_stream_overview(self, df: pd.DataFrame, title: str = "Data Stream Overview"):
        """Plot complete overview of data stream with concept drift - NO CLASS DISTRIBUTION OVER TIME"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        drift_start = 1000
        drift_end = 1500
        
        # Plot 1: Decision Boundary Evolution (替代 Class distribution over time)
        ax1 = axes[0, 0]
        self._plot_decision_boundary_evolution(ax1, df, drift_start, drift_end)
        
        # Plot 2: Feature space visualization
        ax2 = axes[0, 1]
        if 'x1' in df.columns and 'x2' in df.columns:
            scatter = ax2.scatter(df['x1'], df['x2'], c=df['y'],
                                cmap='RdYlBu', alpha=0.6, s=10)
            ax2.set_xlabel('x1')
            ax2.set_ylabel('x2')
            ax2.set_title('Feature Space (colored by class)')
            plt.colorbar(scatter, ax=ax2)
        else:
            ax2.text(0.5, 0.5, 'Feature scatter\nnot available\nfor >2D data',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Feature Space')
        
        # Plot 3: Concept drift statistics
        ax3 = axes[1, 0]
        before_drift = df[df['time_index'] < drift_start]
        during_drift = df[(df['time_index'] >= drift_start) & (df['time_index'] < drift_end)]
        after_drift = df[df['time_index'] >= drift_end]
        
        segments = ['Before Drift', 'During Drift', 'After Drift']
        class_1_ratios_segments = []
        
        for segment_data in [before_drift, during_drift, after_drift]:
            if len(segment_data) > 0:
                ratio = np.mean(segment_data['y'])
                class_1_ratios_segments.append(ratio)
            else:
                class_1_ratios_segments.append(0)
        
        bars = ax3.bar(segments, class_1_ratios_segments,
                       color=['blue', 'orange', 'green'], alpha=0.7)
        ax3.set_ylabel('Class 1 Ratio')
        ax3.set_title('Concept Drift Statistics')
        ax3.set_ylim(0, 1)
        
        for bar, ratio in zip(bars, class_1_ratios_segments):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{ratio:.3f}', ha='center', va='bottom')
        
        # Plot 4: Sample counts
        ax4 = axes[1, 1]
        unique_labels, counts = np.unique(df['y'].values, return_counts=True)
        colors = ['red' if label == 0 else 'blue' for label in unique_labels]
        bars = ax4.bar([f'Class {int(label)}' for label in unique_labels],
                       counts, color=colors, alpha=0.7)
        ax4.set_ylabel('Sample Count')
        ax4.set_title('Overall Class Distribution')
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig

    def _plot_decision_boundary_evolution(self, ax, df: pd.DataFrame, drift_start: int, drift_end: int):
        """Plot decision boundary evolution over time - THIS IS THE NEW PLOT"""
        time_idx = df['time_index'].values
        
        # Create concept indicators based on time
        concepts = np.ones_like(time_idx, dtype=float)
        
        # Map actual time indices to concept phases
        for i, t in enumerate(time_idx):
            if t < drift_start:
                concepts[i] = 1.0  # Concept 1
            elif t < drift_end:
                # Linear transition during drift
                progress = (t - drift_start) / (drift_end - drift_start)
                concepts[i] = 1.0 + progress  # Transition from 1 to 2
            else:
                concepts[i] = 2.0  # Concept 2
        
        # Plot concept evolution
        ax.plot(time_idx, concepts, 'b-', linewidth=2, label='Decision Boundary')
        
        # Mark drift boundaries
        ax.axvline(drift_start, color='r', linestyle='--', alpha=0.7, label='Drift Start')
        ax.axvline(drift_end, color='r', linestyle=':', alpha=0.7, label='Drift End')
        
        # Add concept regions
        ax.axhspan(0.8, 1.2, alpha=0.2, color='green', label='Concept 1')
        ax.axhspan(1.8, 2.2, alpha=0.2, color='red', label='Concept 2')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Concept')
        ax.set_title('Decision Boundary Evolution Over Time')
        ax.set_ylim(0.5, 2.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_decision_boundary_evolution(self, df: pd.DataFrame, generator_type: str = "SINE1"):
        """Plot how decision boundary changes over time"""
        if 'x1' not in df.columns or 'x2' not in df.columns:
            print("Decision boundary visualization only available for 2D data")
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        drift_start = 1000
        drift_end = 1500
        
        phases = [
            ("Before Drift", df[df['time_index'] < drift_start]),
            ("During Drift", df[(df['time_index'] >= drift_start) & (df['time_index'] < drift_end)]),
            ("After Drift", df[df['time_index'] >= drift_end])
        ]
        
        for i, (phase_name, phase_data) in enumerate(phases):
            ax = axes[i]
            
            if len(phase_data) > 0:
                scatter = ax.scatter(phase_data['x1'], phase_data['x2'],
                                   c=phase_data['y'], cmap='RdYlBu',
                                   alpha=0.6, s=20)
                
                if generator_type == "SINE1":
                    x1_line = np.linspace(0, 1, 100)
                    if phase_name == "Before Drift":
                        x2_line = np.sin(x1_line)
                    elif phase_name == "After Drift":
                        x2_line = np.sin(x1_line + 0.15 * np.pi)
                    else:
                        x2_line = np.sin(x1_line + 0.075 * np.pi)
                    
                    valid_mask = (x2_line >= 0) & (x2_line <= 1)
                    ax.plot(x1_line[valid_mask], x2_line[valid_mask],
                           'k-', linewidth=3, label='Decision Boundary')
                
                ax.set_xlabel('x1')
                ax.set_ylabel('x2')
                ax.set_title(f'{phase_name}\n({len(phase_data)} samples)')
                ax.legend()
                plt.colorbar(scatter, ax=ax)
            else:
                ax.text(0.5, 0.5, f'No data\nin {phase_name}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(phase_name)
        
        plt.tight_layout()
        return fig

    def plot_class_imbalance_validation(self, df: pd.DataFrame, expected_ratio: List[float]):
        """Validate class imbalance matches expected ratio"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        total_samples = len(df)
        class_counts = df['y'].value_counts().sort_index()
        actual_ratios = (class_counts / total_samples).values
        
        # Expected vs Actual comparison
        ax1 = axes[0]
        x = np.arange(len(expected_ratio))
        width = 0.35
        
        expected_normalized = np.array(expected_ratio) / sum(expected_ratio)
        
        bars1 = ax1.bar(x - width/2, expected_normalized, width,
                       label='Expected', alpha=0.7, color='blue')
        bars2 = ax1.bar(x + width/2, actual_ratios, width,
                       label='Actual', alpha=0.7, color='orange')
        
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Ratio')
        ax1.set_title('Expected vs Actual Class Ratios')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Class {i}' for i in range(len(expected_ratio))])
        ax1.legend()
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # Class balance stability over time
        ax2 = axes[1]
        window_size = min(100, len(df) // 10)
        time_windows = []
        class_1_ratios = []
        
        for i in range(0, len(df) - window_size, window_size//2):
            window_data = df.iloc[i:i + window_size]
            ratio = np.mean(window_data['y'])
            time_windows.append(window_data['time_index'].mean())
            class_1_ratios.append(ratio)
        
        ax2.plot(time_windows, class_1_ratios, 'b-', linewidth=2)
        target_ratio = expected_ratio[1] / sum(expected_ratio)
        ax2.axhline(target_ratio, color='r', linestyle='--',
                   label=f'Target Ratio ({target_ratio:.3f})')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Class 1 Ratio')
        ax2.set_title('Class Balance Stability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def validate_data_quality(self, df: pd.DataFrame, expected_ratio: List[float], 
                            generator_type: str = "SINE1") -> Dict[str, Any]:
        """Comprehensive data quality validation"""
        validation_results = {}
        
        validation_results['total_samples'] = len(df)
        validation_results['n_features'] = len([col for col in df.columns if col.startswith('x')])
        
        class_counts = df['y'].value_counts().sort_index()
        actual_ratios = (class_counts / len(df)).values
        expected_normalized = np.array(expected_ratio) / sum(expected_ratio)
        
        validation_results['class_counts'] = class_counts.to_dict()
        validation_results['actual_ratios'] = actual_ratios.tolist()
        validation_results['expected_ratios'] = expected_normalized.tolist()
        validation_results['ratio_difference'] = np.abs(actual_ratios - expected_normalized).max()
        
        drift_start, drift_end = 1000, 1500
        before_drift = df[df['time_index'] < drift_start]
        during_drift = df[(df['time_index'] >= drift_start) & (df['time_index'] < drift_end)]
        after_drift = df[df['time_index'] >= drift_end]
        
        validation_results['drift_segments'] = {
            'before_samples': len(before_drift),
            'during_samples': len(during_drift), 
            'after_samples': len(after_drift)
        }
        
        if len(before_drift) > 0 and len(after_drift) > 0:
            ratio_before = np.mean(before_drift['y'])
            ratio_after = np.mean(after_drift['y'])
            validation_results['ratio_change'] = abs(ratio_after - ratio_before)
        
        validation_results['quality_flags'] = {
            'correct_sample_count': len(df) == 2000,  # 新增：检查样本数量
            'balanced_samples': validation_results['ratio_difference'] < 0.1,  # 放宽一点
            'sufficient_drift_data': len(during_drift) > 0,
            'no_missing_values': not df.isnull().any().any(),
            'proper_time_sequence': df['time_index'].is_monotonic_increasing,
            'both_classes_present': len(class_counts) == 2,
            'all_segments_have_both_classes': (
                len(before_drift) > 0 and len(before_drift['y'].unique()) == 2 and 
                len(after_drift) > 0 and len(after_drift['y'].unique()) == 2
            )
        }
        
        validation_results['overall_quality'] = all(validation_results['quality_flags'].values())
        
        return validation_results

    def generate_validation_report(self, df: pd.DataFrame, expected_ratio: List[float],
                                 generator_type: str = "SINE1", save_path: str = None):
        """Generate comprehensive validation report with plots"""
        print("="*60)
        print("DATA STREAM VALIDATION REPORT")
        print("="*60)
        
        results = self.validate_data_quality(df, expected_ratio, generator_type)
        
        print(f"Generator Type: {generator_type}")
        print(f"Total Samples: {results['total_samples']} (Expected: 2000)")
        print(f"Number of Features: {results['n_features']}")
        print(f"Overall Quality: {'PASS' if results['overall_quality'] else 'FAIL'}")
        print()
        
        print("CLASS DISTRIBUTION:")
        for i, (actual, expected) in enumerate(zip(results['actual_ratios'], results['expected_ratios'])):
            print(f"  Class {i}: {actual:.3f} (expected: {expected:.3f})")
        print(f"  Max Ratio Difference: {results['ratio_difference']:.3f}")
        print()
        
        print("DRIFT SEGMENTS:")
        for segment, count in results['drift_segments'].items():
            print(f"  {segment}: {count}")
        print()
        
        print("QUALITY CHECKS:")
        for flag, status in results['quality_flags'].items():
            status_str = "PASS" if status else "FAIL"
            print(f"  {flag}: {status_str}")
        
        print("\nGenerating validation plots...")
        
        fig1 = self.plot_stream_overview(df, f"{generator_type} Data Stream")
        fig2 = self.plot_decision_boundary_evolution(df, generator_type)
        fig3 = self.plot_class_imbalance_validation(df, expected_ratio)
        
        if save_path:
            import os
            os.makedirs(save_path, exist_ok=True)
            
            fig1.savefig(os.path.join(save_path, f"{generator_type}_overview.png"),
                        dpi=300, bbox_inches='tight')
            if fig2:
                fig2.savefig(os.path.join(save_path, f"{generator_type}_boundary.png"),
                            dpi=300, bbox_inches='tight')
            fig3.savefig(os.path.join(save_path, f"{generator_type}_imbalance.png"),
                        dpi=300, bbox_inches='tight')
            
            print(f"Plots saved to: {save_path}")
        
        plt.show()
        return results