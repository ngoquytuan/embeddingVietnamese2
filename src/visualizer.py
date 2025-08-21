import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import json

# Cấu hình matplotlib cho tiếng Việt
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Tahoma', 'DejaVu Sans']
sns.set_style("whitegrid")
sns.set_palette("husl")

class ResultVisualizer:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Cấu hình màu sắc
        self.colors = sns.color_palette("husl", 10)
        
    def plot_model_comparison(self, df_comparison: pd.DataFrame, save_path: str = None) -> None:
        """Vẽ biểu đồ so sánh hiệu suất các models"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('So sánh Hiệu suất các Models Embedding Tiếng Việt', fontsize=16, fontweight='bold')
        
        # MRR Comparison
        axes[0, 0].barh(df_comparison['Model'], df_comparison['MRR'], color=self.colors[0])
        axes[0, 0].set_title('Mean Reciprocal Rank (MRR)')
        axes[0, 0].set_xlabel('MRR Score')
        for i, v in enumerate(df_comparison['MRR']):
            axes[0, 0].text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # Hit Rates Comparison
        hit_metrics = ['Hit_Rate@1', 'Hit_Rate@3', 'Hit_Rate@5']
        x = np.arange(len(df_comparison['Model']))
        width = 0.25
        
        for i, metric in enumerate(hit_metrics):
            if metric in df_comparison.columns:
                axes[0, 1].bar(x + i*width, df_comparison[metric], width, 
                              label=metric, color=self.colors[i+1])
        
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('Hit Rate')
        axes[0, 1].set_title('Hit Rates Comparison')
        axes[0, 1].set_xticks(x + width)
        axes[0, 1].set_xticklabels([name.split('/')[-1] for name in df_comparison['Model']], rotation=45)
        axes[0, 1].legend()
        
        # NDCG@5 và MAP@5
        if 'NDCG@5' in df_comparison.columns and 'MAP@5' in df_comparison.columns:
            x = np.arange(len(df_comparison['Model']))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, df_comparison['NDCG@5'], width, 
                          label='NDCG@5', color=self.colors[4])
            axes[1, 0].bar(x + width/2, df_comparison['MAP@5'], width, 
                          label='MAP@5', color=self.colors[5])
            
            axes[1, 0].set_xlabel('Models')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title('NDCG@5 và MAP@5 Comparison')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels([name.split('/')[-1] for name in df_comparison['Model']], rotation=45)
            axes[1, 0].legend()
        
        # Overall Ranking
        ranking_metrics = ['MRR_rank', 'Hit_Rate@1_rank', 'Hit_Rate@3_rank', 'Hit_Rate@5_rank']
        available_rankings = [col for col in ranking_metrics if col in df_comparison.columns]
        
        if available_rankings:
            avg_rank = df_comparison[available_rankings].mean(axis=1)
            axes[1, 1].barh(df_comparison['Model'], avg_rank, color=self.colors[6])
            axes[1, 1].set_title('Average Ranking (Lower is Better)')
            axes[1, 1].set_xlabel('Average Rank')
            for i, v in enumerate(avg_rank):
                axes[1, 1].text(v + 0.1, i, f'{v:.1f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Đã lưu biểu đồ so sánh models: {save_path}")
        
        plt.show()
    
    def plot_detailed_analysis(self, model_name: str, df_detailed: pd.DataFrame, 
                              metrics_summary: Dict, save_path: str = None) -> None:
        """Vẽ biểu đồ phân tích chi tiết cho một model"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        model_short_name = model_name.split('/')[-1]
        fig.suptitle(f'Phân tích Chi tiết - {model_short_name}', fontsize=16, fontweight='bold')
        
        # 1. Distribution of Found Ranks
        ranks = df_detailed[df_detailed['found_rank'].notna()]['found_rank']
        if len(ranks) > 0:
            axes[0, 0].hist(ranks, bins=range(1, 7), alpha=0.7, color=self.colors[0])
            axes[0, 0].set_xlabel('Rank của Câu trả lời Đúng')
            axes[0, 0].set_ylabel('Số lượng Câu hỏi')
            axes[0, 0].set_title('Phân bố Rank của Câu trả lời Đúng')
            axes[0, 0].set_xticks(range(1, 6))
        
        # 2. Score Distribution
        correct_scores = df_detailed[df_detailed['correct_score'] > 0]['correct_score']
        top1_scores = df_detailed['top_1_score']
        
        axes[0, 1].hist([correct_scores, top1_scores], label=['Correct Answer Score', 'Top-1 Score'], 
                       alpha=0.7, color=self.colors[1:3])
        axes[0, 1].set_xlabel('Cosine Similarity Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Phân bố Điểm số')
        axes[0, 1].legend()
        
        # 3. Performance by Question Position
        question_performance = df_detailed.groupby(df_detailed.index // 4)['hit@1'].mean()  # Group every 4 questions
        axes[0, 2].plot(question_performance.index, question_performance.values, 
                       marker='o', color=self.colors[3])
        axes[0, 2].set_xlabel('Nhóm Câu hỏi (mỗi nhóm 4 câu)')
        axes[0, 2].set_ylabel('Hit Rate@1')
        axes[0, 2].set_title('Hiệu suất theo Nhóm Câu hỏi')
        
        # 4. Metrics Overview
        metrics_names = ['MRR', 'Hit_Rate@1', 'Hit_Rate@3', 'Hit_Rate@5', 'MAP@5', 'NDCG@5']
        available_metrics = [m for m in metrics_names if m in metrics_summary]
        metric_values = [metrics_summary[m] for m in available_metrics]
        
        axes[1, 0].bar(available_metrics, metric_values, color=self.colors[4])
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Tổng quan Các Chỉ số')
        axes[1, 0].set_xticklabels(available_metrics, rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(metric_values):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 5. Question Difficulty Analysis
        df_detailed['difficulty'] = 'Easy'
        df_detailed.loc[df_detailed['found_rank'] > 3, 'difficulty'] = 'Hard'
        df_detailed.loc[df_detailed['found_rank'].isna(), 'difficulty'] = 'Very Hard'
        
        difficulty_counts = df_detailed['difficulty'].value_counts()
        axes[1, 1].pie(difficulty_counts.values, labels=difficulty_counts.index, 
                      autopct='%1.1f%%', colors=self.colors[5:8])
        axes[1, 1].set_title('Phân loại Độ khó Câu hỏi')
        
        # 6. Score Gap Analysis
        df_detailed['score_gap'] = df_detailed['top_1_score'] - df_detailed['correct_score']
        score_gaps = df_detailed[df_detailed['correct_score'] > 0]['score_gap']
        
        axes[1, 2].hist(score_gaps, bins=20, alpha=0.7, color=self.colors[8])
        axes[1, 2].set_xlabel('Score Gap (Top-1 - Correct)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Khoảng cách Điểm số')
        axes[1, 2].axvline(0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Đã lưu biểu đồ phân tích chi tiết: {save_path}")
        
        plt.show()
    
    def create_performance_heatmap(self, all_model_results: Dict, save_path: str = None) -> None:
        """Tạo heatmap so sánh hiệu suất tất cả models"""
        # Chuẩn bị dữ liệu
        models = []
        metrics_data = []
        
        for model_name, result in all_model_results.items():
            models.append(model_name.split('/')[-1])
            metrics = result['summary_metrics']
            metrics_data.append([
                metrics.get('MRR', 0),
                metrics.get('Hit_Rate@1', 0),
                metrics.get('Hit_Rate@3', 0),
                metrics.get('Hit_Rate@5', 0),
                metrics.get('MAP@5', 0),
                metrics.get('NDCG@5', 0)
            ])
        
        df_heatmap = pd.DataFrame(
            metrics_data, 
            index=models,
            columns=['MRR', 'Hit@1', 'Hit@3', 'Hit@5', 'MAP@5', 'NDCG@5']
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_heatmap, annot=True, cmap='YlOrRd', fmt='.3f', 
                   cbar_kws={'label': 'Performance Score'})
        plt.title('Performance Heatmap - All Models', fontsize=14, fontweight='bold')
        plt.ylabel('Models')
        plt.xlabel('Metrics')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Đã lưu heatmap: {save_path}")
        
        plt.show()
    
    def generate_performance_report(self, all_model_results: Dict, 
                                   output_file: str = "performance_report.html") -> None:
        """Tạo báo cáo HTML tổng hợp"""
        html_content = """
        <html>
        <head>
            <title>Vietnamese Embedding Models - Performance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1, h2 { color: #2c3e50; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: center; }
                th { background-color: #f8f9fa; font-weight: bold; }
                .best { background-color: #d4edda; font-weight: bold; }
                .second { background-color: #e2f3ff; }
                .summary { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }
            </style>
        </head>
        <body>
        """
        
        html_content += "<h1>Vietnamese Embedding Models - Performance Report</h1>"
        
        # Tạo bảng so sánh
        models_data = []
        for model_name, result in all_model_results.items():
            models_data.append({
                'Model': model_name.split('/')[-1],
                'Full_Name': model_name,
                'Dimension': result.get('embedding_dimension', 'N/A'),
                'Time': result.get('evaluation_time_seconds', 'N/A'),
                **result['summary_metrics']
            })
        
        df = pd.DataFrame(models_data)
        df = df.sort_values('MRR', ascending=False)
        
        html_content += "<h2>Model Performance Comparison</h2>"
        html_content += df.to_html(classes='performance-table', escape=False, index=False)
        
        # Top performer summary
        best_model = df.iloc[0]
        html_content += f"""
        <div class="summary">
            <h3>🏆 Best Performing Model</h3>
            <p><strong>{best_model['Model']}</strong> achieved the highest MRR score of <strong>{best_model['MRR']:.4f}</strong></p>
            <ul>
                <li>Hit Rate@1: {best_model['Hit_Rate@1']:.2%}</li>
                <li>Hit Rate@5: {best_model['Hit_Rate@5']:.2%}</li>
                <li>Embedding Dimension: {best_model['Dimension']}</li>
                <li>Evaluation Time: {best_model['Time']:.2f}s</li>
            </ul>
        </div>
        """
        
        html_content += "</body></html>"
        
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Đã tạo báo cáo HTML: {output_path}")