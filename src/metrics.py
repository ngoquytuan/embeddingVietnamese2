import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict

class MetricsCalculator:
    def __init__(self):
        self.metrics_names = [
            'MRR', 'Hit_Rate@1', 'Hit_Rate@3', 'Hit_Rate@5', 
            'MAP@5', 'NDCG@5', 'Precision@5', 'Recall@5'
        ]
    
    def calculate_reciprocal_rank(self, correct_chunk_id: str, ranked_results: List[Dict]) -> float:
        """Tính Reciprocal Rank cho một query"""
        for rank, result in enumerate(ranked_results, 1):
            if result['chunk_id'] == correct_chunk_id:
                return 1.0 / rank
        return 0.0
    
    def calculate_hit_rate_at_k(self, correct_chunk_id: str, ranked_results: List[Dict], k: int) -> int:
        """Tính Hit Rate@k cho một query"""
        top_k_ids = [result['chunk_id'] for result in ranked_results[:k]]
        return 1 if correct_chunk_id in top_k_ids else 0
    
    def calculate_precision_at_k(self, correct_chunk_id: str, ranked_results: List[Dict], k: int) -> float:
        """Tính Precision@k"""
        top_k_ids = [result['chunk_id'] for result in ranked_results[:k]]
        relevant_found = sum(1 for chunk_id in top_k_ids if chunk_id == correct_chunk_id)
        return relevant_found / min(k, len(ranked_results))
    
    def calculate_recall_at_k(self, correct_chunk_id: str, ranked_results: List[Dict], k: int) -> float:
        """Tính Recall@k (trong trường hợp này, mỗi query chỉ có 1 correct answer)"""
        top_k_ids = [result['chunk_id'] for result in ranked_results[:k]]
        return 1.0 if correct_chunk_id in top_k_ids else 0.0
    
    def calculate_average_precision(self, correct_chunk_id: str, ranked_results: List[Dict]) -> float:
        """Tính Average Precision"""
        precision_at_k = []
        for k in range(1, len(ranked_results) + 1):
            if ranked_results[k-1]['chunk_id'] == correct_chunk_id:
                precision = self.calculate_precision_at_k(correct_chunk_id, ranked_results, k)
                precision_at_k.append(precision)
        
        return np.mean(precision_at_k) if precision_at_k else 0.0
    
    def calculate_ndcg_at_k(self, correct_chunk_id: str, ranked_results: List[Dict], k: int) -> float:
        """Tính NDCG@k"""
        # Tính DCG@k
        dcg = 0.0
        for i, result in enumerate(ranked_results[:k]):
            if result['chunk_id'] == correct_chunk_id:
                relevance = 1  # Binary relevance
                dcg += relevance / np.log2(i + 2)  # i+2 vì log2(1) = 0
        
        # Tính IDCG@k (trong trường hợp này là 1.0 vì chỉ có 1 correct answer)
        idcg = 1.0  # 1 / log2(2) = 1
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_all_metrics(self, test_results: List[Dict]) -> Dict:
        """Tính tất cả metrics cho toàn bộ test results"""
        if not test_results:
            return {metric: 0.0 for metric in self.metrics_names}
        
        rr_scores = []
        hit_1_scores = []
        hit_3_scores = []
        hit_5_scores = []
        ap_scores = []
        ndcg_5_scores = []
        precision_5_scores = []
        recall_5_scores = []
        
        for result in test_results:
            correct_id = result['correct_chunk_id']
            ranked_results = result['top_5_results']
            
            # Reciprocal Rank
            rr = self.calculate_reciprocal_rank(correct_id, ranked_results)
            rr_scores.append(rr)
            
            # Hit Rates
            hit_1_scores.append(self.calculate_hit_rate_at_k(correct_id, ranked_results, 1))
            hit_3_scores.append(self.calculate_hit_rate_at_k(correct_id, ranked_results, 3))
            hit_5_scores.append(self.calculate_hit_rate_at_k(correct_id, ranked_results, 5))
            
            # Average Precision
            ap = self.calculate_average_precision(correct_id, ranked_results)
            ap_scores.append(ap)
            
            # NDCG@5
            ndcg_5 = self.calculate_ndcg_at_k(correct_id, ranked_results, 5)
            ndcg_5_scores.append(ndcg_5)
            
            # Precision@5 và Recall@5
            precision_5 = self.calculate_precision_at_k(correct_id, ranked_results, 5)
            recall_5 = self.calculate_recall_at_k(correct_id, ranked_results, 5)
            precision_5_scores.append(precision_5)
            recall_5_scores.append(recall_5)
        
        # Tính mean cho tất cả metrics
        metrics = {
            'MRR': np.mean(rr_scores),
            'Hit_Rate@1': np.mean(hit_1_scores),
            'Hit_Rate@3': np.mean(hit_3_scores),
            'Hit_Rate@5': np.mean(hit_5_scores),
            'MAP@5': np.mean(ap_scores),
            'NDCG@5': np.mean(ndcg_5_scores),
            'Precision@5': np.mean(precision_5_scores),
            'Recall@5': np.mean(recall_5_scores)
        }
        
        return metrics
    
    def calculate_detailed_metrics(self, test_results: List[Dict]) -> Tuple[Dict, pd.DataFrame]:
        """Tính metrics chi tiết cho từng câu hỏi"""
        detailed_results = []
        
        for i, result in enumerate(test_results):
            correct_id = result['correct_chunk_id']
            ranked_results = result['top_5_results']
            question = result['question']
            
            # Tìm rank của correct answer
            found_rank = None
            for rank, res in enumerate(ranked_results, 1):
                if res['chunk_id'] == correct_id:
                    found_rank = rank
                    break
            
            detailed_result = {
                'question_id': i + 1,
                'question': question[:50] + "..." if len(question) > 50 else question,
                'correct_chunk_id': correct_id,
                'found_rank': found_rank,
                'reciprocal_rank': self.calculate_reciprocal_rank(correct_id, ranked_results),
                'hit@1': self.calculate_hit_rate_at_k(correct_id, ranked_results, 1),
                'hit@3': self.calculate_hit_rate_at_k(correct_id, ranked_results, 3),
                'hit@5': self.calculate_hit_rate_at_k(correct_id, ranked_results, 5),
                'ap': self.calculate_average_precision(correct_id, ranked_results),
                'ndcg@5': self.calculate_ndcg_at_k(correct_id, ranked_results, 5),
                'top_1_score': ranked_results[0]['score'] if ranked_results else 0.0,
                'correct_score': next((res['score'] for res in ranked_results if res['chunk_id'] == correct_id), 0.0)
            }
            detailed_results.append(detailed_result)
        
        df_detailed = pd.DataFrame(detailed_results)
        summary_metrics = self.calculate_all_metrics(test_results)
        
        return summary_metrics, df_detailed
    
    def get_performance_analysis(self, df_detailed: pd.DataFrame) -> Dict:
        """Phân tích hiệu suất chi tiết"""
        analysis = {
            'total_questions': len(df_detailed),
            'questions_answered_correctly_at_rank_1': int(df_detailed['hit@1'].sum()),
            'questions_answered_correctly_at_rank_3': int(df_detailed['hit@3'].sum()),
            'questions_answered_correctly_at_rank_5': int(df_detailed['hit@5'].sum()),
            'questions_not_found_in_top_5': int((df_detailed['found_rank'].isna()).sum()),
            'average_correct_answer_rank': df_detailed[df_detailed['found_rank'].notna()]['found_rank'].mean(),
            'median_correct_answer_rank': df_detailed[df_detailed['found_rank'].notna()]['found_rank'].median(),
            'score_statistics': {
                'top_1_score_mean': df_detailed['top_1_score'].mean(),
                'top_1_score_std': df_detailed['top_1_score'].std(),
                'correct_score_mean': df_detailed[df_detailed['correct_score'] > 0]['correct_score'].mean(),
                'correct_score_std': df_detailed[df_detailed['correct_score'] > 0]['correct_score'].std()
            }
        }
        
        return analysis
    
    def compare_models(self, model_results: Dict[str, Dict]) -> pd.DataFrame:
        """So sánh hiệu suất giữa các models"""
        comparison_data = []
        
        for model_name, metrics in model_results.items():
            row = {'Model': model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Sắp xếp theo MRR (cao nhất trước)
        df_comparison = df_comparison.sort_values('MRR', ascending=False).reset_index(drop=True)
        
        # Thêm ranking cho từng metric
        for metric in self.metrics_names:
            if metric in df_comparison.columns:
                df_comparison[f'{metric}_rank'] = df_comparison[metric].rank(ascending=False, method='min')
        
        return df_comparison