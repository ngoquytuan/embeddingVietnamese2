import os
import json
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import numpy as np
from pyvi import ViTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import torch
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

from src.data_processor import DataProcessor
from src.embedding_manager import EmbeddingManager
from src.metrics import MetricsCalculator
from src.visualizer import ResultVisualizer

def print_banner():
    """In banner chào mừng"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                  🇻🇳 VIETNAMESE EMBEDDING EVALUATOR 🇻🇳                ║
    ║                                                                      ║
    ║              Đánh giá hiệu suất các model embedding tiếng Việt       ║
    ║                         Phiên bản 1.0 - 2024                        ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def setup_directories():
    """Tạo các thư mục cần thiết"""
    directories = [
        'reports', 
        'reports/visualizations', 
        'reports/individual_models',
        'model_cache', 
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("📁 Đã tạo cấu trúc thư mục")

def check_gpu_availability():
    """Kiểm tra tình trạng GPU"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🚀 GPU khả dụng: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
        return True
    else:
        print("💻 Chỉ có CPU khả dụng (không có GPU)")
        return False

def load_configuration():
    """Load cấu hình models và test suite"""
    print("\n🔧 Đang load cấu hình...")
    
    # Load models list
    try:
        config_path = Path('configs/models.json')
        if not config_path.exists():
            print(f"❌ Không tìm thấy file: {config_path}")
            return None, None
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        models_to_test = config.get('models', [])
        if not models_to_test:
            print("❌ Không có models nào trong config")
            return None, None
            
        print(f"✅ Đã load {len(models_to_test)} models để test:")
        for i, model in enumerate(models_to_test, 1):
            print(f"   {i}. {model}")
            
    except Exception as e:
        print(f"❌ Lỗi khi load models config: {e}")
        return None, None
    
    # Load test suite
    try:
        test_suite_path = Path('data/test_suite.json')
        if not test_suite_path.exists():
            print(f"❌ Không tìm thấy file: {test_suite_path}")
            return None, None
            
        with open(test_suite_path, 'r', encoding='utf-8') as f:
            test_suite = json.load(f)
            
        if not test_suite:
            print("❌ Test suite trống")
            return None, None
            
        print(f"✅ Đã load {len(test_suite)} câu hỏi test")
        
    except Exception as e:
        print(f"❌ Lỗi khi load test suite: {e}")
        return None, None
    
    return models_to_test, test_suite

def process_content_data():
    """Xử lý dữ liệu content"""
    print("\n📚 Đang xử lý dữ liệu content...")
    
    content_path = Path('data/content.md')
    if not content_path.exists():
        print(f"❌ Không tìm thấy file: {content_path}")
        return None
    
    processor = DataProcessor()
    
    # Thử các phương pháp chunking khác nhau
    chunk_methods = ['sentences', 'words']
    best_chunks = None
    
    for method in chunk_methods:
        print(f"\n📝 Thử phương pháp chunking: {method}")
        try:
            chunks = processor.load_and_process_content(str(content_path), chunk_method=method)
            if chunks:
                best_chunks = chunks
                print(f"✅ Sử dụng phương pháp: {method}")
                break
        except Exception as e:
            print(f"⚠️  Phương pháp {method} gặp lỗi: {e}")
            continue
    
    if not best_chunks:
        print("❌ Không thể tạo chunks bằng bất kỳ phương pháp nào")
        return None
    
    # Lưu thông tin chunks
    chunks_info_path = Path("reports") / "chunks_info.json"
    processor.save_chunks_info(best_chunks, str(chunks_info_path))
    
    return best_chunks, processor

def evaluate_single_model(model_name: str, chunks: list, test_suite: list, 
                         chunk_texts: list, chunk_ids: list, processor: DataProcessor) -> dict:
    """Đánh giá một model duy nhất"""
    print(f"\n{'='*60}")
    print(f"🚀 Đang đánh giá model: {model_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Khởi tạo embedding manager
        print("📥 Đang tải model...")
        manager = EmbeddingManager(model_name, cache_dir="model_cache")
        model_info = manager.get_model_info()
        
        print(f"ℹ️  Thông tin model:")
        print(f"   - Tên: {model_info['name']}")
        print(f"   - Dimension: {model_info['dimension']}")
        print(f"   - Device: {model_info['device']}")
        print(f"   - Max sequence length: {model_info['max_sequence_length']}")
        
        # Validate test suite với chunks
        is_valid = processor.validate_test_suite(test_suite, chunk_ids)
        if not is_valid:
            print("⚠️  Test suite có vấn đề, nhưng sẽ tiếp tục...")
        
        # Encode corpus
        print(f"\n📝 Đang encode {len(chunk_texts)} chunks...")
        corpus_embeddings = manager.encode_texts(
            chunk_texts, 
            batch_size=32, 
            show_progress=True
        )
        
        print(f"✅ Đã encode corpus: {corpus_embeddings.shape}")
        
        # Process test cases
        print(f"\n🔍 Đang xử lý {len(test_suite)} câu hỏi test...")
        test_case_results = []
        
        questions = [test_case['question'] for test_case in test_suite]
        
        # Batch encode questions
        print("📝 Đang encode các câu hỏi...")
        question_embeddings = manager.encode_texts(questions, show_progress=False)
        
        # Process each question
        for i, (test_case, query_embedding) in enumerate(zip(test_suite, question_embeddings)):
            question = test_case['question']
            correct_chunk_id = test_case['correct_chunk_id']
            
            # Find similar chunks
            top_indices, top_scores = manager.find_most_similar(
                query_embedding.unsqueeze(0), 
                corpus_embeddings, 
                top_k=5
            )
            
            # Build results
            top_results = []
            for rank, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
                chunk_text = chunks[idx]['text']
                top_results.append({
                    'chunk_id': chunk_ids[idx],
                    'score': float(score),
                    'rank': rank,
                    'text_preview': chunk_text[:150] + "..." if len(chunk_text) > 150 else chunk_text
                })
            
            # Find rank of correct answer
            found_rank = None
            for result in top_results:
                if result['chunk_id'] == correct_chunk_id:
                    found_rank = result['rank']
                    break
            
            test_case_results.append({
                'question': question,
                'correct_chunk_id': correct_chunk_id,
                'found_rank': found_rank,
                'top_5_results': top_results
            })
            
            # Progress update
            if (i + 1) % 5 == 0 or (i + 1) == len(test_suite):
                print(f"  ✓ Đã xử lý {i + 1}/{len(test_suite)} câu hỏi")
        
        # Calculate metrics
        print("\n📊 Đang tính toán metrics...")
        calculator = MetricsCalculator()
        summary_metrics, detailed_df = calculator.calculate_detailed_metrics(test_case_results)
        performance_analysis = calculator.get_performance_analysis(detailed_df)
        
        # Memory info
        memory_info = manager.get_memory_usage()
        
        evaluation_time = time.time() - start_time
        
        # Print summary
        print(f"\n📈 Kết quả đánh giá:")
        print(f"   ⏱️  Thời gian: {evaluation_time:.2f}s")
        print(f"   🎯 MRR: {summary_metrics['MRR']:.4f}")
        print(f"   🥇 Hit Rate@1: {summary_metrics['Hit_Rate@1']:.2%}")
        print(f"   🥉 Hit Rate@3: {summary_metrics['Hit_Rate@3']:.2%}")
        print(f"   🏅 Hit Rate@5: {summary_metrics['Hit_Rate@5']:.2%}")
        
        # Clean up GPU memory
        manager.clear_cache()
        del manager
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'model_name': model_name,
            'model_info': model_info,
            'embedding_dimension': model_info['dimension'],
            'evaluation_time_seconds': evaluation_time,
            'summary_metrics': summary_metrics,
            'detailed_metrics': detailed_df.to_dict('records'),
            'performance_analysis': performance_analysis,
            'test_cases': test_case_results,
            'memory_usage': memory_info,
            'timestamp': datetime.now().isoformat(),
            'total_questions': len(test_suite),
            'total_chunks': len(chunks)
        }
        
    except Exception as e:
        print(f"❌ Lỗi khi đánh giá model {model_name}: {e}")
        import traceback
        print("🔍 Chi tiết lỗi:")
        traceback.print_exc()
        return None

def save_model_report(model_result: dict, reports_dir: Path) -> Path:
    """Lưu báo cáo chi tiết cho một model"""
    if not model_result:
        return None
    
    model_name = model_result['model_name']
    safe_name = model_name.replace('/', '_').replace(':', '_').replace('-', '_')
    
    # Tạo thư mục riêng cho model
    model_dir = reports_dir / "individual_models" / safe_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed JSON report
    json_path = model_dir / f"{safe_name}_full_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(model_result, f, ensure_ascii=False, indent=2)
    
    # Save summary report
    summary_data = {
        'Model': model_name,
        'Dimension': model_result['embedding_dimension'],
        'Evaluation_Time_Seconds': model_result['evaluation_time_seconds'],
        'Total_Questions': model_result['total_questions'],
        'Total_Chunks': model_result['total_chunks'],
        **model_result['summary_metrics'],
        'Timestamp': model_result['timestamp']
    }
    
    summary_path = model_dir / f"{safe_name}_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    # Save detailed results as CSV
    detailed_df = pd.DataFrame(model_result['detailed_metrics'])
    csv_path = model_dir / f"{safe_name}_detailed_results.csv"
    detailed_df.to_csv(csv_path, index=False, encoding='utf-8')
    
    print(f"💾 Đã lưu báo cáo model tại: {model_dir}")
    
    return json_path

def create_overall_summary(all_results: dict, reports_dir: Path):
    """Tạo báo cáo tổng hợp tất cả models"""
    print("\n📋 Đang tạo báo cáo tổng hợp...")
    
    # Prepare summary data
    summary_data = []
    for model_name, result in all_results.items():
        if result is not None:
            row = {
                'Model': model_name,
                'Short_Name': model_name.split('/')[-1],
                'Dimension': result['embedding_dimension'],
                'Time_Seconds': result['evaluation_time_seconds'],
                'Questions': result['total_questions'],
                'Chunks': result['total_chunks'],
                **result['summary_metrics']
            }
            summary_data.append(row)
    
    if not summary_data:
        print("❌ Không có kết quả nào để tạo báo cáo tổng hợp")
        return
    
    # Create DataFrame and sort by MRR
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values('MRR', ascending=False).reset_index(drop=True)
    df_summary['Rank'] = df_summary.index + 1
    
    # Save CSV
    summary_csv_path = reports_dir / "overall_summary.csv"
    df_summary.to_csv(summary_csv_path, index=False, encoding='utf-8')
    
    # Save JSON
    summary_json_path = reports_dir / "overall_summary.json"
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(df_summary.to_dict('records'), f, ensure_ascii=False, indent=2)
    
    # Print summary table
    print("\n🏆 BXH MODELS (Sắp xếp theo MRR):")
    print("=" * 100)
    print(f"{'Rank':<4} {'Model':<40} {'MRR':<8} {'Hit@1':<8} {'Hit@5':<8} {'Time':<8}")
    print("=" * 100)
    
    for _, row in df_summary.iterrows():
        print(f"{row['Rank']:<4} {row['Short_Name']:<40} "
              f"{row['MRR']:<8.4f} {row['Hit_Rate@1']:<8.2%} "
              f"{row['Hit_Rate@5']:<8.2%} {row['Time_Seconds']:<8.1f}")
    
    print("=" * 100)
    print(f"💾 Đã lưu báo cáo tổng hợp: {summary_csv_path}")
    
    return df_summary

def create_visualizations(all_results: dict, visualizer: ResultVisualizer):
    """Tạo tất cả các visualizations"""
    print("\n📈 Đang tạo visualizations...")
    
    try:
        # Model comparison chart
        calculator = MetricsCalculator()
        model_metrics = {name: result['summary_metrics'] 
                        for name, result in all_results.items() 
                        if result is not None}
        
        if len(model_metrics) > 1:
            df_comparison = calculator.compare_models(model_metrics)
            
            # Save comparison plot
            comparison_path = visualizer.output_dir / "visualizations" / "model_comparison.png"
            visualizer.plot_model_comparison(df_comparison, str(comparison_path))
            
            # Create performance heatmap
            heatmap_path = visualizer.output_dir / "visualizations" / "performance_heatmap.png"
            visualizer.create_performance_heatmap(all_results, str(heatmap_path))
            
            # Save comparison data
            comparison_csv = visualizer.output_dir / "model_comparison.csv"
            df_comparison.to_csv(comparison_csv, index=False)
            print(f"💾 Đã lưu bảng so sánh: {comparison_csv}")
        
        # Individual model analysis
        for model_name, result in all_results.items():
            if result is not None:
                detailed_df = pd.DataFrame(result['detailed_metrics'])
                safe_name = model_name.replace('/', '_').replace(':', '_')
                
                detail_path = visualizer.output_dir / "visualizations" / f"{safe_name}_analysis.png"
                visualizer.plot_detailed_analysis(
                    model_name, detailed_df, result['summary_metrics'], str(detail_path)
                )
        
        # Generate HTML report
        visualizer.generate_performance_report(all_results)
        
        print("✅ Đã tạo xong tất cả visualizations")
        
    except Exception as e:
        print(f"❌ Lỗi khi tạo visualizations: {e}")

def main():
    """Main evaluation pipeline"""
    print("=" * 70)
    print("🇻🇳 VIETNAMESE EMBEDDING MODELS EVALUATION PIPELINE")
    print("=" * 70)
    
    # Setup
    setup_directories()
    models_to_test, test_suite = load_configuration()
    
    if not models_to_test or not test_suite:
        print("❌ Không thể load cấu hình. Dừng chương trình.")
        return
    
    # Process data
    print("\n📚 Đang xử lý dữ liệu...")
    processor = DataProcessor()
    chunks = processor.load_and_process_content('data/content.md')
    
    if not chunks:
        print("❌ Không thể load content data. Dừng chương trình.")
        return
    
    chunk_texts = [chunk['text'] for chunk in chunks]
    chunk_ids = [chunk['id'] for chunk in chunks]
    
    # Save chunks info for debugging
    chunks_info_path = Path("reports") / "chunks_info.json"
    processor.save_chunks_info(chunks, chunks_info_path)
    
    # Initialize components
    reports_dir = Path("reports")
    visualizer = ResultVisualizer(str(reports_dir))
    all_results = {}
    
    # Evaluate each model
    print(f"\n🔄 Bắt đầu đánh giá {len(models_to_test)} models...")
    
    for i, model_name in enumerate(models_to_test, 1):
        print(f"\n{'='*50}")
        print(f"📊 Model {i}/{len(models_to_test)}: {model_name}")
        print(f"{'='*50}")
        
        try:
            # Evaluate model
            result = evaluate_single_model(
                model_name, chunks, test_suite, chunk_texts, chunk_ids, processor
            )
            
            if result:
                all_results[model_name] = result
                save_model_report(result, reports_dir)
            
        except KeyboardInterrupt:
            print("\n⚠️  Nhận được tín hiệu dừng. Đang lưu kết quả hiện tại...")
            break
        except Exception as e:
            print(f"❌ Lỗi không mong muốn với model {model_name}: {e}")
            continue
    
    # Create visualizations and final reports
    if all_results:
        create_visualizations(all_results, visualizer)
        
        # Save consolidated results
        consolidated_path = reports_dir / "all_results_consolidated.json"
        with open(consolidated_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Đã lưu kết quả tổng hợp: {consolidated_path}")
        
        # Print final summary
        print("\n" + "="*70)
        print("📋 TỔNG KẾT KỀT QUẢ ĐÁNH GIÁ")
        print("="*70)
        
        # Sort by MRR
        sorted_results = sorted(
            [(name, result['summary_metrics']['MRR']) for name, result in all_results.items()],
            key=lambda x: x[1], reverse=True
        )
        
        print("🏆 BẢNG XẾP HẠNG (theo MRR):")
        for rank, (model_name, mrr) in enumerate(sorted_results, 1):
            result = all_results[model_name]
            hit_1 = result['summary_metrics']['Hit_Rate@1']
            time_taken = result['evaluation_time_seconds']
            
            print(f"{rank:2d}. {model_name.split('/')[-1]:<40} "
                  f"MRR: {mrr:.4f} | Hit@1: {hit_1:.2%} | Time: {time_taken:.1f}s")
        
        best_model = sorted_results[0][0]
        best_metrics = all_results[best_model]['summary_metrics']
        
        print(f"\n🥇 CHAMPION: {best_model}")
        print(f"   📊 MRR: {best_metrics['MRR']:.4f}")
        print(f"   🎯 Hit Rate@1: {best_metrics['Hit_Rate@1']:.2%}")
        print(f"   📈 Hit Rate@5: {best_metrics['Hit_Rate@5']:.2%}")
        
        print(f"\n📁 Tất cả báo cáo đã được lưu trong thư mục: {reports_dir.absolute()}")
        print("📊 Kiểm tra file 'performance_report.html' để xem báo cáo chi tiết")
        
    else:
        print("❌ Không có kết quả nào được tạo ra. Vui lòng kiểm tra cấu hình và dữ liệu.")
    
    print("\n🎉 Hoàn thành pipeline đánh giá!")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Chương trình đã bị dừng bởi người dùng.")
    except Exception as e:
        print(f"\n❌ Lỗi không mong muốn: {e}")
        import traceback
        traceback.print_exc()