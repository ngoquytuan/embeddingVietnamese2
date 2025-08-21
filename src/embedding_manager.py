import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple, Dict
import time

class EmbeddingManager:
    def __init__(self, model_name: str, cache_dir: str = "./model_cache"):
        """
        Khởi tạo Embedding Manager với tối ưu GPU
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Kiểm tra GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            print(f"Sử dụng GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM khả dụng: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("Sử dụng CPU")
        
        # Load model
        print(f"Đang tải model: {model_name}...")
        start_time = time.time()
        
        try:
            self.model = SentenceTransformer(
                model_name, 
                device=self.device,
                cache_folder=cache_dir
            )
            load_time = time.time() - start_time
            print(f"Đã tải model thành công trong {load_time:.2f}s")
            print(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            print(f"Lỗi khi tải model {model_name}: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """Lấy thông tin model"""
        return {
            'name': self.model_name,
            'dimension': self.model.get_sentence_embedding_dimension(),
            'device': self.device,
            'max_sequence_length': getattr(self.model, 'max_seq_length', 'Unknown')
        }
    
    def encode_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> torch.Tensor:
        """
        Encode danh sách texts thành embeddings với tối ưu GPU
        """
        if not texts:
            return torch.empty(0, self.model.get_sentence_embedding_dimension())
        
        print(f"Đang encode {len(texts)} texts...")
        start_time = time.time()
        
        try:
            # Tối ưu batch size dựa trên VRAM
            if self.device == "cuda":
                available_memory = torch.cuda.get_device_properties(0).total_memory
                if available_memory < 4e9:  # < 4GB
                    batch_size = min(16, batch_size)
                elif available_memory < 8e9:  # < 8GB  
                    batch_size = min(32, batch_size)
                else:  # >= 8GB
                    batch_size = min(64, batch_size)
            
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_tensor=True,
                device=self.device,
                normalize_embeddings=True  # Normalize để tối ưu cosine similarity
            )
            
            encode_time = time.time() - start_time
            print(f"Hoàn thành encoding trong {encode_time:.2f}s")
            print(f"Shape: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            print(f"Lỗi khi encode: {e}")
            raise
    
    def find_most_similar(self, query_embedding: torch.Tensor, 
                         corpus_embeddings: torch.Tensor, 
                         top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tìm top-k embeddings tương đồng nhất với query
        """
        try:
            # Tính cosine similarity
            similarities = util.cos_sim(query_embedding, corpus_embeddings)
            
            # Lấy top-k results
            top_k = min(top_k, corpus_embeddings.shape[0])
            top_results = torch.topk(similarities, k=top_k, dim=-1)
            
            indices = top_results.indices.cpu().numpy().flatten()
            scores = top_results.values.cpu().numpy().flatten()
            
            return indices, scores
            
        except Exception as e:
            print(f"Lỗi khi tìm kiếm tương đồng: {e}")
            raise
    
    def batch_search(self, queries: List[str], corpus_embeddings: torch.Tensor, 
                    chunk_ids: List[str], top_k: int = 5) -> List[Dict]:
        """
        Thực hiện batch search cho nhiều queries
        """
        print(f"Đang thực hiện batch search cho {len(queries)} queries...")
        
        # Encode tất cả queries
        query_embeddings = self.encode_texts(queries, show_progress=False)
        
        results = []
        for i, (query, query_emb) in enumerate(zip(queries, query_embeddings)):
            indices, scores = self.find_most_similar(
                query_emb.unsqueeze(0), 
                corpus_embeddings, 
                top_k
            )
            
            top_chunks = []
            for idx, score in zip(indices, scores):
                top_chunks.append({
                    'chunk_id': chunk_ids[idx],
                    'score': float(score),
                    'rank': len(top_chunks) + 1
                })
            
            results.append({
                'query': query,
                'top_results': top_chunks
            })
        
        return results
    
    def clear_cache(self):
        """Dọn dẹp cache GPU"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            print("Đã dọn dẹp GPU cache")
    
    def get_memory_usage(self) -> Dict:
        """Lấy thông tin sử dụng memory"""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            return {
                'allocated_gb': allocated,
                'cached_gb': cached,
                'device': torch.cuda.get_device_name(0)
            }
        else:
            return {'device': 'CPU', 'allocated_gb': 0, 'cached_gb': 0}