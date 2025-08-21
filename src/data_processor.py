import os
import re
import json
from pathlib import Path
from pyvi import ViTokenizer
from typing import List, Dict
import unicodedata

class DataProcessor:
    def __init__(self):
        self.chunk_size = 150  # Số từ tối đa trong một chunk
        self.overlap = 30      # Số từ overlap giữa các chunk
    
    def clean_text(self, text: str) -> str:
        """Làm sạch văn bản tiếng Việt"""
        # Normalize unicode
        text = unicodedata.normalize('NFC', text)
        
        # Xóa markdown headers
        text = re.sub(r'#+\s*', '', text)
        
        # Xóa các ký tự đặc biệt không cần thiết
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Xóa các ký tự không phải chữ cái, số, dấu câu tiếng Việt
        text = re.sub(r'[^\w\s\.,!?;:()\-"\'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]', ' ', text)
        
        return text.strip()
    
    def simple_sentence_split(self, text: str) -> List[str]:
        """Tách câu đơn giản cho tiếng Việt"""
        # Các dấu kết thúc câu
        sentence_endings = r'[.!?]\s+'
        
        # Tách câu
        sentences = re.split(sentence_endings, text)
        
        # Loại bỏ câu rỗng và câu quá ngắn
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        return sentences
    
    def segment_vietnamese_text(self, text: str) -> List[str]:
        """Tách từ tiếng Việt sử dụng pyvi"""
        try:
            # Tách câu trước
            sentences = self.simple_sentence_split(text)
            
            # Tách từ cho mỗi câu
            segmented_sentences = []
            for sentence in sentences:
                if sentence.strip():
                    segmented = ViTokenizer.tokenize(sentence.strip())
                    segmented_sentences.append(segmented)
            
            return segmented_sentences
        except Exception as e:
            print(f"Lỗi khi tách từ với pyvi: {e}")
            # Fallback: chỉ tách câu đơn giản
            return self.simple_sentence_split(text)
    
    def create_chunks_by_sentences(self, text: str) -> List[Dict[str, str]]:
        """Chia văn bản thành chunks theo câu"""
        cleaned_text = self.clean_text(text)
        sentences = self.segment_vietnamese_text(cleaned_text)
        
        chunks = []
        current_chunk_sentences = []
        current_word_count = 0
        chunk_counter = 0
        
        for sentence in sentences:
            words = sentence.split()
            sentence_word_count = len(words)
            
            # Nếu thêm câu này vào chunk hiện tại vượt quá giới hạn
            if current_word_count + sentence_word_count > self.chunk_size:
                if current_chunk_sentences:
                    # Tạo chunk từ các câu hiện tại
                    chunk_text = ' '.join(current_chunk_sentences)
                    chunks.append({
                        'id': f'chunk_{chunk_counter}',
                        'text': chunk_text,
                        'word_count': current_word_count,
                        'sentence_count': len(current_chunk_sentences)
                    })
                    chunk_counter += 1
                    
                    # Giữ lại một vài câu cuối để tạo overlap
                    overlap_sentences = current_chunk_sentences[-2:] if len(current_chunk_sentences) > 2 else current_chunk_sentences
                    current_chunk_sentences = overlap_sentences + [sentence]
                    current_word_count = sum(len(s.split()) for s in current_chunk_sentences)
                else:
                    current_chunk_sentences = [sentence]
                    current_word_count = sentence_word_count
            else:
                current_chunk_sentences.append(sentence)
                current_word_count += sentence_word_count
        
        # Thêm chunk cuối cùng nếu có
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append({
                'id': f'chunk_{chunk_counter}',
                'text': chunk_text,
                'word_count': current_word_count,
                'sentence_count': len(current_chunk_sentences)
            })
        
        return chunks
    
    def create_chunks_by_words(self, text: str) -> List[Dict[str, str]]:
        """Chia văn bản thành chunks theo từ (sliding window)"""
        cleaned_text = self.clean_text(text)
        
        # Tách từ cho toàn bộ văn bản
        try:
            segmented_text = ViTokenizer.tokenize(cleaned_text)
            words = segmented_text.split()
        except Exception as e:
            print(f"Lỗi khi tách từ: {e}. Sử dụng tách từ đơn giản.")
            words = cleaned_text.split()
        
        chunks = []
        chunk_counter = 0
        
        # Tạo chunks với sliding window
        start_idx = 0
        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))
            
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'id': f'chunk_{chunk_counter}',
                'text': chunk_text,
                'word_count': len(chunk_words),
                'start_word_idx': start_idx,
                'end_word_idx': end_idx
            })
            
            chunk_counter += 1
            
            # Di chuyển window với overlap
            start_idx += (self.chunk_size - self.overlap)
            
            # Nếu chunk tiếp theo sẽ quá ngắn, dừng lại
            if len(words) - start_idx < self.overlap:
                break
        
        return chunks
    
    def create_chunks(self, text: str, method: str = 'sentences') -> List[Dict[str, str]]:
        """Chia văn bản thành chunks với phương pháp được chọn"""
        if method == 'sentences':
            return self.create_chunks_by_sentences(text)
        elif method == 'words':
            return self.create_chunks_by_words(text)
        else:
            raise ValueError(f"Phương pháp không hỗ trợ: {method}")
    
    def load_and_process_content(self, file_path: str, chunk_method: str = 'sentences') -> List[Dict[str, str]]:
        """Load và xử lý file content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"Đã đọc file {file_path} ({len(content)} ký tự)")
            
            chunks = self.create_chunks(content, method=chunk_method)
            print(f"Đã tạo {len(chunks)} chunks bằng phương pháp '{chunk_method}'")
            
            # In thống kê
            if chunks:
                word_counts = [chunk['word_count'] for chunk in chunks]
                print(f"Số từ trung bình mỗi chunk: {sum(word_counts) / len(word_counts):.1f}")
                print(f"Chunk ngắn nhất: {min(word_counts)} từ")
                print(f"Chunk dài nhất: {max(word_counts)} từ")
                
                # In một vài chunk mẫu
                print("\n--- Chunk mẫu ---")
                for i in range(min(3, len(chunks))):
                    print(f"Chunk {i}: {chunks[i]['text'][:100]}...")
            
            return chunks
            
        except FileNotFoundError:
            print(f"❌ Không tìm thấy file: {file_path}")
            return []
        except Exception as e:
            print(f"❌ Lỗi khi xử lý file: {e}")
            return []
    
    def load_test_suite(self, file_path: str) -> List[Dict]:
        """Load bộ test questions"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                test_suite = json.load(f)
            print(f"✅ Đã load {len(test_suite)} câu hỏi test")
            
            # Kiểm tra format
            for i, test_case in enumerate(test_suite):
                if 'question' not in test_case or 'correct_chunk_id' not in test_case:
                    print(f"⚠️  Test case {i} thiếu trường bắt buộc")
            
            return test_suite
        except Exception as e:
            print(f"❌ Lỗi khi load test suite: {e}")
            return []
    
    def validate_test_suite(self, test_suite: List[Dict], chunk_ids: List[str]) -> bool:
        """Kiểm tra tính hợp lệ của test suite"""
        print("🔍 Đang kiểm tra tính hợp lệ của test suite...")
        
        valid = True
        missing_chunks = []
        
        for i, test_case in enumerate(test_suite):
            correct_id = test_case.get('correct_chunk_id')
            if correct_id not in chunk_ids:
                missing_chunks.append((i, correct_id))
                valid = False
        
        if missing_chunks:
            print(f"⚠️  Tìm thấy {len(missing_chunks)} test cases có chunk_id không tồn tại:")
            for i, chunk_id in missing_chunks[:5]:  # Hiển thị 5 cái đầu
                print(f"   Test case {i}: {chunk_id}")
            if len(missing_chunks) > 5:
                print(f"   ... và {len(missing_chunks) - 5} cases khác")
        else:
            print("✅ Test suite hợp lệ")
        
        return valid
    
    def save_chunks_info(self, chunks: List[Dict], output_path: str):
        """Lưu thông tin chunks để debug"""
        chunks_info = {
            'total_chunks': len(chunks),
            'avg_word_count': sum(c['word_count'] for c in chunks) / len(chunks) if chunks else 0,
            'chunks': chunks
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_info, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Đã lưu thông tin chunks vào: {output_path}")
    
    def create_balanced_test_suite(self, chunks: List[Dict], num_questions: int = 20) -> List[Dict]:
        """Tạo test suite cân bằng từ chunks (để testing)"""
        if len(chunks) < num_questions:
            print(f"⚠️  Chỉ có {len(chunks)} chunks, không thể tạo {num_questions} câu hỏi")
            num_questions = len(chunks)
        
        # Chọn chunks đại diện
        step = max(1, len(chunks) // num_questions)
        selected_chunks = chunks[::step][:num_questions]
        
        test_suite = []
        for i, chunk in enumerate(selected_chunks):
            # Tạo câu hỏi đơn giản từ chunk
            text = chunk['text']
            sentences = self.simple_sentence_split(text)
            
            if sentences:
                # Lấy câu đầu tiên làm base cho câu hỏi
                first_sentence = sentences[0]
                question = f"Nội dung nào nói về {first_sentence[:50]}...?"
                
                test_suite.append({
                    'question': question,
                    'correct_chunk_id': chunk['id']
                })
        
        return test_suite