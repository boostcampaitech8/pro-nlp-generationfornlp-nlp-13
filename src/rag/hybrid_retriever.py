import pickle
import os
import pandas as pd
import faiss
import numpy as np
from typing import List, Optional, Dict, Tuple, Union
from tqdm.auto import tqdm
from FlagEmbedding import BGEM3FlagModel
from collections import defaultdict
import glob

class HybridRetriever:
    """
    BGE-M3 기반 하이브리드 리트리버
    - Dense: FAISS 인덱스
    - Sparse: BM25 pickle
    - Fusion: RRF 또는 Weighted
    """
    
    def __init__(
        self,
        model_name: str = 'BAAI/bge-m3',
        use_fp16: bool = True,
        data_path: str = '../Jang',
        fusion_method: str = 'rrf',
        rrf_k: int = 60,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            model_name: BGE-M3 모델 이름
            use_fp16: FP16 사용 여부
            data_path: 데이터 파일들이 있는 경로
            fusion_method: 'rrf' 또는 'weighted'
            rrf_k: RRF 상수 (보통 60)
            weights: weighted fusion 시 가중치 {'dense': 0.5, 'sparse': 0.5}
        """
        print("Initializing BGE-M3 Model...")
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.data_path = data_path
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        
        if weights is None:
            self.weights = {'dense': 0.5, 'sparse': 0.5}
        else:
            self.weights = weights
        
        self.dense_index: Optional[faiss.Index] = None
        self.sparse_vecs: Optional[List[Dict]] = None
        self.meta_df: Optional[pd.DataFrame] = None
        
        self._load_files()
    
    def _load_files(self):
        index_path = os.path.join(self.data_path, 'wikipedia_bge_m3.index')
        if os.path.exists(index_path):
            print(f"Loading Dense Index: {index_path}")
            self.dense_index = faiss.read_index(index_path)
            print(f"  → Loaded {self.dense_index.ntotal} vectors")
        else:
            raise FileNotFoundError(f"Dense index not found: {index_path}")
        
        meta_path = os.path.join(self.data_path, 'wikipedia_chunks_meta.parquet')
        if os.path.exists(meta_path):
            print(f"Loading Metadata: {meta_path}")
            self.meta_df = pd.read_parquet(meta_path)
            print(f"  → Loaded {len(self.meta_df)} documents")
        else:
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        
        sparse_parts_dir = os.path.join(self.data_path, 'wikipedia_sparse_parts')
        file_pattern = os.path.join(sparse_parts_dir, "wiki_sparse_part_*.pkl")
        part_files = sorted(
            glob.glob(file_pattern),
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        
        if part_files:
            print(f"Loading Sparse Vectors (Parts): {sparse_parts_dir}")
            print(f"  → 발견된 파일: {len(part_files)}개")
            
            self.sparse_vecs = []
            for file_path in part_files:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    self.sparse_vecs.extend(data)
            
            print(f"  → Loaded {len(self.sparse_vecs)} sparse vectors")
        
        # 검증
        if len(self.meta_df) != len(self.sparse_vecs):
            print(f"[WARNING] Metadata({len(self.meta_df)}) and Sparse({len(self.sparse_vecs)}) count mismatch!")
        if self.dense_index.ntotal != len(self.meta_df):
            print(f"[WARNING] Dense Index({self.dense_index.ntotal}) and Metadata({len(self.meta_df)}) count mismatch!")
    
    def _compute_sparse_score(self, query_weights: Dict, doc_weights: Dict) -> float:
        """Sparse 점수 계산"""
        score = 0.0
        for token, q_weight in query_weights.items():
            if token in doc_weights:
                score += q_weight * doc_weights[token]
        return score
    
    def retrieve_dense(
        self, 
        query: str, 
        top_k: int = 5,
        only_scores: bool = False
    ) -> Union[pd.DataFrame, List[Tuple[int, float]]]:
        """
        Dense 검색만 수행 (외부 호출 가능)
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수
            return_scores: True면 DataFrame 반환, False면 (idx, score) 리스트 반환
        
        Returns:
            return_scores=True: 검색 결과 DataFrame
            return_scores=False: [(idx, score), ...] 리스트
        """

        q_vec = self.model.encode(
            [query],
            batch_size=1,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )['dense_vecs']
        
        q_vec = q_vec.astype('float32')
        faiss.normalize_L2(q_vec)
        scores, indices = self.dense_index.search(q_vec, top_k)
        
        # (index, score) 튜플 리스트 생성
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:  
                results.append((int(idx), float(score)))
        
        # 스코어만 반환
        if only_scores:
            return results
        
        # DataFrame으로 변환하여 반환
        df_results = []
        for rank, (idx, score) in enumerate(results, start=1):
            if idx < len(self.meta_df):
                doc = self.meta_df.iloc[idx]
                df_results.append({
                    'rank': rank,
                    'score': score,
                    'doc_id': doc.get('doc_id', idx),
                    'title': doc.get('title', ''),
                    'text': doc.get('text', '')
                })
        
        return pd.DataFrame(df_results)
    
    def retrieve_sparse(
        self, 
        query: str, 
        top_k: int = 5,
        only_scores: bool = False
    ) -> Union[pd.DataFrame, List[Tuple[int, float]]]:
        """
        Sparse 검색만 수행 (외부 호출 가능)
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수
            return_scores: True면 DataFrame 반환, False면 (idx, score) 리스트 반환
        
        Returns:
            return_scores=True: 검색 결과 DataFrame
            return_scores=False: [(idx, score), ...] 리스트
        """
        query_output = self.model.encode(
            [query],
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False
        )
        query_sparse = query_output['lexical_weights'][0]
        
        all_scores = [
            self._compute_sparse_score(query_sparse, doc_vec)
            for doc_vec in self.sparse_vecs
        ]
        
        top_indices = np.argsort(all_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((int(idx), float(all_scores[idx])))
        
        # 스코어만 반환
        if only_scores:
            return results
        
        # DataFrame으로 변환하여 반환
        df_results = []
        for rank, (idx, score) in enumerate(results, start=1):
            if idx < len(self.meta_df):
                doc = self.meta_df.iloc[idx]
                df_results.append({
                    'rank': rank,
                    'score': score,
                    'doc_id': doc.get('doc_id', idx),
                    'title': doc.get('title', ''),
                    'text': doc.get('text', '')
                })
        
        return pd.DataFrame(df_results)
    
    def _retrieve_dense(self, query: str, k: int) -> List[Tuple[int, float]]:
        """내부용 Dense 검색 (return_scores=False와 동일)"""
        return self.retrieve_dense(query, top_k=k, return_scores=False)
    
    def _retrieve_sparse(self, query: str, k: int) -> List[Tuple[int, float]]:
        """내부용 Sparse 검색 (return_scores=False와 동일)"""
        return self.retrieve_sparse(query, top_k=k, return_scores=False)
    
    def _rrf_fusion(
        self,
        dense_results: List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """RRF(Reciprocal Rank Fusion) 적용"""
        rrf_scores: Dict[int, float] = defaultdict(float)
        
        # Dense 순위 반영
        for rank, (idx, _) in enumerate(dense_results, start=1):
            rrf_scores[idx] += 1.0 / (self.rrf_k + rank)
        
        # Sparse 순위 반영
        for rank, (idx, _) in enumerate(sparse_results, start=1):
            rrf_scores[idx] += 1.0 / (self.rrf_k + rank)
        
        # 점수 기준 정렬
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in sorted_items]
    
    def _weighted_fusion(
        self,
        dense_results: List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """가중치 기반 점수 결합"""
        def normalize(results):
            if not results:
                return {}
            scores = [s for _, s in results]
            min_s, max_s = min(scores), max(scores)
            if max_s - min_s == 0:
                return {idx: 1.0 for idx, _ in results}
            return {idx: (s - min_s) / (max_s - min_s) for idx, s in results}
        
        dense_norm = normalize(dense_results)
        sparse_norm = normalize(sparse_results)
        
        # 가중치 적용
        final_scores: Dict[int, float] = defaultdict(float)
        for idx, score in dense_norm.items():
            final_scores[idx] += self.weights['dense'] * score
        for idx, score in sparse_norm.items():
            final_scores[idx] += self.weights['sparse'] * score
        
        # 정렬
        sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in sorted_items]

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        dense_k: int = 100,
        sparse_k: int = 100
    ) -> pd.DataFrame:
        """
        하이브리드 검색 (Dense + Sparse + Fusion)
        
        Args:
            query: 검색 쿼리
            top_k: 최종 반환 문서 수
            dense_k: Dense에서 가져올 후보 수
            sparse_k: Sparse에서 가져올 후보 수
        
        Returns:
            검색 결과 DataFrame
        """
        dense_results = self._retrieve_dense(query, dense_k)
        sparse_results = self._retrieve_sparse(query, sparse_k)
        
        if self.fusion_method == 'rrf':
            fused_results = self._rrf_fusion(dense_results, sparse_results)
        elif self.fusion_method == 'weighted':
            fused_results = self._weighted_fusion(dense_results, sparse_results)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Top-K 추출 및 메타데이터 결합
        results = []
        for rank, (idx, score) in enumerate(fused_results[:top_k], start=1):
            if idx < len(self.meta_df):
                doc = self.meta_df.iloc[idx]
                results.append({
                    'rank': rank,
                    'score': score,
                    'doc_id': doc.get('doc_id', idx),
                    'title': doc.get('title', ''),
                    'text': doc.get('text', '')
                })
        
        return pd.DataFrame(results)
    
    def retrieve_single(self, *args, **kwargs) -> pd.DataFrame:
        """retrieve_hybrid의 별칭 (하위 호환성)"""
        return self.retrieve_hybrid(*args, **kwargs)
    
    def retrieve_batch(
        self,
        question_df: pd.DataFrame,
        paragraph_col: str = 'paragraph',
        question_col: str = 'question',
        n: Optional[int] = None,
        top_k: int = 5,
        dense_k: int = 100,
        sparse_k: int = 100,
        seed: int = 42,
        mode: str = 'hybrid'  
    ) -> pd.DataFrame:
        """
        DataFrame의 모든 질문에 대해 검색 수행
        
        Args:
            question_df: 질문 DataFrame
            paragraph_col: paragraph 컬럼명
            question_col: question 컬럼명
            n: 샘플링 개수 (None이면 전체)
            top_k: 각 질문당 반환할 문서 수
            dense_k: Dense 후보 수 (hybrid 모드에만 사용)
            sparse_k: Sparse 후보 수 (hybrid 모드에만 사용)
            seed: 랜덤 시드
            mode: 'hybrid', 'dense', 'sparse' 중 선택
        
        Returns:
            확장된 DataFrame (각 질문당 top_k개 행 생성)
        """
        if n is None:
            samples = question_df.copy()
        else:
            real_n = min(n, len(question_df))
            samples = question_df.sample(n=real_n, random_state=seed).copy()
        
        print(f"\n{len(samples)}개의 질문에 대해 Top-{top_k} {mode.upper()} 검색을 수행합니다.")
        print(f"예상 결과 행: {len(samples) * top_k}개\n")
        
        merged_results = []
        
        for idx, row in tqdm(samples.iterrows(), total=len(samples), desc="Searching"):
            query = f"{row[paragraph_col]} \n\n {row[question_col]}"
            
            if mode == 'dense':
                results = self.retrieve_dense(query, top_k=top_k, only_scores=True)
            elif mode == 'sparse':
                results = self.retrieve_sparse(query, top_k=top_k, only_scores=True)
            elif mode == 'hybrid':
                dense_results = self._retrieve_dense(query, dense_k)
                sparse_results = self._retrieve_sparse(query, sparse_k)
                
                if self.fusion_method == 'rrf':
                    fused_results = self._rrf_fusion(dense_results, sparse_results)
                elif self.fusion_method == 'weighted':
                    fused_results = self._weighted_fusion(dense_results, sparse_results)
                else:
                    raise ValueError(f"Unknown fusion method: {self.fusion_method}")
                
                results = fused_results[:top_k]
            else:
                raise ValueError(f"Unknown mode: {mode}. Use 'hybrid', 'dense', or 'sparse'")
            
            # Top-K 결과 처리
            for rank, (doc_idx, score) in enumerate(results, start=1):
                if doc_idx >= len(self.meta_df):
                    continue
                
                combined_row = row.to_dict()
                
                doc = self.meta_df.iloc[doc_idx]
                combined_row['ctx_rank'] = rank
                combined_row['ctx_score'] = score
                combined_row['ctx_id'] = doc.get('doc_id', doc_idx)
                combined_row['ctx_title'] = doc.get('title', '')
                combined_row['ctx_text'] = doc.get('text', '')
                
                merged_results.append(combined_row)
        
        final_df = pd.DataFrame(merged_results)
        print(f"\n완료! 총 {len(final_df)}개의 행이 생성되었습니다.")
        
        return final_df
    