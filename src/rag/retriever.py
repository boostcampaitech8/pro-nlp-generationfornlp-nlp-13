import os
import pickle
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Union, Callable
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import faiss
from FlagEmbedding import BGEM3FlagModel

class HybridRetriever:

    def __init__(self, model_name):
        # 공통
        self.model = BGEM3FlagModel(model_name,  use_fp16=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # self.current_dir = os.path.dirname(os.path.abspath(__file__))
        # sparse 관련

        self.bm25 = None
        # pickle_name = "wikipedia_sparse_vecs.pkl"
        # bm_path = os.path.join(self.current_dir, pickle_name)

        # if os.path.isfile(bm_path):
        #     print("피클 찾음!")
        #     with open(bm_path, "rb") as file:
        #         self.bm25 = pickle.load(file)
        #         self.set_data()
        #     print(f"Sparse Embedding (BM25) loaded: {pickle_name}")     

        # dense 관련


    def get_sparse_embedding(self) -> None:
        pickle_name = "wikipedia_sparse_vecs.pkl"

        bm_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(bm_path):
            with open(bm_path, "rb") as file:
                self.bm25 = pickle.load(file)
                self.set_data()
            print(f"Sparse Embedding (BM25) loaded: {pickle_name}")     
        else:
            print("피클 파일 빌드부터 하세요.")

        
    def _compute_sparse_score(q_dict, d_dict):
        """두 딕셔너리 간의 가중치 곱 합산"""
        score = 0
        for token, weight in q_dict.items():
            if token in d_dict:
                score += weight * d_dict[token]
        return score
    
    
    def retrieve_sparse(self, query_text, k=5, topk: Optional[int] = 5, n_jobs: Optional[int] = None):
        # 질문에서 sparse 가중치 추출
        query_output = self.model.encode([query_text], return_dense=False, return_sparse=True)
        query_sparse = query_output['lexical_weights'][0]

        # 전체 sparse_vecs에 대해 점수 계산 (List Comprehension으로 속도 확보)
        all_scores = [self._compute_sparse_score(query_sparse, d_vec) for d_vec in self.bm25]
        
        # 상위 k개 인덱스 추출
        top_indices = np.argsort(all_scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append({
                'index': idx,
                'score': all_scores[idx],
                'title': self.df_meta.iloc[idx]['title'],
                'text': self.df_meta.iloc[idx]['text']
            })
        
        return pd.DataFrame(results)
    

    def retrive_dense(self,     
                    question_df: pd.DataFrame,  # 원본 데이터셋 (df4)
                    meta_df: pd.DataFrame,      # 검색 대상 문서 (df)
                    index: faiss.Index,         # FAISS 인덱스
                    n: int = 10,                # 샘플링 개수 (전체를 하려면 len(df4) 넣으면 됨)
                    k: int = 5,                 # Top-K (몇 배로 늘릴지)
                    seed: int = 42              # 랜덤 시드
    ):
        if index.ntotal != len(meta_df):
            print(f"[주의] 인덱스({index.ntotal})와 문서({len(meta_df)}) 개수 불일치")
    
        real_n = min(n, len(question_df))
        samples = question_df.sample(n=real_n, random_state=seed).copy()
        
        print(f"{real_n}개의 질문에 대해 Top-{k} 검색을 수행합니다.")
        print(f"   예상되는 결과 행의 개수: {real_n * k}개\n")
        
        merged_results = []

        for i, (idx, row) in enumerate(tqdm(samples.iterrows(), total=len(samples), desc="Processing")):
        
            # 쿼리 생성
            qid = row['id']
            query = f"{row['paragraph']} \n\n {row['question']}"
            
            # --- 검색 로직 ---
            q_vec = self.model.encode([query], batch_size=1, max_length=1024)['dense_vecs']
            q_vec = q_vec.astype("float32")
            faiss.normalize_L2(q_vec)
            D, I = index.search(q_vec, k)
            # ----------------
            
            # Top-K만큼 반복하며 원본 데이터 + 검색 데이터 결합
            for rank, doc_idx in enumerate(I[0]):
                if doc_idx < 0 or doc_idx >= len(meta_df):
                    continue
                
                # 검색된 문서 가져오기
                retrieved_doc = meta_df.iloc[doc_idx]
                similarity_score = float(D[0][rank])
                
                # 이렇게 하면 id, paragraph, question, choices, answer 등이 다 들어감
                combined_row = row.to_dict()
                
                # 검색 결과 데이터 추가 (컬럼명 구분)
                combined_row['ctx_rank'] = rank + 1                # 순위
                combined_row['ctx_score'] = similarity_score       # 유사도 점수
                combined_row['ctx_title'] = retrieved_doc['title'] # 검색된 문서 제목
                combined_row['ctx_text'] = retrieved_doc['text']   # 검색된 문서 내용
                combined_row['ctx_id'] = retrieved_doc['doc_id']   # (있다면) 문서 ID
                
                merged_results.append(combined_row)

        # 4. DataFrame 생성
        final_df = pd.DataFrame(merged_results)
        
        print(f"\n완료! 총 {len(final_df)}개의 행이 생성되었습니다.")
        return final_df
    