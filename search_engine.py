import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# YÖNTEM 1: Klasik Yaklaşım (TF-IDF Lexical Search)
class TFIDFSearchEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.df = None

    def fit(self, df, text_column):
        self.df = df
        self.tfidf_matrix = self.vectorizer.fit_transform(df[text_column].fillna(''))

    def search(self, query, top_k=5):
        start_time = time.time()
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            res = self.df.iloc[idx].copy()
            res['score'] = similarities[idx]
            results.append(res)
        
        latency = time.time() - start_time
        return pd.DataFrame(results), latency


# YÖNTEM 2: BM25 (Gelişmiş Lexical Search)
class BM25SearchEngine:
    def __init__(self):
        self.bm25 = None
        self.df = None

    def fit(self, df, text_column):
        self.df = df
        # BM25 requires tokenized lists of strings
        tokenized_corpus = [str(doc).lower().split(" ") for doc in df[text_column].fillna('')]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, top_k=5):
        start_time = time.time()
        tokenized_query = query.lower().split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            res = self.df.iloc[idx].copy()
            res['score'] = doc_scores[idx]
            results.append(res)
            
        latency = time.time() - start_time
        return pd.DataFrame(results), latency


# YÖNTEM 3: Yapay Zeka Özellikli Yaklaşım (SBERT Anlamsal - Semantic Search)
class SBERTSearchEngine:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.df = None

    def fit(self, df, text_column):
        self.df = df
        print("SBERT Embeddings başlatıldı. (Bu işlem veri büyüklüğüne göre biraz sürebilir...)")
        self.embeddings = self.model.encode(df[text_column].fillna('').tolist(), convert_to_tensor=True)
        print("SBERT Embeddings tamamlandı.")

    def search(self, query, top_k=5):
        from sentence_transformers import util
        start_time = time.time()
        
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        
        top_results = np.argpartition(-cos_scores.cpu().numpy(), range(top_k))[:top_k]
        top_results = sorted(top_results, key=lambda x: cos_scores[x], reverse=True)
        
        results = []
        for idx in top_results:
            res = self.df.iloc[idx].copy()
            score = cos_scores[idx].item()
            res['score'] = score
            results.append(res)
            
        latency = time.time() - start_time
        return pd.DataFrame(results), latency


# YÖNTEM 4: Hibrit Arama (Ensemble: SBERT %70 + BM25 %30)
class HybridSearchEngine:
    def __init__(self, sbert_engine, bm25_engine, alpha=0.70):
        self.sbert = sbert_engine
        self.bm25 = bm25_engine
        self.alpha = alpha  # SBERT'in ağırlığı
        self.df = sbert_engine.df

    def _minmax_scale(self, scores):
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val - min_val == 0:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    def search(self, query, top_k=5):
        start_time = time.time()
        
        # 1. SBERT Skorları
        from sentence_transformers import util
        query_embedding = self.sbert.model.encode(query, convert_to_tensor=True)
        sbert_scores = util.cos_sim(query_embedding, self.sbert.embeddings)[0].cpu().numpy()
        
        # 2. BM25 Skorları
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.bm25.get_scores(tokenized_query)
        
        # Min-Max Normalizasyonu (Skorları 0-1 aralığına sıkıştırma)
        sbert_scaled = self._minmax_scale(sbert_scores)
        bm25_scaled = self._minmax_scale(bm25_scores)
        
        # Ağırlıklandırılarak birleştirilmesi (Rapor Tablo 1'deki Hibrit Yöntem)
        hybrid_scores = (self.alpha * sbert_scaled) + ((1.0 - self.alpha) * bm25_scaled)
        
        # En iyileri sıralama
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            res = self.df.iloc[idx].copy()
            res['score'] = hybrid_scores[idx]
            results.append(res)
            
        latency = time.time() - start_time
        return pd.DataFrame(results), latency
