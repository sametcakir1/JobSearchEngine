import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class BM25SearchEngine:
    def __init__(self):
        self.bm25_model = None
        self.df = None

    def fit(self, df, text_column):
        self.df = df
        # BM25 expects a list of words for each document
        tokenized_corpus = df[text_column].fillna('').apply(lambda x: x.lower().split()).tolist()
        self.bm25_model = BM25Okapi(tokenized_corpus)

    def search(self, query, top_k=5):
        start_time = time.time()
        tokenized_query = query.lower().split()
        
        # Calculate BM25 scores
        doc_scores = self.bm25_model.get_scores(tokenized_query)
        
        # Get top indices
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            res = self.df.iloc[idx].copy()
            res['score'] = doc_scores[idx]
            results.append(res)
            
        latency = time.time() - start_time
        return pd.DataFrame(results), latency

# YÖNTEM 1: Klasik Yaklaşım (TF-IDF Lexical Search)
class TFIDFSearchEngine:
    def __init__(self):
        # Stop words olarak ingilizce verilmiş ama biz Türkçe metinler yazdıysak basic bir vectorizer kuralım
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.df = None

    def fit(self, df, text_column):
        self.df = df
        self.tfidf_matrix = self.vectorizer.fit_transform(df[text_column].fillna(''))

    def search(self, query, top_k=5):
        start_time = time.time()
        # Sorguyu vektör uzayına çevirme
        query_vec = self.vectorizer.transform([query])
        
        # Kosinüs benzerliği (Sorgu vs Tüm İlanlar)
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # En yüksek skorlu Top K indexi alma
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            res = self.df.iloc[idx].copy()
            res['score'] = similarities[idx]
            results.append(res)
        
        latency = time.time() - start_time
        return pd.DataFrame(results), latency


# YÖNTEM 2: Yapay Zeka Özellikli Yaklaşım (SBERT Anlamsal - Semantic Search)
class SBERTSearchEngine:
    # Çok dilli (Türkçe destekli) ve hızlı bir model kullanıyoruz
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.df = None

    def fit(self, df, text_column):
        self.df = df
        print("SBERT Embeddings başlatıldı. (Bu işlem veri büyüklüğüne göre biraz sürebilir...)")
        # Tüm ilan veri setini embedding vektörlerine çeviriyoruz
        self.embeddings = self.model.encode(df[text_column].fillna('').tolist(), convert_to_tensor=True)
        print("SBERT Embeddings tamamlandı.")

    def search(self, query, top_k=5):
        from sentence_transformers import util
        start_time = time.time()
        
        # Kullanıcı sorgusunu vektöre çevirme
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Vektörler arası kosinüs benzerliği hesaplama
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        
        # En iyi sonuçları getirme
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


# YÖNTEM 3: Hibrit Arama (BM25 + SBERT)
class HybridSearchEngine:
    def __init__(self, lexical_engine, sbert_engine, sbert_weight=0.7):
        self.lexical_engine = lexical_engine
        self.sbert_engine = sbert_engine
        self.sbert_weight = sbert_weight # Anlama %70, kelimeye %30 önem ver

    def search(self, query, top_k=5):
        start_time = time.time()
        
        # Lexical (BM25) Skorları
        lex_res, _ = self.lexical_engine.search(query, top_k=len(self.lexical_engine.df))
        # SBERT Skorları
        sbert_res, _ = self.sbert_engine.search(query, top_k=len(self.sbert_engine.df))
        
        lex_scores = lex_res.set_index(lex_res.index)['score']
        sbert_scores = sbert_res.set_index(sbert_res.index)['score']
        
        df_combined = self.lexical_engine.df.copy()
        
        df_combined['lexical_score'] = df_combined.index.map(lex_scores).fillna(0)
        df_combined['sbert_score'] = df_combined.index.map(sbert_scores).fillna(0)
        
        # BM25 skorları sınırsızdır (Örn: 15.6, 22.1). SBERT ise 0-1 arasıdır.
        # Toplayabilmek için BM25 skorlarını 0-1 arasına normalize ediyoruz (Min-Max Scaling)
        max_lex = df_combined['lexical_score'].max()
        if max_lex > 0:
            df_combined['lexical_score_norm'] = df_combined['lexical_score'] / max_lex
        else:
            df_combined['lexical_score_norm'] = 0.0
            
        # SBERT zaten 0-1 arası kosinüs değeridir
        df_combined['sbert_score_norm'] = df_combined['sbert_score'].clip(lower=0) 
        
        # HİBRİT FORMÜL (Normalize edilmiş skorlarla)
        df_combined['score'] = (df_combined['sbert_score_norm'] * self.sbert_weight) + (df_combined['lexical_score_norm'] * (1 - self.sbert_weight))
        
        # Skorlara göre sırala ve top_k kadarını al
        df_hybrid_top = df_combined.sort_values(by='score', ascending=False).head(top_k)
        
        latency = time.time() - start_time
        return df_hybrid_top, latency
