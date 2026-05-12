from flask import Flask, request, jsonify, render_template
import pandas as pd
from search_engine import TFIDFSearchEngine, BM25SearchEngine, SBERTSearchEngine, HybridSearchEngine
import os

app = Flask(__name__)

print("Arama motorları ve veri setleri başlatılıyor, lütfen bekleyin...")
df = pd.read_csv("job_dataset.csv")
df['search_content'] = df['title'] + " " + df['skills'] + " " + df['description']

# Yapay Zeka anlamsal modeli (En yavaş yüklenen)
sbert_engine = SBERTSearchEngine()
sbert_engine.fit(df, 'search_content')

# İstatistiksel model 1
tfidf_engine = TFIDFSearchEngine()
tfidf_engine.fit(df, 'search_content')

# İstatistiksel model 2 (Gelişmiş)
bm25_engine = BM25SearchEngine()
bm25_engine.fit(df, 'search_content')

# Hibrit Model (Altın Standart: SBERT %70 + BM25 %30)
hybrid_engine = HybridSearchEngine(sbert_engine, bm25_engine, alpha=0.7)

print("✅ Tüm Modeller (TF-IDF, BM25, SBERT, Hibrit) başarıyla yüklendi! Web arayüzüne gidebilirsiniz.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    method = data.get('method', 'hybrid')
    
    if not query:
        return jsonify([])
        
    try:
        # Seçilen metoda göre arama yapma
        if method == 'hybrid':
            results_df, latency = hybrid_engine.search(query, top_k=8)
        elif method == 'bm25':
            results_df, latency = bm25_engine.search(query, top_k=8)
        elif method == 'sbert':
            results_df, latency = sbert_engine.search(query, top_k=8)
        else:
            results_df, latency = tfidf_engine.search(query, top_k=8)
            
        results = results_df.to_dict(orient='records')
        
        formatted_results = []
        for r in results:
            formatted_results.append({
                "title": r.get('title', ''),
                "company": r.get('company', ''),
                "skills": str(r.get('skills', '')).split(','),
                "description": r.get('description', ''),
                "score": round(r.get('score', 0) * 100, 1),
                "latency_ms": round(latency * 1000, 2)
            })
            
        return jsonify(formatted_results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Aura Akıllı İş Arama uygulamanız http://127.0.0.1:5000 adresinde ayağa kalktı!")
    app.run(debug=True, port=5000)
