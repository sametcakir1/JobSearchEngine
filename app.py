from flask import Flask, request, jsonify, render_template
import pandas as pd
from search_engine import TFIDFSearchEngine, SBERTSearchEngine, HybridSearchEngine, BM25SearchEngine
import os

app = Flask(__name__)

print("Arama motorları ve veri setleri başlatılıyor, lütfen bekleyin...")
df = pd.read_csv("job_dataset.csv")
df['search_content'] = df['title'] + " " + df['skills'] + " " + df['description']

# Modelleri yükleyelim (ilk uygulamaya girişte birkaç saniye sürebilir)
sbert_engine = SBERTSearchEngine()
sbert_engine.fit(df, 'search_content')

tfidf_engine = TFIDFSearchEngine()
tfidf_engine.fit(df, 'search_content')

bm25_engine = BM25SearchEngine()
bm25_engine.fit(df, 'search_content')

hybrid_engine = HybridSearchEngine(lexical_engine=bm25_engine, sbert_engine=sbert_engine, sbert_weight=0.7)

print("✅ Modeller başarıyla yüklendi. Web arayüzüne gidebilirsiniz.")

@app.route('/')
def index():
    # templates/index.html dosyasını render eder
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    method = data.get('method', 'sbert')
    
    if not query:
        return jsonify([])
        
    try:
        if method == 'sbert':
            results_df, latency = sbert_engine.search(query, top_k=8)
        elif method == 'hybrid':
            results_df, latency = hybrid_engine.search(query, top_k=8)
        else:
            results_df, latency = tfidf_engine.search(query, top_k=8)
            
        results = results_df.to_dict(orient='records')
        
        # Sonuçları web arayüzüne göndermek için hazırla
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
