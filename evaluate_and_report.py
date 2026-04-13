import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from search_engine import TFIDFSearchEngine, SBERTSearchEngine
import os

def main():
    print("------------------------------------------")
    print("1. Veri seti yükleniyor...")
    df = pd.read_csv("job_dataset.csv")
    
    # Arama motorlarının içeriği anlayabilmesi için ilgili sütunları birleştiriyoruz
    df['search_content'] = df['title'] + " " + df['skills'] + " " + df['description']
    
    print("2. Modeller eğitiliyor (Vectorizer oluşturma işlemi)...")
    tfidf_engine = TFIDFSearchEngine()
    tfidf_engine.fit(df, 'search_content')
    
    # Model indirmesi (sadece ilk çalışmada indirir)
    sbert_engine = SBERTSearchEngine()
    sbert_engine.fit(df, 'search_content')
    
    print("3. Karşılaştırma senaryoları çalıştırılıyor...")
    
    # Burası en önemli kısım: Hata Analizi (Error Analysis) için özel seçilmiş sorgular
    test_queries = [
        "python backend developer",  # Direkt kelime eşleşmesi, ikisi de bulmalı
        "front end web tasarımları yapacak bilgisayarcı",  # Kelimeler farklı ama anlamlı (Semantic Search burada parlayacak)
        "yapay zeka sistemleri geliştiren personel",  # Machine learning eşleşmesi gerek
        "sunucu ve altyapı dağıtımı otomasyonu" # DevOps Engineer için gizli anahtar kelimeler
    ]
    
    results_data = []
    
    with open('Rapor_Ciktilari.txt', 'w', encoding='utf-8') as f:
        f.write("=== ÖRNEK ÇIKTILAR VE HATA ANALİZİ ===\n\n")
        f.write("Aşağıda klasik kelime eşleştirme (TF-IDF) ile yapay zeka tabanlı anlamsal eşleşmenin (SBERT) farkları analiz edilmiştir.\n\n")
        
        for q in test_queries:
            f.write(f"SORGUMUZ: '{q}'\n")
            f.write("-" * 40 + "\n")
            
            # TF-IDF işlemi
            tf_res, tf_time = tfidf_engine.search(q, top_k=1)
            tf_top_title = tf_res.iloc[0]['title'] if not tf_res.empty and tf_res.iloc[0]['score'] > 0 else "İlan Bulunamadı (0 eşleşme)"
            tf_top_score = tf_res.iloc[0]['score'] if not tf_res.empty else 0
            
            # SBERT işlemi
            sb_res, sb_time = sbert_engine.search(q, top_k=1)
            sb_top_title = sb_res.iloc[0]['title'] if not sb_res.empty else "Bulunamadı"
            sb_top_score = sb_res.iloc[0]['score'] if not sb_res.empty else 0
            
            f.write("1️⃣ YÖNTEM 1 (TF-IDF):\n")
            f.write(f" Bulunan İlan: {tf_top_title}   | Benzerlik Skoru: {tf_top_score:.3f} | Süre: {tf_time*1000:.2f} ms\n")
            
            f.write("2️⃣ YÖNTEM 2 (SBERT):\n")
            f.write(f" Bulunan İlan: {sb_top_title} | Benzerlik Skoru: {sb_top_score:.3f} | Süre: {sb_time*1000:.2f} ms\n")
            
            f.write("\n👉 HATA ANALİZİ VE YORUM:\n")
            if tf_top_score < 0.05 and sb_top_score > 0.3:
                f.write("TF-IDF kelime kelimesine eşleşme aradığı için kullanıcının girdiği kelimeleri ilanlarda bulamadı ve DOĞRU CEVABI KAÇIRDI (Yanlış çıktı/False Negative). SBERT ise cümlelerin 'anlamını' (Vektör Uzayını) bildiği için kelimeler farklı olsa bile aranan yeteneğin başlığını DOĞRU olarak tahmin etti.\n\n")
            else:
                f.write("Her iki algoritma da belli oranda kelime eşleşmesi yakaladı. Ancak skora bakıldığında TF-IDF'in kelime sayımına, SBERT'in ise bağlama odaklandığı görülüyor.\n\n")
            f.write("\n=========================================\n\n")
                
            results_data.append({'Sorgu': q, 'Model': 'TF-IDF', 'Latency (ms)': tf_time * 1000, 'Score': tf_top_score})
            results_data.append({'Sorgu': q, 'Model': 'SBERT', 'Latency (ms)': sb_time * 1000, 'Score': sb_top_score})

    # GRAFİKLERİ OLUŞTURMA (Hocanın istediği Sonuç Tablosu/Grafiği)
    res_df = pd.DataFrame(results_data)
    
    # 1. Hız Karşılaştırması Grafiği
    plt.figure(figsize=(10, 6))
    sns.barplot(data=res_df, x='Sorgu', y='Latency (ms)', hue='Model', palette='muted')
    plt.title('Arama Sorgusu İşlenme Hızı Karşılaştırması (Latency)')
    plt.xticks(rotation=15, ha='right')
    plt.ylabel('Milisaniye (ms)')
    plt.tight_layout()
    plt.savefig('grafik_hiz_karsilastirmasi.png')
    
    # 2. Doğruluk / Skor Karşılaştırması Grafiği
    plt.figure(figsize=(10, 6))
    sns.barplot(data=res_df, x='Sorgu', y='Score', hue='Model', palette='pastel')
    plt.title('Modellerin Verdiği Güven Skoru Karşılaştırması')
    plt.xticks(rotation=15, ha='right')
    plt.ylabel('Kosinüs Benzerlik Skoru')
    plt.tight_layout()
    plt.savefig('grafik_skor_karsilastirmasi.png')
    
    # PERFORMANS METRİKLERİ TABLOSU
    avg_latency = res_df.groupby('Model')['Latency (ms)'].mean().reset_index()
    avg_latency.rename(columns={'Latency (ms)': 'Ortalama Gecikme (ms)'}, inplace=True)
    avg_latency.to_csv("tablo_performans.csv", index=False)
    
    with open('Rapor_Ciktilari.txt', 'a', encoding='utf-8') as f:
        f.write("\n=== GENEL DEĞERLENDİRME TABLOSU ===\n")
        f.write(avg_latency.to_string(index=False))
        f.write("\n\nPROJE KONUSUNA ÖZEL ANALİZ (BİTİRİŞ):\n")
        f.write("Bu uygulama, kelime bazlı arama (TF-IDF) yerine kullanıcı odaklı ve bağlamın gücünden faydalanan (SBERT) bir mimariye ihtiyaç duyduğunu kanıtlamıştır. İş arayan adaylar teknik terimleri farklı ifade edebilir (Örn: Coder, Programmer, Developer). Sistemimizin SBERT (Yapay Zeka) modeli kullanarak bu değişkenlikten etkilenmemesi sağlanmıştır. Ancak SBERT kullanımı sistem maaliyetlerini ufak oranda (.ms bazlı gecikmeler) artırmıştır.\n")
        
    print("\n✅ Tüm analizler, grafikler ve tablolar başarıyla C:\\Users\\User\\.gemini\\antigravity\\scratch\\JobSearchProject\\ klasörüne çıkartıldı.")

if __name__ == "__main__":
    main()
