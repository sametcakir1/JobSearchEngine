import pandas as pd
import random

# Hocanızın "Veri seti açıklaması" bölümü için oluşturduğumuz baz veri
jobs = [
    {"title": "Frontend Developer", "company": "TechNova", "skills": "React, JavaScript, CSS, HTML", "description": "Modern kullanıcı arayüzleri geliştirecek, tasarıma önem veren frontend geliştirici arıyoruz."},
    {"title": "Backend Engineer", "company": "DataCorp", "skills": "Python, Django, PostgreSQL", "description": "Ölçeklenebilir API'ler ve mikroservis mimarileri geliştirecek mühendis."},
    {"title": "Data Scientist", "company": "AI Solutions", "skills": "Python, Machine Learning, TensorFlow", "description": "Büyük veri setlerini analiz edip şirketimiz için tahminleme modelleri kuracak veri bilimci."},
    {"title": "Full Stack Developer", "company": "WebWorks", "skills": "JavaScript, Node.js, React, MongoDB", "description": "Uygulamanın hem sunucu hem de istemci tarafını yönetebilecek deneyimli takım arkadaşı."},
    {"title": "DevOps Engineer", "company": "CloudSys", "skills": "AWS, Docker, Kubernetes, CI/CD", "description": "Bulut altyapımızı yönetecek ve dağıtım süreçlerimizi (deployment) otomatize edecek mühendis."},
    {"title": "UI/UX Designer", "company": "DesignStudio", "skills": "Figma, Adobe XD, Sketch", "description": "Kullanıcı deneyimini güçlendirecek, modern ve sezgisel tasarımlar yapacak yetenek."},
    {"title": "Software Engineer", "company": "TechNova", "skills": "Java, Spring Boot, MySQL", "description": "Kurumsal düzeyde sağlam uygulamalar geliştirecek yazılım mühendisi aranıyor."},
    {"title": "Machine Learning Engineer", "company": "AI Solutions", "skills": "PyTorch, Python, NLP", "description": "Arama motorumuz ve öneri sistemimiz için gelişmiş NLP (Doğal Dil İşleme) modelleri geliştirecek uzman."},
    {"title": "Data Analyst", "company": "FinancePro", "skills": "SQL, Excel, Tableau", "description": "Finansal verilerden iş kararlarını destekleyecek anlamlı raporlar ve içgörüler üretecek veri analisti."},
    {"title": "Mobile App Developer", "company": "AppTech", "skills": "Flutter, Dart, Firebase", "description": "iOS ve Android cihazlarda sorunsuz çalışacak çapraz platform mobil uygulamalar geliştirecek arkadaş aranıyor."},
    {"title": "Cybersecurity Specialist", "company": "SecureNet", "skills": "Network Security, Penetration Testing, Linux", "description": "Şirket içi ağlarımızın ve müşteri verilerimizin güvenliğini sağlayacak, sızma testleri yapacak uzman."},
    {"title": "Product Manager", "company": "InnoSoft", "skills": "Agile, Scrum, Jira", "description": "Yazılım ürünlerini zamanında ve kaliteli çıkartmak için farklı fonksiyonlu takımlara liderlik yapacak yönetici."}
]

# Testlerde metriklerin ve sürelerin anlamlı görünmesi için veriyi çoğaltalım (Sentetik olarak büyütülmüş veri seti)
ek_cumleler = [
    " Esnek çalışma saatleri ve harika yan haklar sunuyoruz.",
    " Hibrit ve uzaktan (remote) çalışma imkanı mevcuttur.",
    " Hızlı tempolu, yenilikçi ve harika bir ekibe katılın.",
    " Kendini geliştirmeye açık, öğrenmeyi seven birini arıyoruz."
]

buyuk_veri_seti = []
for _ in range(15):  # Toplam 12 * 15 = 180 ilanlık genişletilmiş veri
    for item in jobs:
        new_item = item.copy()
        new_item['description'] = new_item['description'] + random.choice(ek_cumleler)
        buyuk_veri_seti.append(new_item)

df = pd.DataFrame(buyuk_veri_seti)
df.to_csv("job_dataset.csv", index=False)
print(f"Başarı: 'job_dataset.csv' adlı veri seti oluşturuldu. Toplam ilan sayısı: {len(df)}")
