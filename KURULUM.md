# ABMEM Kurulumu

ABMEM iki şekilde kurulabilir.

| | Docker (önerilen) | Geliştirme kurulumu |
|---|---|---|
| Kimin için | Simülatörü kullanmak isteyenler | Kodu değiştirmek ve testleri çalıştırmak isteyenler |
| Kurulacaklar | Docker Desktop ve Git | Python 3.11, PostgreSQL 16 ve Git |
| Veritabanı | Bir container içinde kendiliğinden oluşturulur | Sen oluşturursun |
| Süre | İlk açılışta yaklaşık 10 dakika | Yaklaşık 20 dakika |

İkisinin sonunda da aynı web arayüzü açılır. Bu dokümanın İngilizcesi: `INSTALL.md`.

---

## A. Docker ile kurulum (önerilen)

### A1. Docker Desktop'ı kur

**Windows 10/11**
1. Docker Desktop'ı https://www.docker.com/products/docker-desktop/ adresinden indirip kurulum dosyasını çalıştır.
2. "Use WSL 2 instead of Hyper-V" seçeneğini işaretli bırak.
3. Kurulum isterse bilgisayarı yeniden başlat.
4. Docker Desktop'ı aç ve "Engine running" yazısını görene kadar bekle.

Docker Desktop WSL'in eksik ya da eski olduğunu söylerse bir terminal (PowerShell) açıp `wsl --update` komutunu çalıştır, sonra Docker Desktop'ı yeniden aç. Sanallaştırmanın kapalı olduğunu söylerse bilgisayarın BIOS/UEFI ayarlarından "Intel VT-x" ya da "AMD-V/SVM" seçeneğini aç.

**macOS**: Docker Desktop'ın işlemcine uygun sürümünü (Apple silicon ya da Intel) indir, Applications klasörüne sürükle ve aç.

**Linux**: Docker Engine ve Docker Compose eklentisini https://docs.docker.com/engine/install/ adresindeki dağıtımına uygun talimatlarla kur.

Kurulumu bir terminalde kontrol et:

```
docker --version
docker compose version
```

Disk alanı: İmaj yaklaşık 2,5 GB yer kaplar, veritabanı da run sayısı arttıkça büyür. Docker Desktop verisini varsayılan olarak sistem sürücüsünde tutar. O sürücüde 10 GB'tan az boş yer varsa önce Docker Desktop'ta Settings > Resources > Advanced > Disk image location ayarından başka bir sürücüye taşı.

### A2. Git'i kur ve projeyi indir

Git'i https://git-scm.com/downloads adresinden kur (Windows'ta varsayılan seçenekleri değiştirme). Sonra bir terminalde projenin duracağı klasöre gidip şunları çalıştır:

```
git clone https://github.com/Erdemhan/abmem_web.git
cd abmem_web
```

Git kullanmak istemiyorsan https://github.com/Erdemhan/abmem_web sayfasındaki "Code" düğmesinden ZIP olarak indirip aç. Bu durumda güncellemek için `git pull` yerine yeniden indirmen gerekir.

### A3. Başlat

Proje klasöründe (`docker-compose.yml` dosyasının bulunduğu klasör):

```
docker compose up --build -d
```

İlk açılışta Python, PyTorch ve PostgreSQL indirilir ve imaj oluşturulur. Bu birkaç dakika sürer. Komut bittikten sonra uygulamanın veritabanını hazırlaması için birkaç saniye daha bekle. Sonra tarayıcıda şu adresi aç:

**http://localhost:8000/**

İlk açılışta veritabanı oluşturulur, bütün migration'lar uygulanır ve başlangıç verisi yüklenir: 8 kaynak ve 3 senaryo şablonu (Mixed, Only Fossil, Only Renewable), library agent'larıyla birlikte.

### A4. Çalıştığını kontrol et

1. Soldaki menüden "Templates" sayfasını aç. Üç senaryo listelenmeli.
2. "Only Renewable" senaryosunu çalıştır. Yaklaşık 10–20 saniye içinde run FINISHED olmalı ve analiz sayfası açılmalı.

Sayfa açılmıyorsa logu incele:

```
docker compose logs web
```

Son satırlarda "Starter data loaded ..." (sadece ilk açılışta) ve "Listening at: http://0.0.0.0:8000" görünmeli.

### A5. Günlük kullanım

| İş | Komut (proje klasöründe) |
|---|---|
| Başlat | `docker compose up -d` |
| Durdur (veriler korunur) | `docker compose down` |
| Son sürüme güncelle | `git pull` ve ardından `docker compose up --build -d` |
| Logu izle | `docker compose logs -f web` |
| Veritabanını yedekle | `docker compose exec db pg_dump -U postgres -Fc abmem > abmem.dump` |
| Her şeyi sil (bütün run'lar dahil) | `docker compose down -v` |

Bu komutlar için Docker Desktop'ın açık olması gerekir. Migration'lar her açılışta kendiliğinden uygulanır. Güncellemek için `git pull` ve `docker compose up --build -d` yeterlidir.

### A6. İsteğe bağlı ayarlar

Hiçbir ayar yapmazsan varsayılanlar kullanılır. Değiştirmek için proje klasöründeki `.env.docker.example` dosyasını `.env` adıyla kopyala ve düzenle:

- `ABMEM_PORT`: web sayfasının portu (varsayılan 8000). Başka bir program 8000'i kullanıyorsa değiştir.
- `DB_PASSWORD`: container'daki veritabanının şifresi.
- `ABMEM_LOAD_STARTER_DATA=false`: boş bir veritabanıyla başlar.

Ardından `docker compose up -d` komutunu tekrar çalıştır. Uygulamaya yalnızca bu bilgisayardan erişilebilir. Ağdan erişime açmak ve Docker'ın veriyi nerede tuttuğu `docs/docker.md` dosyasında anlatılıyor.

---

## B. Geliştirme kurulumu

Kodu değiştirmek ya da testleri çalıştırmak için bu kurulumu kullan.

### B1. Gerekenler

- Python 3.11 (python.org kurulum dosyası ya da bir conda ortamı: `conda create -n abmem python=3.11`)
- PostgreSQL 16 (https://www.postgresql.org/download/). Kurulum sırasında belirlediğin `postgres` kullanıcısının şifresini unutma.
- Git

### B2. Projeyi indir ve paketleri kur

```
git clone https://github.com/Erdemhan/abmem_web.git
cd abmem_web
python -m venv .venv
```

Ortamı etkinleştir:

- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- Windows (Git Bash): `source .venv/Scripts/activate`
- macOS/Linux: `source .venv/bin/activate`

conda kullanıyorsan venv yerine `conda activate abmem` komutunu kullan.

Önce PyTorch'u, sonra diğer paketleri kur:

```
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
```

NVIDIA ekran kartın varsa bunun yerine torch 2.4.0'ın sürücüne uygun CUDA sürümünü kurabilirsin (bkz. https://pytorch.org/get-started/previous-versions/). Run'lar için GPU gerekmez.

### B3. Ayarlar

Proje klasöründeki `.env.example` dosyasını `.env` adıyla kopyala ve en azından şunları yaz:

```
DJANGO_DEBUG=true
DB_PASSWORD=<postgres kullanıcısının şifresi>
```

`DJANGO_DEBUG=true` olduğu sürece `DJANGO_SECRET_KEY` boş kalabilir. `DJANGO_DEBUG=false` yaparsan bir anahtar gerekir. Anahtarı şu komutla üretebilirsin:

```
python -c "from django.core.management.utils import get_random_secret_key as g; print(g())"
```

### B4. Veritabanını oluştur

pgAdmin ile ya da bir terminalde `abmem` adlı, UTF-8 kodlamalı boş bir veritabanı oluştur:

```
createdb -U postgres -E UTF8 -T template0 abmem
```

Sonra tabloları oluştur ve başlangıç verisini yükle:

```
python manage.py migrate
python manage.py load_starter_data
```

### B5. Çalıştır

```
python manage.py runserver
```

Tarayıcıda http://127.0.0.1:8000/ adresini aç. Geliştirme sunucusu bir dosya değiştiğinde kendini yeniden başlatır. Yeni bir sürümü çektikten sonra sunucuyu başlatmadan önce `python manage.py migrate` çalıştır, çünkü yeni sürümler veritabanı migration'ları ekleyebilir.

### B6. Testler

```
python -m pytest
```

Testler kendi geçici veritabanlarını oluşturur, bu yüzden PostgreSQL kullanıcısının veritabanı oluşturma (CREATEDB) yetkisi olmalı. `postgres` kullanıcısında bu yetki vardır, sadece sonradan oluşturduğun kullanıcılarda eksik olabilir. Gerekirse pgAdmin ya da psql'de bir kez `ALTER ROLE <kullanıcı> CREATEDB;` komutunu çalıştır. Docker duman testi (`docker/smoke_test.sh`) her push'tan sonra GitHub Actions'ta otomatik koşar.

---

## Sorun giderme

| Mesaj ya da belirti | Sebep ve çözüm |
|---|---|
| `failed to connect to the docker API` | Docker Desktop çalışmıyor. Aç ve "Engine running" yazısını bekle. |
| `port is already allocated` ya da başka bir programın sayfası açılıyor | 8000 portu kullanımda. `.env` dosyasında `ABMEM_PORT=8001` yap (A6) ve http://localhost:8001/ adresini aç. |
| Docker `read-only file system` ya da `input/output error` veriyor | Docker'ın disk dosyasının bulunduğu sürücü dolu. Yer aç ya da disk dosyasını başka sürücüye taşı (A1). |
| `exec docker/entrypoint.sh: no such file or directory` | Dosyalar eski bir Git ayarı yüzünden Windows satır sonlarıyla indirilmiş. `git config core.autocrlf input` komutunu çalıştır, klasörü silip yeniden klonla. |
| `DJANGO_SECRET_KEY must be set when DJANGO_DEBUG is false` (geliştirme kurulumu) | `.env` dosyasında `DJANGO_DEBUG=true` yap ya da bir anahtar gir (B3). |
| `password authentication failed for user "postgres"` | `.env` dosyasındaki `DB_PASSWORD`, PostgreSQL şifresiyle aynı değil. |
| `database "abmem" does not exist` | Veritabanını oluştur (B4). |
| Testlerde `permission denied to create database` | CREATEDB yetkisini ver (B6). |
| Run ilk periyottan sonra bekliyor | Senaryo "Period by period" modunda. İzleme sayfasındaki Continue düğmesini kullan ya da senaryoyu "Only result" moduna al. |
