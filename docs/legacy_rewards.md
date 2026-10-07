# Eski Ödül Tanımları (arşiv)

**Amaç:** K1 kararıyla (2026-10-06) ajan ödülü "normalize, kırpılmamış periyot kârı" olarak değiştirildi. Bu dosya, önceki ödül tanımlarını ve hangi sonuçların hangi tanımla üretildiğini kayıt altında tutar. Yeni tanım `FORMULATION.md`'de yer alacaktır.

Tüm sürümlerde `AgentAlgorithm.learn` ödülü `int(reward)` ile tam sayıya çevirir (`analiz.md`, C1). Bu dönüşüm, ödül para cinsindeyken (sürüm 1 ve 2) önemsizdir, normalize edildiğinde (sürüm 3 ve 4) ödülü 0'a indirir.

| Sürüm | Commit aralığı | Tanım | `int()` sonrası |
|---|---|---|---|
| 1 | `222fea8` (2025-01-21) – `2b60ef5` | Gelir | Ham değer korunur |
| 2 | `3216f5d` (2025-02-17) – `105ee1d` | Kâr (para cinsinden) | Ham değer korunur |
| 3 | `12c4d74` (2025-05-27) – `33dbe50` (HEAD) | 0.5 · normalize büyüme + 0.5 · benzerlik | Kayıtlı verilerde hep 0 |
| 4 | commit edilmemiş çalışma ağacı (son değişiklik 2025-06-17) | 0.4 · norm_bütçe + 0.4 · norm_kâr + 0.2 · benzerlik, [0,1]'e kırpılmış | Hep 0 |

Ortak gösterim (bir teklif için):
- `q` teklif miktarı (santral kapasitesi), `q_acc` kabul edilen miktar, `p_acc` kabul fiyatı, `p` teklif fiyatı, `PTF` periyodun takas fiyatı.
- `c_s` sabit maliyet (`omCost + investmentCost + co2Cost`), `c_v` değişken maliyet (`fuelCost`).
- `P_max = 200` (kodda sabit kodlanmış fiyat tavanı).

Kâr (bütçe artışı):

```
g = q_acc · p_acc − (c_s · q + c_v · q_acc)      q_acc > 0 ise
g = − c_s · q                                     teklif reddedildiyse
```

## Sürüm 1 — Gelir
Ajanın bir önceki periyottaki tüm teklifleri üzerinden: `r = Σ p_acc · q_acc`. Maliyetler yok.

## Sürüm 2 — Kâr
Ajanın bir önceki periyottaki tüm teklifleri üzerinden: `r = Σ g`. Para cinsinden, normalize edilmemiş.

## Sürüm 3 — HEAD bileşik ödülü (`compute_composite_reward`, `33dbe50`)
Sadece ilk teklif (`last_offers[0]`) kullanılır.

```
normalize_büyüme = g / (P_max · q)                         (kırpılmaz)
benzerlik        = clip(1 − |p − PTF| / P_max, 0, 1)
r                = 0.5 · normalize_büyüme + 0.5 · benzerlik
```

Kayıtlı verilerde aralık: -0.013 ile 0.84.

## Sürüm 4 — Çalışma ağacındaki bileşik skor (`agent_service.py:118-158`)
Sadece ilk teklif kullanılır. `max_capacity` argümanı, önceki döngüden kalan `offer.amount` değeridir.

```
norm_bütçe = clip(g / (P_max · q), −1, 1)
norm_kâr   = clip((g / q_acc) / P_max, −1, 1)      teklif reddedildiyse 0
benzerlik  = clip(1 − |p − PTF| / P_max, 0, 1)
r          = clip(0.4 · norm_bütçe + 0.4 · norm_kâr + 0.2 · benzerlik, 0, 1)
```

Kayıtlı verilerde aralık: 0 ile 0.75, ortalama yaklaşık 0.34.

Ödül olarak sorunları:
1. Alttan 0'da kırpıldığı için zarar sinyali yok.
2. `norm_bütçe` ve `norm_kâr` aynı `g`'den türediği için kâr iki kez sayılır.
3. `benzerlik` PTF'ye yakın teklifi kârdan bağımsız ödüllendirir. Bu, PAYASPTF altında kârlı strateji olan maliyete yakın teklif vermeyle çelişir.
4. Çok santralli ajanlarda sadece ilk santral hesaba katılır.
5. `P_max = 200` sabittir, piyasa sınırları kullanılmaz.
6. 0.4 / 0.4 / 0.2 ağırlıkları için bir gerekçe kayıtlı değil.

## HPO değerlendirme metriği (ödül değil)
`hyperparam_optimization.evaluate_result`'taki `agent_score` sürüm 4 ile aynı yapıdadır. Tek fark, simülasyon boyunca toplanmış değerlerle hesaplanmasıdır (`analiz.md` 11.6). K1 kararı bu metriği değiştirmez. Metriğin yeni hali 1.11 maddesinde ayrıca belirlenecektir.

## Etkilenen kayıtlar
- `abc_log/`, `abc_logs/` (2025-06-14 – 2025-07-07): sürüm 3 veya 4 ile üretildi. Öğrenme sinyali 0'dı.
- 2026-10-06'da veritabanından silinen proxy olmayan simülasyonlar: hangi sürümle üretildikleri kayıtlı değil.
