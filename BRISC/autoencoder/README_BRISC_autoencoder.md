# BRISC 2D Autoencoder Starter

Bu klasör, BRISC 2025 verisiyle ilk autoencoder denemesini yapmak için hazırlanmıştır.

## Neden BRISC ile 2D autoencoder?

BRISC verisi 2D JPG görüntülerden oluştuğu için BraTS gibi 3D NIfTI verilerine göre daha hızlı denenir. Bu aşamada amaç `missing modality reconstruction` değil, **denoising/reconstruction pretraining** yapmaktır.

Yani model şu problemi öğrenir:

```text
Corrupted/noisy MRI image -> Autoencoder -> Clean MRI image
```

Bu, daha sonra segmentasyon modelinin encoder tarafına ön-eğitim olarak aktarılabilir.

## Dataset bilgisi

Mevcut analiz çıktına göre BRISC tarafı yaklaşık olarak:

- 6000 classification image
- 4793 segmentation image
- 4793 segmentation mask
- 2D RGB JPG image
- En yaygın çözünürlük: 512 x 512
- Sınıflar: glioma, meningioma, pituitary, no_tumor
- Düzlemler: axial, coronal, sagittal

Autoencoder eğitiminde maskeler kullanılmaz. Sadece image dosyaları kullanılır.

## Kurulum

```bash
pip install -r requirements_autoencoder.txt
```

veya manuel:

```bash
pip install torch torchvision pandas pillow matplotlib tqdm numpy
```

## Eğitim komutu

Windows örneği:

```bash
python BRISC/autoencoder/train_brisc_autoencoder.py ^
  --dataset_path "D:/Downloads/brisc_Dataset2025/brisc2025" ^
  --output_dir "BRISC/outputs/autoencoder_v1" ^
  --image_size 256 ^
  --batch_size 16 ^
  --epochs 30
```

Linux/Mac örneği:

```bash
python BRISC/autoencoder/train_brisc_autoencoder.py \
  --dataset_path "/path/to/brisc2025" \
  --output_dir "BRISC/outputs/autoencoder_v1" \
  --image_size 256 \
  --batch_size 16 \
  --epochs 30
```

## Üretilen çıktılar

Eğitim sonunda şu dosyalar oluşur:

```text
BRISC/outputs/autoencoder_v1/
├── best_autoencoder.pt
├── last_autoencoder_state_dict.pt
├── training_history.csv
├── config.json
└── best_reconstruction_examples.png
```

## İlk deney ayarları

Önerilen başlangıç:

```text
image_size = 256
batch_size = 16
learning_rate = 1e-3
epochs = 30
loss = L1 + 0.25 * MSE
input = noisy / randomly erased image
target = clean image
```

GPU belleğin yeterliyse batch size 32 denenebilir. Eğitim yavaşsa `image_size=128` ile hızlı prototip yapılabilir.

## Sonraki adım

Autoencoder düzgün reconstruction üretiyorsa bir sonraki deney şudur:

```text
1. BRISC segmentation baseline kur: image -> U-Net -> mask
2. Autoencoder encoder ağırlıklarını U-Net encoder'a aktar
3. Random init U-Net vs AE-pretrained U-Net karşılaştır
4. Dice / IoU / pixel accuracy sonuçlarını raporla
```
