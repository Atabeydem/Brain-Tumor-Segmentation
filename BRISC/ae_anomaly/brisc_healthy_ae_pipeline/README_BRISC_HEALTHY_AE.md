# BRISC Healthy-only Autoencoder Pipeline

Bu paket Mehmet hocanın tarif ettiği yönteme göre hazırlanmıştır.

## Mantık

1. Autoencoder sadece `classification_task/train/no_tumor` ile eğitilir.
2. Reconstruction başarısı gözle değil, PSNR / MSE / MAE / opsiyonel SSIM ile ölçülür.
3. Original ve reconstructed image pair'leri iki ayrı klasöre kaydedilir.
4. Tümörlü segmentation görüntülerinde error map çıkarılır: `error_map = abs(original - reconstructed)`.
5. Error map threshold edilerek predicted mask üretilir ve ground truth mask ile Dice / IoU hesaplanır.

## Kurulum

```powershell
cd "C:\Users\Atabey\Brain-Tumor-Segmentation\BRISC"
mkdir ae_anomaly
# zip içindeki dosyaları ae_anomaly içine çıkar
cd ae_anomaly
pip install scikit-image
```

`scikit-image` sadece SSIM için gereklidir. Kurulu değilse PSNR/MSE yine hesaplanır.

## 1) Healthy-only U-Net Autoencoder eğitimi

```powershell
python .\train_brisc_healthy_autoencoder.py `
  --dataset_root "C:\Users\Atabey\Downloads\brisc_Dataset2025\brisc2025" `
  --output_dir ".\outputs\brisc_healthy_unet_ae_256_e50" `
  --model unet `
  --image_size 256 `
  --base_channels 32 `
  --batch_size 8 `
  --epochs 50 `
  --lr 0.0001
```

## 2) Sağlıklı test görüntülerinde reconstruction metrikleri

Bağımsız healthy test klasörü 140 görüntü içeriyor:

```powershell
python .\evaluate_brisc_ae_reconstruction_metrics.py `
  --checkpoint ".\outputs\brisc_healthy_unet_ae_256_e50\best_autoencoder.pt" `
  --dataset_root "C:\Users\Atabey\Downloads\brisc_Dataset2025\brisc2025" `
  --eval_folder "classification_task/test/no_tumor" `
  --output_dir ".\outputs\brisc_healthy_unet_ae_256_e50\eval_no_tumor_test"
```

Hocanın 1000 pair fikrine yakın ek analiz için train/no_tumor üzerinde ilk 1000 görüntü:

```powershell
python .\evaluate_brisc_ae_reconstruction_metrics.py `
  --checkpoint ".\outputs\brisc_healthy_unet_ae_256_e50\best_autoencoder.pt" `
  --dataset_root "C:\Users\Atabey\Downloads\brisc_Dataset2025\brisc2025" `
  --eval_folder "classification_task/train/no_tumor" `
  --max_images 1000 `
  --output_dir ".\outputs\brisc_healthy_unet_ae_256_e50\eval_no_tumor_train_1000"
```

Not: Train üstündeki 1000 görüntü generalization kanıtı değildir; ana generalization sonucu `test/no_tumor` 140 görüntüdür.

## 3) Tümörlü görüntülerde reconstruction-error tabanlı segmentasyon

```powershell
python .\evaluate_brisc_ae_anomaly_segmentation.py `
  --checkpoint ".\outputs\brisc_healthy_unet_ae_256_e50\best_autoencoder.pt" `
  --dataset_root "C:\Users\Atabey\Downloads\brisc_Dataset2025\brisc2025" `
  --split test `
  --output_dir ".\outputs\brisc_healthy_unet_ae_256_e50\anomaly_seg_test"
```
