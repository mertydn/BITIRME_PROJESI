# STGCN+

## Giriş

STGCN+, uzamsal ve zamansal modül mimarisinde bazı değişikliklerle geliştirdiğimiz bir STGCN çeşididir. 2D iskeletler (HRNet) ile NTURGB+D üzerinde eğitilmiş STGCN+ ve eğitim ayarlı 3D iskeletler sağlıyoruz. Dört modalite için kontrol noktaları sağlıyoruz: Eklem, Kemik, Eklem Hareketi ve Kemik Hareketi. STGCN+ mimarisi açıklanmıştır [tech report](https://arxiv.org/pdf/1801.07455.pdf).

## Referans

```BibTeX
@misc{Yan_Xiong_Lin_2018, title={Spatial temporal graph convolutional networks for skeleton-based action recognition}, url={https://arxiv.org/abs/1801.07455}, journal={arXiv.org}, author={Yan, Sijie and Xiong, Yuanjun and Lin, Dahua}, year={2018}, month={Jan}} 
}
```

## Model

NTURGB+D 120'de eğitilmiş kontrol noktası yayınlıyoruz. Her modalitenin doğruluğu, ağırlık dosyasına bağlıdır.

| Dataset | Annotation | GPUs | Joint Top1 | Bone Top1 | Joint Motion Top1 | Bone-Motion Top1 | Two-Stream Top1 | Four Stream Top1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D 120 XSub | Official 3D Skeleton | 8 | [joint_config](/configs/stgcn+/stgcn+_ntu120_xsub_3dkp/j.py): [83.2] | 87.0 | 87.5 |

**Not**

1. Doğrusal ölçekleme öğrenme oranını kullanıyoruz (**İlk LR ∝ Parti Boyutu**). Eğitim grubu boyutunu değiştirirseniz, ilk LR'yi orantılı olarak değiştirmeyi unutmayın.

