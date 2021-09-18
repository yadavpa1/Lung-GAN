# Lung-GANs
This is the code for [Lung-GANs: Unsupervised Representation Learning for Lung Disease Classification using Chest CT and X-ray Images](https://www.researchgate.net/publication/354228470_Lung-GANs_Unsupervised_Representation_Learning_for_Lung_Disease_Classification_Using_Chest_CT_and_X-Ray_Images). Lung-GANs is a deep unsupervised framework to classify lung diseases from chest CT and X-ray images.

## Dataset
- Tuberculosis vs. Healthy: https://mmcheng.net/tb/
- Healthy vs. Sick        : https://mmcheng.net/tb/
- Pneumonia vs. Normal    : https://data.mendeley.com/datasets/rscbjbr9sj/2
- COVID-19 vs. Pneumonia  : https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
- COVID-19 vs. Non-COVID CT & X-ray : https://data.mendeley.com/datasets/8h65ywd2jr/3

## Dependencies
NVIDIA GPU + CUDA CuDNN (CPU mode and CUDA without CuDNN mode are also available but is significantly slower)
- tensorflow  - 1.15.0
- tensorlayer - 1.6.0
- sklearn - 0.20.4
- numpy - 1.16.1

## Usage
Training the GAN
```
python train_lung_gan.py
```

Extracting features
```
python extract_features.py
```

Training classifier
```
python train_classifier.py
```

## Citation
If this code is useful for your research, do cite:
```
@article{yadav2021lung,
  title={Lung-GANs: Unsupervised Representation Learning for Lung Disease Classification Using Chest CT and X-Ray Images},
  author={Yadav, Pooja and Menon, Neeraj and Ravi, Vinayakumar and Vishvanathan, Sowmya},
  journal={IEEE Transactions on Engineering Management},
  year={2021},
  publisher={IEEE}
}
```
