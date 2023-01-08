## VAE-GUP Variational autoencoder based generation of user profiles

This repo implements a method called VAE-GUP, wherein Variational autoencoders are used to 
generate multiple user profiles for retail users. This method aims to improve item and user 
level diversity in recommender systems to improve customer satisfaction. This codebase has been extended from the PyTorch implementation [1] of Variational autoencoders
for collaborative filtering in PyTorch presented in [2],
and also does beta-VAE [3] to use user profiles in collaborative filtering.
This method was tested on an ecommerce company's dataset and MovieLens. To protect the confidentiality of the ecommerce company, we have only included the MovieLens dataset in this repository.

### Dataset

[MovieLens 20M Dataset](https://grouplens.org/datasets/movielens/20m/): This dataset contains rating for movies by users.

The dataset has been created by [vae_data_creation.ipynb](vae_data_creation.ipynb)
Follow the steps in the notebook to create the dataset. 
### Training and validation

Usage: 

```bash
python demo.py train <options>
```

Example: train --n_items 26164

### Results
NDCG@N, ILD@N, TILD@N and aggregate diveristy@N of baseline (Single user profile) vs multiple user profiles for MovieLens 20M
![results_movielens.png](results%2Fresults_movielens.png)

NDCG@N, ILD@N, TILD@N and aggregate diveristy@N of baseline (Single user profile) vs multiple user profiles for proprietary dataset
![results_proprietary_dataset.png](results%2Fresults_proprietary_dataset.png)


## References

1. Cydonia. Variational autoencoders for collaborative filtering in PyTorch,
   Github repository, 2019.
   [github](https://github.com/cydonia999/variational-autoencoders-for-collaborative-filtering-pytorch)

2. Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, Tony Jebara. Variational Autoencoders for Collaborative Filtering,
    The Web Conference (WWW), 2018.  
    [arXiv](https://arxiv.org/abs/1802.05814), [github](https://github.com/dawenl/vae_cf)
     
3. Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, Alexander Lerchner. 
    beta-VAE: Learning basic visual concepts with a constrained variational framework. ICLR, 2017.
    [Paper link](https://openreview.net/forum?id=Sy2fzU9gl)

