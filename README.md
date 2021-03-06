# Sampling Methods to speed up Clustering Algorithms β­

## K-Means Clustering π₯

- KDD Biotrain Dataset

![alt text](images/kdd/kmeanscoresets.png?raw=true)

![alt text](images/kdd/kmeansvol_lev.png?raw=true)

- Worms Dataset

![alt text](images/worms/kmeansall.png?raw=true)

## Bisecting K-Means Clustering π₯

- KDD Biotrain Dataset

![alt text](images/kdd/bisectingcoresets.png?raw=true)

![alt text](images/kdd/bisectingvol_lev.png?raw=true)

- Worms Dataset

![alt text](images/worms/bisectingall.png?raw=true)

## K-Center Clustering π₯

- KDD Biotrain Dataset

![alt text](images/kdd/kcenterall.png?raw=true)

- Worms Dataset

![alt text](images/worms/kcenterall.png?raw=true)

## K-Medoids Clustering π₯

- K-medoids clustering is performed on artificial dataset because of its expensive runtime
- Artificial Dataset (dataset visualization)

![alt text](images/artificial.png?raw=true)

- All sampling methods 

![alt text](images/kmedoids.png?raw=true)


## Directory Structure π

- The directory structure is as follows -
- There are 4 directories for each clustering algorithm

```
β   Averaged_Data_collection.xlsx
β   README.md
β
ββββbisecting
β       kdd_coresets.py
β       kdd_leverage.py
β       kdd_uniform.py
β       kdd_volume.py
β       worms_coresets.py
β       worms_leverage.py
β       worms_uniform.py
β       worms_volume.py
β
ββββimages
β   β   artificial.png
β   β   kmedoids.png
β   β
β   ββββkdd
β   β       bisectingcoresets.png
β   β       bisectingvol_lev.png
β   β       kcenterall.png
β   β       kmeanscoresets.png
β   β       kmeansvol_lev.png
β   β
β   ββββworms
β           bisectingall.png
β           kcenterall.png
β           kmeansall.png
β
ββββkcenter
β       kdd_coresets.py
β       kdd_leverage.py
β       kdd_uniform.py
β       kdd_volume.py
β       worms_coresets.py
β       worms_leverage.py
β       worms_uniform.py
β       worms_volume.py
β
ββββkdd
β       bio_train.dat
β       kdd_reduced.pickle
β       kdd_reduced_1k.pickle
β       kdd_reduced_20k.pickle
β       kdd_reduced_30k.pickle
β       kdd_reduced_40k.pickle
β
ββββkmeans
β       kdd_coresets.py
β       kdd_leverage.py
β       kdd_uniform.py
β       kdd_volume.py
β       worms_coresets.py
β       worms_leverage.py
β       worms_uniform.py
β       worms_volume.py
β
ββββkmedoid
β       artificial_all.py
β
ββββworms
        README.txt
        worms_2d.png
        worms_2d.txt
        worms_64d.txt
        worms_reduced.pickle
        worms_reduced_20k.pickle
        worms_reduced_30k.pickle
        worms_reduced_40k.pickle
```

## Results π₯

- Lightweight coresets outperform all other sampling techniques in each combination ofdatasets and clustering algorithms.

- For the KDD dataset, Leverage sampling performs better than Volume sampling, in eachcombination of KDD dataset and clustering algorithms.

- For the Worms dataset, Volume sampling performs better than Leverage sampling, in eachcombination of Worms dataset and clustering algorithms.

- Although lightweight coresets were designed for kmeans, they show a good performance onthe kcenters algorithm as well, beating rest of the sampling techniques.

