# scripts

This directory contains the analytic scripts comprising the backbone of the manuscript.
If you read something in the manuscript and have a question about the methodology or implementation chances are you can find the answer in one of these files.

## `1_spectral_&_topological_feats`

- [`spectral_features.py`](./1_spectral_&_topological_feats/spectral_features.py):
  Estimates the normalized Laplacian eigenspectra of the connectivity matrices, as well as their kernel density estimation (kde). The generated eigenspectra are saved to `raw_results/res_100/eig.csv` and `raw_results/res_100/eig_kde.csv`.
- [`topological_features.py`](./1_spectral_&_topological_feats/topological_features.py):
  Estimates the (average and standard deviation) local and global network features of the connectivity matrices. Network features are saved to `raw_results/res_100/df_props.csv`.
- [`eigenfunctions.py`](./1_spectral_&_topological_feats/eigenfunctions.py):
  Contains the functions to estimate the normalized Laplacian of the connectivity matrices and their Gaussian kernel density approximation.

  You can run these scripts from the root of the repository using the command:

  ```bash
  python scripts/1_spectral_&_topological_feats/spectral_features.py
  python scripts/1_spectral_&_topological_feats/topological_features.py  
  ```

## `2_spectral_&_topological_distance`

- [`distance_matrix.py`](./empirical/fetch_hcp_myelin.py):
  Estimates a matrix of pairwise cosine distance measures between all species, including replicas. Distances are calculated based on the spectral and topological features estimated in the previous step. Distance matrices are saved to `raw_results/res_100/spectral_distance.csv` and `raw_results/res_100/topological_distance.csv`.
  It also estimates topological distance matrices using different subsets of topological features. These matrices are saved to `raw_results/res_100/top_{scale}_{conn}_dist.csv`, where {scale} can be either 'local' or 'global', and {conn} can be either 'wei' (for weighted) or 'bin' (for binary), depending on the subset of features.
- [`replicas_control.py`](./2_spectral_&_topological_distance/replicas_control.py):
  Controls for the fact that some species have multiple replicas by randomly selecting one sample per species in the distance matrix. This procedure is repeated 10000 times and it returns an average (across iterations) inter-species (spectral and topological) distance matrix. Distance matrices are saved to `raw_results/res_100/avg_spectral_distance.csv` and `raw_results/res_100/avg_topological_distance.csv`.
  The same procedure is applied to the different topological distance matrices, and results are saved to `raw_results/res_100/avg_top_{scale}_{conn}_dist.csv`, where {scale} can be either 'local' or 'global', and {conn} can be either 'wei' (for weighted) or 'bin' (for binary), depending on the subset of features.

  You can run these scripts from the root of the repository using the command:

  ```bash
  python scripts/2_spectral_&_topological_distance/distance_matrix.py
  python scripts/2_spectral_&_topological_distance/replicas_control.py  
  ```

## `3_nulls`

- [`generate_nulls.py`](./3_nulls/generate_nulls.py):
  Generates 1000 randomly rewired nulls that preserve node degree sequence and distribution of the empirical connectivity matrices. Generated nulls are saved to `raw_results/res_100/nulls/nulls_{i}.npy`, where {i} indicates the file name of each sample. This script uses parallelization to speed the computation.
- [`topological_feats_nulls.py`](./3_nulls/topological_feats_nulls.py):
  Estimates the (average and standard deviation) local and global network features of the nulls. Network features are saved to `raw_results/res_100/nulls/{i}_null_props.csv`, where {i} indicates the file name of each sample. This script uses parallelization to speed the computation.

  You can run these scripts from the root of the repository using the command:

  ```bash
  python scripts/3_nulls/generate_nulls.py
  python scripts/3_nulls/topological_feats_nulls.py  
  ```

## `4_density_control`

- [`density_control.py`](./4_density_control/density_control.py):
  Regresses out the effects of network density on the local and global topological features of the connectivity matrices by fitting either a linear or an exponential model, and replacing each regressed feature with the residuals of the fitted model. The new topological features are saved to `raw_results/res_100/df_props_reg.csv`.
- [`distance_matrix.py`](./4_density_control/distance_matrix.py):
  Estimates a matrix of pairwise cosine distance measures between all species, including replicas. Distances are calculated based on the topological features after controlling for network density. Distance matrices are saved to `raw_results/res_100/reg_top_{scale}_{conn}_dist.csv`, where {scale} can be either 'local' or 'global', and {conn} can be either 'wei' (for weighted) or 'bin' (for binary), depending on the subset of features.
- [`replicas_control.py`](./4_density_control/replicas_control.py):
  Controls for the fact that some species have multiple replicas by randomly selecting one sample per species in the distance matrix. This procedure is repeated 10000 times and it returns an average (across iterations) inter-species topological distance matrix. Distance matrices are saved to `raw_results/res_100/avg_reg_top_{scale}_{conn}_dist.csv`, where {scale} can be either 'local' or 'global', and {conn} can be either 'wei' (for weighted) or 'bin' (for binary), depending on the subset of features.
- [`curve_fitting.py`](./4_density_control/curve_fitting.py):
  Contains the functions to fit and select a regression model (either linear or exponential) that takes network density as explanatory variable and each topological features as response variable.

  You can run these scripts from the root of the repository using the command:

  ```bash
  python scripts/4_density_control/density_control.py
  python scripts/4_density_control/distance_matrix.py  
  python scripts/4_density_control/replicas_control.py  
  ```

## `5_rich_club_&_community_structure`

- [`community_structure.py`](./5_rich_club_&_community_structure/community_structure.py):
  Assesses the community structure of the connectivity matrices across a range of resolution parameters. Outputs are saved to `raw_results/res_100/community_detection/`.
- [`rich_club_structure.py`](./5_rich_club_&_community_structure/rich_club_structure.py):
  Assesses the rich-club structure of the connectivity matrices. Outputs are saved to `raw_results/res_100/rich_club/`.

  You can run these scripts from the root of the repository using the command:

  ```bash
  python scripts/5_rich_club_&_community_structure/community_structure.py
  python scripts/5_rich_club_&_community_structure/rich_club_structure.py  
  ```

## `6_figures`

Once you have run all the previous analyses, you can now generate the figures of the manuscript using the command:

```bash
python scripts/6_figures/F{fig_num}_{fig_name}.py
```

Replace {fig_num} by the number of the figure, and {fig_name} by the name of the figure.

If you just want to generate the figures of the manuscript without having to run all the analyses, you can download the `raw_results` folder from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6382386.svg)](https://doi.org/10.5281/zenodo.6382386) and place it into the `suarez_connectometaxonomy` repository's root directory. After that, run the above command for each figure.
