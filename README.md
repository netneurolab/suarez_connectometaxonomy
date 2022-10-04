# A connectomics-based taxonomy of mammals

## "What's in this repository?"

This repository contains code for the analysis presented in the preprint "[A connectomics-based taxonomy of mammals](https://doi.org/10.1101/2022.03.11.483995)" by Laura E. Suarez, Yossi Yovel, Martijn P. van den Heuvel, Olaf Sporns, Yaniv Assaf, Guillaume Lajoie and Bratislav Misic.

We revised previously morphologically and genetically established taxonomies from a connectomics perspective. Specifically, we investigated the extent to which inter-species differences in the organization of connectome wiring conform to these traditional taxonomies.

We've tried to document the various aspects of this repository with several README files, so feel free to jump around and check things out.

## "Just let me run the things!"

Itching to just run the analyses? You'll need to make sure you have installed the appropriate software packages and have downloaded the appropriate data files.

1. Git clone the [suarez_connectometaxonomy](https://github.com/netneurolab/suarez_connectometaxonomy.git) repository. To do so, in the command line type:

```bash
git clone https://github.com/netneurolab/suarez_connectometaxonomy
cd suarez_connectometaxonomy
conda env create -f environment.yml
conda activate suarez_connectometaxonomy
```

2. Download the `data` folder from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7143143.svg)]https://doi.org/10.5281/zenodo.7143143) and place it into the repository's root directory.

3. Follow the step by step given in the README file of the `scripts` folder.

## "I have some questions..."

[Open an issue](https://github.com/netneurolab/suarez_connectometaxonomy/issues) on this repository and someone will try and get back to you as soon as possible!
