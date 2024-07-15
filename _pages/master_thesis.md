---
layout: archive
title: "Master Thesis: Deep Learning for SNP-Disease Correlation Analysis"
permalink: /master_thesis/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}


# Challenge

This thesis addresses a critical challenge in genetic research: identifying correlations between Single Nucleotide Polymorphisms (SNPs) and diseases, particularly when these correlations depend on nonlinear interactions among multiple SNPs. 

The motivation for this thesis stems from understanding that the correlation between Single Nucleotide Polymorphisms (SNPs) and a disease may depend strictly on
nonlinear interactions of a few SNPs, among the hundreds present in a dataset. This complexity poses a challenge for
traditional statistical methods as analyzing individual SNPs may yield zero correlation with the disease. The
correlations can only be accurately assessed when SNPs are considered in combination and given the exponential
increase in the number of combinations as the number of SNPs rises, the developed model provides a valuable
alternative to exhaustive methods.

The solution was the implementation of a deep learning classification model for extracting correlations between genetic information
and diseases. Following the model training, the learned weights were employed to emphasize the most significant
SNPs. The features identified as significant by the model underwent a subsequent filtration process using a
chi-square test. In datasets where the impact of SNPs contributed merely 1% to the variation in the probability of
disease for the samples, the model exhibited its capability to accurately identify the relevant combination of
significant SNPs, despite the subtle nature of the correlation. The model was able to outperform the state of the
art non exhaustive solutions/models.

The thesis is available at [Master Thesis](https://hbvsa.github.io/files/Henrique_Sousa_MSc_Thesis.pdf)

# Experience gained and results

### Deep Learning Architectures
The extensive preparatory and literature review phase allowed me to gain a deep and intuitive understanding behind the different state of the art deep learning architectures (Transformers, MLPs, CNNs and RNNs) and their advantages and disadvantages for modelling different types of data.

### Custom Model Development
The model architecture required custom implementation of certain layers which provided experience in manipulating model components and ensuring proper functionality. During the thesis I also had to review, apply and test all state of the art techniques which might enhance model training such as batch norm, dropout, early stopping.

### Creative Model Development

The proposed model innovates the type of deep learning model used for epistasis detection (Single Nucleotide Polymorphisms interactions detection) by enconding the features values as vectors using an embedding layer (as in words for a Transformer). The model also reduces the number of parameters, reduces random feature noise overfit, which is especially important for datasets with high cardinality and low overall feature correlation, and allows for model interpretability by making the first layer of the deep learning model mimic a linear regression. This change in the model architecture was tested by comparing it with the alternative state of the art deep learning model used for epistasis detection which only difference is the feature encoding and first layer. The proposed model was able to outperform the alternative model in all datasets.

### Handling non linear and subtle feature-target correlations datasets

In this specific problem the correlation between the features and target variable (disease) can be really low to the point a good model - one that detected the significant features - will not demonstrate a notably high accuracy, with even 51% potentially indicating a good score. There were also datasets where the correlation between the features and target variable are exclusively non linear leading to abrupt learning patterns where the training loss would remain stable for many epochs before suddenly decreasing, or in some cases, failing to learn the pattern entirely with a given weight initialization.

### Adaptating the evaluation of deep learning models for problematic datasets

These combination of factors required me to develop a new way to do model evaluation. The models were evaluated by running them with different initializations and measuring the number of required initializations per dataset to find the significative features. Given the unpredictable learning process early stopping was not a option and a fix number of epochs was used. After each training session, the model would suggest the most significative features and was evaluated against the ground truth, in this case the Single Nucleotide Polymorphisms known to be correlated with the disease.

### Outperforming the state of the art approaches

The proposed model was able to surpass the performance of all the non exhaustive state of the art
methods including an alternative deep learning model on the marginal datasets tested (datasets where there is some individual feature correlation). 

The experimental results also showed the proposed model was capable of achieving 99% recall (detection rate)
in non marginal effect datasets (purely non linear) of 3rd order interactions while the alternative deep learning model and
two of the top performing state of the art non exhaustive methods in marginal datasets achieved a recall
of 0%.
