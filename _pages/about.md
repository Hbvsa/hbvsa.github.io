---
permalink: /
title: "Welcome to my website"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

# Selected Experience
## Education Background

I have a Master of Science in Computer Science and Engineering from Instituto Superior Técnico, Lisboa, during which I studied machine learning and artificial intelligence topics in multiple courses such as Data Science, Machine Learning, PLIDM (Planning, Learning and Intelligent Decision Making) basically Reinforcement Learning and Markov Decision Processes, Artificial Intelligence in Games, Autonomous Agents and Multi-Agent Systems, and Natural Language Processing. I have also completed a [online course](https://coursera.org/share/3a02f88e77a05ca31ecbe596b30a2ccf) which I recommend for anyone wanting to get a good understanding of the math behind machine learning.

## Master Thesis - Deep Learning project to extract correlation between genetic factors and disease

### Thesis description
The motivation for my thesis starts by understanding that the correlation between Single Nucleotide Polymorphisms (SNPs) and a disease may hinge on a strictly
nonlinear combination of two or three SNPs, among the hundreds present in a dataset, which poses a challenge for
traditional statistical methods since analyzing individual SNPs might yield zero correlation with the disease. The
correlations can only be accurately assessed when SNPs are considered in combination and given the exponential
increase in the number of combinations as the number of SNPs rises, the developed model provides a valuable
alternative to exhaustive methods.

The solution was the implementation of a deep learning classification model for extracting correlations between genetic information
and diseases. Following the model training, the learned weights were employed to emphasize the most significant
SNPs. The features identified as significant by the model underwent a subsequent filtration process using a
chi-square test. In datasets where the impact of SNPs contributed merely 1% to the variation in the probability of
disease for the samples, the model exhibited its capability to accurately identify the pertinent combination of
significant SNPs, despite the subtle nature of the correlation. The model was able to outperform the state of the
art non exhaustive solutions/models.

To check all the information go to [Master Thesis](https://hbvsa.github.io/files/Henrique_Sousa_MSc_Thesis.pdf)

### Experience gained

The preparation and background review phase allowed me truly have a deep and intuitive understanding behind the different state of the art deep learning architectures (Transformers, MLPs, CNNs and RNNs) and their advantages and disadvantages based on the type of data we are trying to model.

The model had some layers which had to be customly implemented which gave me experience in manipulating model layers and making sure the model is working properly. During the thesis I also had to review, apply and test all state of the art techniques which might help the model training such as batch norm, dropout, early stopping, etc ...

In this specific problem the correlation between the features and target variable (disease) can be really low to the point a good model, meaning, a model that detected the significative features, is not noticeable in the classification accuracy, 51% might be a good score. There were also datasets where the correlation between the features and target variable are exclusively non linear which made the model recognize the pattern in sudden manner, meaning, the training loss would stay steady for a lot of epochs until suddenly the model learned the pattern or it never learn the pattern at least with that weight initialization.

These combination of factors required me to develop a new way to evaluate a good model. The models were evaluated by running them with different initializations and measuring the number of required initializations per dataset to find the significative features. Since it was impossible to know when the model had learned the pattern, a fix number of epochs was used. After each training session for a single initialization the model would suggest the most significative features and was evaluated by comparing these with the ground truth, in this case the Single Nucleotide Polymorphisms known to be correlated with the disease.

The best model setup would be the one with lowest average number of required seed initializations per dataset.

## Deep learning projects
