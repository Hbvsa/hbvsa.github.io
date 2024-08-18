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

The motivation stems from understanding that the correlation between SNPs and a disease may depend strictly on nonlinear interactions of a few SNPs, among the hundreds present in a dataset. This complexity poses a challenge for traditional statistical methods as analyzing individual SNPs may yield zero correlation with the disease. The correlations can only be accurately assessed when SNPs are considered in combination, and given the exponential increase in the number of combinations as the number of SNPs rises, the developed model provides a valuable alternative to exhaustive methods.

# Solution

I implemented a deep learning classification model for extracting correlations between genetic information and diseases. Following the model training, the learned weights were employed to emphasize the most significant SNPs. The features identified as significant by the model underwent a subsequent filtration process using a chi-square test.
In datasets where the impact of SNPs contributed merely 1% to the variation in the probability of disease for the samples, the model exhibited its capability to accurately identify the relevant combination of significant SNPs, despite the subtle nature of the correlation. Notably, the model was able to outperform the state-of-the-art non-exhaustive solutions/models.

The thesis is available at [Master Thesis](https://hbvsa.github.io/files/Henrique_Sousa_MSc_Thesis.pdf)

---

# Experience gained and results

### Deep Learning Architectures
Through extensive preparatory and literature review, I gained a deep and intuitive understanding of different state-of-the-art deep learning architectures (Transformers, MLPs, CNNs, and RNNs) and their advantages and disadvantages for modeling different types of data.

### Custom Model Development
The model architecture required custom implementation of certain layers which provided experience in manipulating model components and ensuring proper functionality. I also reviewed, applied, and tested all state of the art techniques that might enhance model training, such as batch normalization, dropout, and early stopping.

### Creative Model Development

The proposed model innovates the type of deep learning model used for epistasis detection (Single Nucleotide Polymorphisms interactions detection) by encoding the feature values as vectors using an embedding layer (similar to word embeddings in Transformers). The model also reduces the number of parameters, reduces random feature noise overfit, which is especially important for datasets with high cardinality and low overall feature correlation, and allows for model interpretability by making the first layer of the deep learning model mimic a linear regression. This change in the model architecture was tested by comparing it with the alternative state of the art deep learning model used for epistasis detection which only difference is the feature encoding and first layer. The proposed model was able to outperform the alternative model in all datasets.

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

---

# Detailed Breakdown of the Proposed Deep Learning Model

## Overview

The proposed model is a deep learning neural network designed to detect epistasis (interactive effects) between Single Nucleotide Polymorphisms (SNPs) and their correlation with diseases. The model's architecture is carefully crafted to address the unique challenges of epistasis detection while maintaining interpretability and efficiency.

## Key Components

### 1. Embedding Layer

**Description:**
The first layer of the model is an embedding layer that transforms each SNP value into a vector representation.

**Rationale:**
- Enables the model to learn optimized N-dimensional space representations of SNP values.
- Makes it easier for subsequent layers to learn decision boundaries between disease presence and absence.
- Allows for adjustment of vector values during training, similar to word embeddings in NLP tasks.
- SNPs correlated with disease probability can be differentiated from redundant SNPs in the spatial representation.

**Implementation Details:**
- Each SNP value is first converted into a unique token using the formula: Y = X_i + 3 * i, where X_i is the SNP value and i is the index of the SNP in the sample sequence.
- The embedding layer uses these tokens to retrieve corresponding vectors of dimension 32.

### 2. Attention Layer

**Description:**
This layer assigns importance weights to each SNP vector.

**Rationale:**
- Allows the model to learn which SNPs are most important for disease classification.
- Enables filtering of noise by giving lower importance to irrelevant SNPs.
- Facilitates model interpretability by providing clear importance scores for each SNP.

**Implementation Details:**
- Implemented as a dense layer with one weight per SNP.
- The input to this layer is separated from the main neural network because the importance of each SNP is represented as a single scalar value, independent of the SNP vector features. This separation allows the model to learn how much information from each SNP vector should be passed forward, regardless of the specific values in those vectors.

### 3. Weighted Sum of SNP Vectors

**Description:**
After the attention layer, each SNP vector is scaled by its importance weight and then all vectors are summed element-wise.

**Rationale:**
- Combines information from all SNPs while preserving their learned importance.
- Allows the model to account for epistatic interactions by mixing values from different SNPs.
- Reduces the number of parameters in subsequent layers, making the model more efficient.

**Implementation Details:**
- Custom layer that performs element-wise multiplication of each SNP vector with its attention weight, followed by element-wise addition of all resulting vectors.

### 4. Final Dense Layers

**Description:**
Two dense layers process the summed vector to produce the final classification.

**Rationale:**
- Allows for complex non-linear transformations of the combined SNP information.
- Enables the model to learn high-order interactions between SNPs.

**Implementation Details:**
- Two dense layers with 32 neurons each.
- LeakyReLU activation function used for both layers.
- Final output neuron uses sigmoid activation for binary classification.

### 5. Batch Normalization

**Description:**
Applied to both the summed SNP vector and the outputs of the final dense layers.

**Rationale:**
- Stabilizes the learning process and reduces the number of training epochs required.
- Allows the model to be less sensitive to poor initialization of weights.

**Implementation Details:**
- Applied after the weighted sum operation and after each dense layer.

## Model Training and Interpretation

### Training Process

- The model is run multiple times on each dataset, each time with a different random initialization of weights. This approach is crucial for mitigating the model's sensitivity to initial weights, which is especially important for datasets with subtle epistatic effects.
- For each dataset run, the model is trained for a fixed number of epochs since early stopping is unreliable in this type of datasets.
- The model's performance is evaluated based on its ability to identify the correct SNPs involved in epistatic interactions, rather than traditional classification metrics on the predictions of the target class.
- Key evaluation metrics include:
  - Recall: The percentage of datasets in which the epistatic interaction was correctly identified.
  - Precision: The percentage of SNP combinations proposed by the model that are actually part of the true epistatic interaction.
  - F-score: A combined measure of precision and recall.
- The model continues to be run with different initializations until it successfully detects a significant epistatic interaction of the desired order, or until it reaches a predefined maximum number of attempts.

### Interpretation of Results

- The weights learned by the attention layer directly represent the importance of each SNP.
- This allows for easy identification of significant SNPs without need for additional computations post-training. This is important for efficiency when compared to interpretability through perturbation based methods.
- The model's architecture resembles a linear model, making it inherently interpretable while still capturing complex non-linear relationships.

## Filtering Stage for Epistasis Detection

- SNPs are filtered based on their attention values, using standard deviations from the mean as a threshold.
- Filtered SNPs undergo statistical testing (chi-square and conditional chi-square tests) to identify significant epistatic interactions.
- This two-stage approach (deep learning followed by statistical testing) allows for efficient detection of epistasis, even in high-dimensional data.

---
