# Machine Learning for Predicting Targeted Protein Degradation

This repository contains my master thesis work done in Spring 2023 at AstraZeneca in Gothenburg.
The final report can be read at this [link](Machine%20Learning%20for%20Predicting%20Targeted%20Protein%20Degradation.pdf).

## Abstract:

_PROteolysis TArgeting Chimeras (PROTACs) are an emerging high-potential therapeutic technology.
PROTACs leverage the ubiquitination and proteasome processes within a cell to degrade a Protein Of Interest (POI). %, for example generated during a viral infection.
Designing new PROTAC molecules, however, is a challenging task, as assessing the degradation efficacy of PROTACs often requires extensive effort, mostly in terms of expertise, cost and time, for instance via laboratory assays.
Machine Learning (ML) and Deep Learning (DL) technologies are revolutionizing many scientific fields, including the drug development pipeline.
In this thesis, we present the data collection and curation strategy, as well as several candidate DL models, for ultimately predicting the degradation efficacy of PROTAC molecules.
In order to train and evaluate our system, we propose a curated version of open source datasets from literature._
_Relevant features such as_ $pDC_{50}$, $D_{max}$ _, E3 ligase type, POI amino acid sequence, and experimental cell type are carefully organized and parsed via a Named Entity Recognition system based on a BERT model.
The curated datasets have been used for developing three candidate DL models._
_Each DL model is designed to leverage different PROTAC representations: molecular fingerprints, molecular graphs and tokenized SMILES.
The proposed models are evaluated against an XGBoost model baseline and the State-of-The-Art (SOTA) model for predicting PROTACs degradation activity.
Overall, our best DL models achieved a validation accuracy of 80.26% versus SOTA's 77.95% score, and a Area Under the Curve (AUC) validation score of 0.849 versus SOTA' 0.847._

## Methods

### Data Curation

The work is based on two datasets: PROTAC-DB and PROTAC-Pedia. PROTAC-DB contains around four thousand PROTAC complexes and is sourced from scientific literature and web scraping. PROTAC-Pedia includes around one thousand crowd-sourced entries about PROTAC molecules. Numerical features are extracted from the datasets, including molecular SMILES, cell type, E3 ligase, POI sequence, and degradation performance. Cell types and E3 ligases are encoded using ordinal encoders, and the POI sequences are mutated and vectorized using a count vectorizer. PROTAC degradation activity is defined based on $pDC_{50}$ and $D_{max}$ values, and entries are categorized as active or inactive. The data curation process results in each data point containing information about activity, PROTAC SMILES, cell type, E3 ligase, and POI sequence features.

### Proposed Deep Learning Model

The thesis proposes a deep learning model to efficiently encode information about PROTAC-POI-E3 ligase complexes. The model architecture involves different branches that process various features of the complex, including E3 ligase, cell type, and POI $m$-to-$n$-grams. The embeddings produced by these branches are concatenated and fed into a final MLP or Linear head. For PROTAC molecules, different model architectures are proposed, tailored to specific molecular representations derived from SMILES. The SMILES encoder is responsible for generating PROTAC embeddings based on these representations. The following figure provides an overview of the entire model:

![image](https://github.com/ribesstefano/ml-for-protacs/assets/17163014/5adedfd7-9e5e-419b-bc8f-5334c9a41c4f)

Following is instead a figure illustrating the different SMILES encoder architectures:

![image](https://github.com/ribesstefano/ml-for-protacs/assets/17163014/56770101-247f-40d2-928a-2ee85b81ca45)

To evaluate the proposed model with different SMILES encoder architectures, a PyTorch Lightning module was developed following the aforementioned design. The module includes a PyTorch sub-module representing the SMILES encoder. Hyperparameters, such as the number and size of layers, play a crucial role in achieving optimal model performance. Hence, the Optuna optimization framework was used to tune the hyperparameters, employing the Hyperband optimization scheduler and a TPE sampler with 1,000 trials to find the best combinations. As a baseline for comparison, an XGBoost-based model was used, which utilizes weak learners (trees) in an ensemble manner and is well-known for its high performance based on gradient boosting.

The study includes various candidate SMILES encoders and different variations of the baseline model, as listed in the following Table. Certain hyperparameters were fixed and not optimized using Optuna to investigate specific effects on performance.

| Baseline and SMILES Encoders | Design Point                  |
|-----------------------------|-------------------------------|
| XGBoost baseline            | Fingerprints at 1024 bit <br> Fingerprints at 2048 bit <br> Fingerprints at 4096 bit <br> Fingerprints at 1024 bit w/ extra features <br> Fingerprints at 2048 bit w/ extra features <br> Fingerprints at 4096 bit w/ extra features |
| MLP-based                  | Fingerprints at 1024 bit <br> Fingerprints at 2048 bit <br> Fingerprints at 4096 bit |
| GNN-based                  | AttentiveFP <br> GAT <br> GCN <br> GIN |
| Transformer-based          | Roberta_zinc_480m <br> ChemBERTa-zinc-base-v1 <br> ChemBERTa-10M-MTR <br> SSL_Roberta_zinc_480m <br> SSL_ChemBERTa-zinc-base-v1 <br> SSL_ChemBERTa-10M-MTR |

## Results

The following figures show a comparison of the accuracy and AUC scores of the different models.
A dummy score was added for both accuracy and AUC results.
For accuracy, the dummy model always outputs the most frequent class in the dataset, meaning that its accuracy score will be equal to the percentage of most abundant class label.

![image](https://github.com/ribesstefano/ml-for-protacs/assets/17163014/a081a9e7-4636-4fe2-ba7b-731414dea504)

![image](https://github.com/ribesstefano/ml-for-protacs/assets/17163014/35bdf748-4a8d-4efb-8d69-7bf62abb7afc)

The results show that XGBoost models perform competitively across different feature representations, with the highest validation accuracy achieved by XGBoost - MACCS-1024b with extra features (81.58%) and the highest validation ROC AUC achieved by XGBoost - MORGAN-2048b (AUC of 0.861). MLP models also demonstrate promising performance, with MLP - MORGAN-2048b achieving a validation accuracy of 80.26%. However, GNN models and Transformer models, including AttentiveFP, GAT, GCN, GIN, BERT - Roberta_zinc_480m, and BERT - ChemBERTa-zinc-base-v1, show relatively lower performance compared to XGBoost and MLP models. This indicates potential limitations of graph-based and transformer-based approaches for the given curated data.

When comparing with DeepPROTACs, we did not have access to the specific datasets used in DeepPROTACs and found its model not straightforward to use due to its reliance on time-consuming molecular docking outputs as input features. Thus, they were unable to evaluate DeepPROTACs on their test set from PROTAC-Pedia and reported its validation results from the original publication. The validation performance of DeepPROTACs was relatively lower compared to the best models in this study, indicating that the proposed models may be more effective in capturing relevant patterns and characteristics of PROTAC complexes despite the limited data available. This suggests potential limitations in DeepPROTACs' approach or the need for further optimization or simplifications to improve its performance on this specific dataset.

The results show that several models may have experienced overfitting on the limited training data from PROTAC-DB, as indicated by significant drops in accuracy and ROC AUC from the validation set to the test set in PROTAC-Pedia. XGBoost models demonstrated considerable validation-test accuracy drops (average 23.0%), with some models showing relatively small drops and others experiencing larger drops, suggesting varying levels of generalization. MLP models exhibited mixed results, with MLP - MORGAN-2048b showing better generalization (7.6% drop) and MLP - MORGAN-1024b showing a higher likelihood of overfitting (25.6% drop). GNN models generally experienced significant validation-test accuracy drops (average 17.4%), indicating difficulty in generalizing to unseen PROTAC complexes. In contrast, Transformer models demonstrated relatively consistent validation-test accuracy drops (average 14.4%), with SSL\_ChemBERTa-10M-MTR showing the smallest drop (9.7%), suggesting enhanced generalization capability due to the Self-Supervised Learning (SSL) pretraining process. However, overall, SSL training did not guarantee high performance in accuracy or AUC scores compared to non-SSL finetuned models.

## Conclusions

This document outlines the research work focused on developing a deep learning-based system for predicting the degradation activity of PROTACs. The results include dataset analysis, accuracy and AUC scores of candidate models, a comparison with DeepPROTACs, and a discussion of the limited training dataset's impact on model performance. XGBoost demonstrated superior performance compared to deep learning models (MLP, GNN, Transformer) on the PROTAC-DB dataset, likely due to its effective handling of tabular data and limited training data. However, DL models still showed competitive performance. The SOTA model DeepPROTACs exhibited lower performance compared to the proposed models. No clear best model emerged for the test dataset based on PROTAC-Pedia, highlighting the challenge of learning PROTAC degradation activity with limited usable training data. Further exploration of data augmentation, regularization, and additional labeled data could potentially enhance DL model performance in future works.
