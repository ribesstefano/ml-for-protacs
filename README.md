# Machine Learning for Predicting Targeted Protein Degradation

This repository contains my master thesis work done in Spring 2023 at AstraZeneca in Gothenburg.
The final report can be read at this [link](Machine%20Learning%20for%20Predicting%20Targeted%20Protein%20Degradation.pdf).

Abstract:

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
