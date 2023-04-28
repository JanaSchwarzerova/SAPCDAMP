# concept_drift_in_metabolomics_predictions
In this repository we offer semi-autonomic pipeline for enhaced metabolomics predictions. 

The whole pipeline is divided into three sections ([A], [B], [C]) at the end of which the evaluation parameters of individual prediction models are shown and summarized.

Section A represents selected conventional methods that are frequently used in predictive metabolomic modelling. 

Section B three distinct, systematically divided parts:
b1] includes concept drift detectors DDM and EDDM. 
b2] represents identification of confouding factors (the need to add biological knowledge based on measured clinical data) – Due to this step, the pipeline becomes semi-automatic
b3] based on b2 assumption, we tested two hypotheses included in our semi-autonomic pipeline for creating enhanced metabolomics classifiers. The first is based on proper segmentation of the input data. The segmentation is thought as selection a region of interest. That is, we eliminated the dataset of individuals who were less than 26 years old.
The second hypotheses rely on standardization dataset using feature scaling, thus, training and testing data is not lost as in segmentation techniques. Scaling was done based on the time-threshold found, in our case i.e. 26 year.

Section C presents new modelling of classifiers using new data inputs to improve accuracy of classifiers.  
