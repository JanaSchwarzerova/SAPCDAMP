# Semi-Automated Pipeline using Concept Drift Analysis for improving Metabolomics Predictions (SAPCDAMP)

In this repository, we offer a semi-automated pipeline for enhanced metabolomics predictions. This pipeline leverages various machine learning techniques, concept drift detection methods, and data preprocessing strategies to improve predictive accuracy.

The whole pipeline is divided into three sections ([A], [B], [C]), at the end of which the evaluation parameters of individual prediction models are shown and summarized.

## Table of Contents

1. [Introduction](#introduction)
2. [How to Use](#how-to-use)
3. [Sections Overview](#sections-overview)
   - [Section A](#section-a)
   - [Section B](#section-b)
     - [b1] Concept Drift Detection
     - [b2] Identification of Confounding Factors
     - [b3] Hypothesis Testing for Enhanced Classifiers
   - [Section C](#section-c)
4. [Evaluation](#evaluation)

## Introduction

The SAPCDAMP pipeline is designed for metabolomics prediction using multiple classifiers. We leverage concept drift detection techniques like DDM (Drift Detection Method) and EDDM (Enhanced Drift Detection Method) to monitor changes in data distributions over time. This allows us to fine-tune predictive models continuously.

The pipeline aims to improve the accuracy and robustness of metabolomics predictions by:

- Detecting concept drift.
- Identifying potential confounding factors.
- Testing hypotheses related to segmentation and data standardization.
- Using multiple classifiers and comparing their performance.

## How to Use

### Running the Pipeline on Google Colab

1. **Clone the Repository**
   - Open Google Colab (https://colab.research.google.com/).
   - Clone the repository to your Colab environment using the following command:

     ```python
     !git clone <repository_url>
     ```

     Replace `<repository_url>` with the actual URL of the repository you're working with.

2. **Install Required Dependencies**
   - Install necessary libraries and dependencies. For example, you can use:

     ```python
     !pip install -r requirements.txt
     ```

     Make sure the `requirements.txt` file contains all the necessary libraries (like `pandas`, `numpy`, `sklearn`, etc.).

3. **Upload the Dataset**
   - You will need to upload the datasets for the pipeline. You can do so by using the Colab interface to upload files or by using URLs to directly load the data from external sources. Use the following to upload a dataset:

     ```python
     from google.colab import files
     uploaded = files.upload()
     ```

   Alternatively, if your dataset is hosted online, you can load it directly:

     ```python
     url_Chu_et_al_scaled = "<dataset_url>"
     url_Li_et_al_scaled = "<dataset_url>"
     url_Kar_et_al_scaled = "<dataset_url>"
     ```

4. **Set Up the Pipeline**
   - In your Colab notebook, load and clean the dataset using the provided functions.

     ```python
     data = load_and_clean_data(url_Chu_et_al_scaled)
     ```

5. **Choose Scaling Method**
   - You will be prompted to choose a scaling method. This can be done through a simple input prompt in the notebook.

     ```python
     scaling_method = input("Enter the scaling method (Centering, Autoscaling, Range Scaling, Pareto Scaling, Vast Scaling, Level Scaling, Log Transformation, Power Transformation): ")
     ```

     Based on the input, the dataset will be scaled using the corresponding method. If necessary, ensure that the function `apply_scaling_by_class` accepts and processes this input correctly.

6. **Run the Concept Drift Detection**
   - After preprocessing, run the **Concept Drift Detection** steps for each classifier. For example:

     ```python
     detect_drift_ddm(predicted_Ridge, "RR")
     detect_drift_ddm(predicted_SVR, "SVR")
     detect_drift_ddm(predicted_RF, "RF")
     detect_drift_ddm(predicted_DNN, "DNN")
     ```

7. **Train and Evaluate the Classifiers**
   - Follow the steps to train and test classifiers based on the cleaned and preprocessed data.

     ```python
     model_RF = RandomForestClassifier(n_estimators=100)
     model_RF.fit(met_train_X, met_train_y)
     ```

   - Evaluate the performance of the classifier:

     ```python
     print(f"Random Forest model accuracy on the test set: {model_RF.score(met_test_X, met_test_y)}")
     ```

8. **View Results and Adjust Parameters**
   - After running the pipeline, the evaluation metrics will be printed. Review the results, compare accuracy, precision, and recall across different models.

   - You can adjust parameters such as the **scaling method**, **concept drift detection** settings, or **classifier configurations** and re-run the pipeline for improved results.

## Sections Overview

### Section A: Conventional Methods in Metabolomics

This section involves the selection of conventional methods that are frequently used in predictive metabolomic modeling. These include basic machine learning classifiers like Ridge Regression (RR), Support Vector Regression (SVR), Random Forest (RF), and Deep Neural Networks (DNN). These models are trained and evaluated using preprocessed metabolomics data.

### Section B: Concept Drift Detection and Hypothesis Testing

Section B is divided into three distinct parts:

#### b1] Concept Drift Detection (DDM and EDDM)
Concept drift detection is implemented using two methods:

- **DDM (Drift Detection Method)**
- **EDDM (Enhanced Drift Detection Method)**

These methods help identify whether changes have occurred in the data distribution over time. Concept drift can affect model accuracy, and detecting it early can help adjust or retrain models accordingly.

#### b2] Identification of Confounding Factors
This part focuses on identifying confounding factors using clinical data and biological knowledge. It highlights the need for human expertise to interpret data effectively and adjust the model accordingly, turning the pipeline into a semi-automatic process.

#### b3] Hypothesis Testing for Enhanced Classifiers
This part tests two hypotheses aimed at improving metabolomics classifiers:

1. **Segmentation**: This involves eliminating individuals below a certain age (e.g., 26 years old) to focus on a region of interest and remove outliers that could skew predictions.
2. **Feature Scaling**: Standardizing the dataset to ensure consistency and remove bias due to varying feature ranges. Feature scaling is applied based on a time-threshold (e.g., 26 years).

### Section C: Enhanced Classifiers and Improved Accuracy

Section C implements new classifiers and models that incorporate the concepts tested in Section B. These models use new data inputs to enhance the accuracy of predictions and refine the machine learning models based on findings from the previous sections.

## Evaluation

After running the pipeline, the accuracy, precision, recall, and other evaluation metrics are displayed for each classifier used in the pipeline. These metrics help you understand the effectiveness of each model and provide insight into areas that may need improvement. The models can be further adjusted and fine-tuned based on this feedback to improve prediction performance.

---
**Note**: The repository is designed to run on Google Colab, making it easier to run experiments in a cloud-based environment without worrying about local setups. Make sure to upload the necessary datasets and follow the steps as outlined above.

