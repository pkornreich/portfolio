# Churn Model Interpretation & Business Recommendations Report

## Executive Summary

This report outlines the development and validation of a predictive model to predict customers at high risk of **churn** (attrition) in order to maximize the efficiency of retention efforts. 



### Key Findings

- **Model Performance:** The model that produced the best results, an **XGBoost Classifier**, achieved a Precision of **$76.19\%$** and an overall Accuracy of **$85.93\%$** on the test set, validated by a Cross-Validation Precision score of $0.7670$.

- **Targeting Efficiency:** The model can confidently identify **over 76 out of every 100 customers** it flags, ensuring that retention spending is focused on true churn risks, minimizing wasted marketing budget (False Positives).

- **Coverage:** The model successfully identifies **$68.45\%$** of all customers who will ultimately churn (Recall), providing substantial coverage to prevent customer loss.

- **Primary Drivers:** Churn is overwhelmingly driven by three factors: users on **Contract** (_i.e._, the Month-to-Month plans), **Fiber Optic Internet Service**, and a high frequency of **Technical Support Tickets**.



### Core Recommendation

It is highly recommended to **immediately deploy this model** to generate daily lists of high-risk customers.

Retention strategy should be two-fold:

1. **Targeted Intervention:** Use the high-confidence list to execute cost-effective, high-value retention offers (e.g., discounted long-term contracts).

2. **Operational Improvement:** Focus engineering and service efforts on mitigating dissatisfaction within the **Fiber Optic** customer segment and reducing the necessity for **Tech Support** contact.

## 

## Model Performance Analysis: Translating Metrics

A summary of the model's predictions is shown in the following Confusion Matrix:

| True (Correct) Negatives | False Positives          |
|:------------------------:|:------------------------:|
| 953                      | 80                       |
| False Negatives          | True (Correct) Positives |
| 118                      | 256                      |
