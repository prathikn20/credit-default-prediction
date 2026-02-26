# Credit Card Default Prediction

## Project Overview
The goal of this project is to predict which credit card clients are more likely to default. By accurately identifying high-risk clients, financial institutions can proactively manage credit limits and mitigate significant losses. 
The cost of a False Negative (failing to identify a defaulter) is higher than a False Positive (lowering a good customer's credit limit). Therefore, the model is optimized for **Recall** and **ROC-AUC** as opposed to the standard accuracy metric.

## Skills & Tools Used
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Techniques:** Exploratory Data Analysis (EDA), One-Hot Encoding, Handling Class Imbalances (`class_weight`), Hyperparameter Tuning, Feature Importance Analysis.
* **Models Tested:** Decision Trees, Random Forests

## Results
After testing and tuning models via custom hyperparameter-tuning functions, the **Random Forest Classifier** performed the best. 
* **Recall (Class 1):** 61.6% (Successfully caught the majority of actual defaulters)
* **ROC-AUC Score:** 0.774
* **Precision:** 46.7%

## Business Insights
Through Feature Importance extraction, the model revealed that demographic factors (Age, Education, Marriage) were relatively insignificant. The strongest predictors of a future default, in order of importance, were:
1. The repayment status in the most recent month (`PAY_0`).
2. The repayment status in the prior month (`PAY_2`).
3. Having a history of payment delays exceeding 9 months.

**Recommendation:** Financial Institutions should ensure that they automatically flag accounts that fall a month or 2 behind in their payments to reevaluate the accounts and risk-profile, before a full default on the payments occurs. 
