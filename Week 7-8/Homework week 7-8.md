Question 1

The Scenario: The "Stream-All" Retention Engine 

You are a data scientist at Stream-All, the subscription video service we have explored over the last two weeks. You are building a model to be run on the 1st of every month. Its goal is to identify users who are likely to cancel their subscription by the end of that month so that marketing can send them a "Stay & Save" discount code on the 2nd. 

The following table represents the data available in your warehouse. You are looking at a historical training slice from October 2025. To maintain as-of thinking, your model must only use information that would have been known on October 1st, 2025. 
 
The Dataset 

Target Variable (The Goal): is_churned_by_oct_31 (1 if the user cancelled by the end of the month, 0 if they stayed). 

Column: user_id 

Unique identifier for the customer. 

Column: avg_minutes_watched_sept 

The average daily minutes the user watched in the month of September. 

Column: last_login_date 

The date the user last opened the app. 

Column: support_ticket_count_oct 

Total number of customer support tickets filed by the user during the month of October. 

account_age_days 

Column: How many days the account has been active as of the calculation date. 

subscription_plan_level 

Column: The tier of service (Basic, Standard, Premium) the user is currently on. 

cancellation_survey_score 

Column: A 1-10 score from a survey asking "How likely are you to recommend us?" filled out at the time of account closing. 

total_billing_errors_sept 

Column: Number of times a payment failed or was contested in September. 

last_payment_status 

Column: The status of the most recent payment (e.g., Success, Pending, Failed). 
 
Task 

Your must categorize each column as Safe to Use, Temporal Leakage (Discard), or Ambiguous (Requires Investigation) and provide a written justification for each. 

Question 2 

The Scenario: The "Swift-Lend" Post-Mortem 

You have been hired to investigate why the Swift-Lend automated approval engine, which was highly profitable in 2024, is now causing the company to lose millions in 2026. Your first task is to generate the historical and current data files to compare the relationship between Income, Spending, and Loan Outcomes. 

The Data

Use two files: training_data_2024.csv and serving_data_2026.csv. 
 
The Student Assignment 
After opening the CSV files in a spreadsheet or text editor, answer the following: 
1. The Hidden Formula: Compare the disposable_income, luxury_spending, and financial_health_score columns in both files. What is the exact mathematical relationship being used to calculate the score in both 2024 and 2026? 
2. The Environment Shift: In the 2024 data, high luxury spending generally occurs alongside high income. In 2026, look at users 901 and 902. What has happened to the relationship between income and spending? 
3. The Failure Diagnosis: Identify the user in the 2026 data who has the highest "Financial Health Score." Did they default? Based on your answer to Question 2, explain why the model's scoring logic is now a "Strategic Hallucination." 
4. The Corrective Action: If you were to update the logic for financial_health_score to make it Temporally Valid for the 2026 environment, how would you change the formula using the available columns? 

Question 3 

Part 1 

In this activity, you are the Lead AI Strategist overseeing the launch of two critical models we’ve explored this week: Swift-Lend (an instant-approval loan engine) and Stream-All's Churn-Buster (a subscriber retention system). 

Your technical teams have provided two sets of code for each model: 

The Research Pipeline (Python): Used to train the model on historical data. 

The Production API (SQL/Java): The live code that will make decisions in real-time. 

The Problem 

Early testing shows a mirage of success. The models performed with 95% accuracy in the laboratory, but during a small-scale live pilot, accuracy plummeted to 60%. The engineering team insists the code is identical in both environments, yet the results suggest a fundamental failure in how the models handle time. 

Your Mission 

You must use a Large Language Model (LLM) to audit these two codebases. However, the quality of your audit depends entirely on your prompt strategy. You are looking for more than just "bugs" or "typos." You are looking for: 

Temporal Leakage: Does the training code "peek" at outcomes that haven't happened yet? 

Knowledge Horizons: Does the training code use "Event Time" (perfect data) while the live system is stuck with "System Time" (stale/delayed data)? 

Autopsy Logic: Is the model accidentally using "symptoms" of a crash to predict the crash? 
 
The Exercise

Below are five potential prompts (shuffled) that you could provide to an LLM to perform this audit. 
Your Task: 

Look at each prompt in detail. Then, explore: 

What are the advantages and disadvantages of each prompt?  

Justify your ranking. Getting the exact right answers is less important than writing a good justification. 
 
Prompts 

Prompt V: "Audit the Python and SQL files to ensure the feature definitions are identical. Check that the 'Monthly_Usage' window is exactly 30 days in both, verify that null values are handled with the same defaults, and confirm that the rounding precision for currency matches. Report any discrepancies in the arithmetic." 

Prompt W: "Check these scripts for instances where the training logic uses Event Time (when a user acted) while the serving logic relies on System Time (when the database updated). Flag any features that would be unavailable or 'stale' at the exact second the live model must fire its prediction." 

Prompt X: "Please review the attached training and serving codebases for any potential issues. As a senior engineer, I need you to ensure that the logic is consistent, the variables are correctly mapped, and the overall architecture is robust enough for a production launch. Provide a comprehensive list of any bugs or inconsistencies you find." 

Prompt Y: "Analyze these files to find where the training code performs an 'autopsy' on the data. Look for features that are symptoms of the outcome—things only known after the fact—rather than predictors. Identify where the model assumes the market is 'stationary' when the production data source has actually shifted." 

Prompt Z: "I am providing a Python script and a Java API. Please compare the formulas for 'Credit_Score' and 'Debt_Ratio.' Check if the weights and coefficients used in the training model are the same ones being applied in the live production environment. Let me know if the equations don't match up." 
 
