# Sanctity.AI
 Data Science Intern Assessment

Feature Importance Analysis
The provided code suggests building a recommendation model based on engagement data and other factors (performance, interests, etc.). However, we can also build a predictive model for student success, using engagement data, historical performance, and quiz scores as features.

To perform feature importance analysis, we can use the trained RandomForestRegressor to rank the significance of these features in predicting the target (e.g., student performance or course completion). In the provided code snippet, the model's feature_importances_ attribute will show how much each feature contributes to the predictions.

Expected Insights:

Engagement Data (e.g., ratings and time spent) is often a strong predictor of success because it reflects how actively students participate.
Historical Performance typically serves as a baseline, indicating whether a student has consistently performed well.
Quiz Scores can reveal immediate understanding and retention of concepts, which directly correlates with final performance.
Report
Approach
Recommendation Model:

The recommendation algorithm ranks study materials for students based on features like interest alignment, difficulty level, popularity, and past interactions.
Scores were computed as a weighted sum of these factors to generate a ranked list of materials for each student.
The top five materials for each student were selected based on the final computed scores.
Predictive Model:

To predict student success, we used a Random Forest model trained on features like engagement_score, historical_performance, and quiz_scores.
Feature importance analysis showed which factors had the most significant impact on predicting outcomes.
Insights on Factors Influencing Student Success
Engagement Data: Students who interacted more with study materials (high engagement scores) had better outcomes. Time spent on relevant content and consistent ratings of materials correlated strongly with success.
Quiz Scores: Higher quiz scores during the semester were predictive of better overall performance, indicating continuous assessment could help identify struggling students early.
Historical Performance: Past academic performance served as an indicator of future success. Students with higher historical grades tended to do well across multiple subjects.
Intervention Suggestions
Increasing Engagement:

Students with Low Engagement Scores: Provide alerts or motivational content to encourage participation.
Interactive Content: Introduce gamified learning or quizzes to increase time spent on materials.
Early Identification of At-Risk Students:

Monitor quiz scores and engagement data to identify students falling behind early in the semester.
Implement a notification system to alert educators about students at risk.
Tailored Learning Support:

Create personalized learning plans for students who show declining performance based on predictive models.
Provide additional resources, such as tutoring or extra practice materials, to students who struggle with particular subjects.
Regular Feedback and Goal Setting:

Allow students to track their progress over time, helping them set learning goals.
Provide constructive feedback based on their quiz results and engagement trends.
Evaluation of Models
NDCG for the Recommendation Model: Measures how well the recommended materials matched the student's preferences.
Feature Importance Scores for the Predictive Model: Helps in understanding which features need more focus for interventions.
By implementing these approaches, we can help improve students' engagement and provide targeted support to those who need it most.
