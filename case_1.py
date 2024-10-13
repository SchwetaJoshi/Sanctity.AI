import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Student Data
students = pd.DataFrame({
    'student_id': range(1, 101),
    'course': np.random.choice(['CS', 'ME', 'EE'], 100),
    'year': np.random.choice([1, 2, 3, 4], 100),
    'performance': np.random.uniform(50, 100, 100),
    'interests': np.random.choice(['AI', 'Blockchain', 'Data Science'], 100)
})

# Study Material Data
materials = pd.DataFrame({
    'material_id': range(1, 51),
    'subject': np.random.choice(['AI', 'Blockchain', 'Data Science'], 50),
    'difficulty': np.random.randint(1, 6, 50),  # Scale: 1 to 5
    'popularity': np.random.uniform(1, 5, 50),  # Scale: 1 to 5
    'content_length': np.random.randint(100, 1000, 50)  # Words count
})

# Engagement Data
engagement = pd.DataFrame({
    'student_id': np.random.randint(1, 101, 200),
    'material_id': np.random.randint(1, 51, 200),
    'rating': np.random.uniform(1, 5, 200)  # Ratings: 1 to 5
})

# Calculate average ratings per student per subject
student_subject_ratings = (
    engagement
    .merge(materials[['material_id', 'subject']], on='material_id')
    .groupby(['student_id', 'subject'])['rating']
    .mean()
    .unstack(fill_value=0)  # Fill missing subjects with 0
)
# Ensure that both matrices have the same subjects (columns)
subject_list = ['AI', 'Blockchain', 'Data Science']
student_subject_ratings = student_subject_ratings.reindex(columns=subject_list, fill_value=0)


# Normalize data for consistent scoring
scaler = MinMaxScaler()

# Normalize performance
students['performance_normalized'] = scaler.fit_transform(students[['performance']])

# One-hot encode interests
interests_encoded = pd.get_dummies(students['interests'], prefix='interest')

# Normalize material popularity
materials['popularity_normalized'] = scaler.fit_transform(materials[['popularity']])

# Normalize difficulty
materials['difficulty_normalized'] = scaler.fit_transform(materials[['difficulty']])

# Scoring Function
def calculate_score(student, material):
    # Interest Matching Score (1 if interest matches, 0 otherwise)
    interest_score = 1 if student['interests'] == material['subject'] else 0

    # Performance Adjustment (use normalized performance)
    performance_score = student['performance_normalized']

    # Engagement Weight (assume higher difficulty is preferred for higher performance)
    difficulty_score = 1 - abs(material['difficulty_normalized'] - performance_score)

    # Material Popularity
    popularity_score = material['popularity_normalized']

    # Composite Score
    score = (0.4 * interest_score + 0.3 * performance_score + 
             0.2 * difficulty_score + 0.1 * popularity_score)
    
    return score

# Generating Recommendations
def recommend_materials(student, materials, top_n=5):
    # Calculate score for each material
    materials['score'] = materials.apply(lambda mat: calculate_score(student, mat), axis=1)
    
    # Sort materials by score in descending order and get top_n
    top_recommendations = materials.sort_values(by='score', ascending=False).head(top_n)['material_id'].tolist()
    
    return top_recommendations

# Get Top 5 Recommendations for Student 1
student_1 = students.iloc[0]  
top_materials_for_student_1 = recommend_materials(student_1, materials)
print(f"Top 5 Recommendations for Student 1: {top_materials_for_student_1}")

# Evaluation with NDCG
def ndcg_score(true_list, pred_list, k=5):
    DCG = sum([int(pred_list[i] in true_list) / np.log2(i + 2) for i in range(k)])
    IDCG = sum([1.0 / np.log2(i + 2) for i in range(min(len(true_list), k))])
    return DCG / IDCG if IDCG > 0 else 0

# Evaluate with Random True List
true_materials = [1, 5, 10, 15, 20]
ndcg = ndcg_score(true_materials, top_materials_for_student_1)
print(f"NDCG Score: {ndcg}")

# Student Performance Distribution
plt.figure(figsize=(10, 6))
sns.histplot(students['performance'], bins=10, kde=True, color='blue')
plt.title('Distribution of Student Performance')
plt.xlabel('Performance')
plt.ylabel('Frequency')
plt.show()

# Engagement Trends by Subject
avg_engagement_by_subject = (
    engagement
    .merge(materials[['material_id', 'subject']], on='material_id')
    .groupby('subject')['rating']
    .mean()
)

plt.figure(figsize=(10, 6))
avg_engagement_by_subject.plot(kind='bar', color='green')
plt.title('Average Engagement by Subject')
plt.xlabel('Subject')
plt.ylabel('Average Rating')
plt.show()

