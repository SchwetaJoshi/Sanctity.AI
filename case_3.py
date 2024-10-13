import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report

# Student Data
students = pd.DataFrame({
    'student_id': range(1, 201),
    'interests': np.random.choice(['AI', 'Blockchain', 'Data Science'], 200),
    'academic_year': np.random.choice([1, 2, 3, 4], 200),
    'quiz_score_avg': np.random.uniform(50, 100, 200)
})

# Study Materials Data
materials = pd.DataFrame({
    'material_id': range(1, 51),
    'subject': np.random.choice(['AI', 'Blockchain', 'Data Science'], 50),
    'difficulty': np.random.randint(1, 6, 50),  # Scale 1 to 5
    'popularity': np.random.uniform(1, 5, 50)  # Scale 1 to 5
})

# Engagement Data
engagement = pd.DataFrame({
    'student_id': np.random.choice(students['student_id'], 500),
    'material_id': np.random.choice(materials['material_id'], 500),
    'time_spent': np.random.uniform(1, 5, 500),  # In hours
    'quiz_score': np.random.uniform(0, 100, 500)
})


# Calculate Similarity Between Students and Materials
student_interest_matrix = pd.get_dummies(students['interests'])
material_subject_matrix = pd.get_dummies(materials['subject'])

similarity = cosine_similarity(student_interest_matrix, material_subject_matrix)

# Recommend Top 5 Materials for Each Student
recommendations = pd.DataFrame(similarity, index=students['student_id'], columns=materials['material_id'])

def get_recommendations(student_id, n=5):
    return recommendations.loc[student_id].nlargest(n).index.tolist()

print(f"Top 5 Recommendations for Student 1: {get_recommendations(1)}")


# Generate Success Labels (1: Completed, 0: Dropped)
engagement['success'] = (engagement['quiz_score'] > 60).astype(int)

# Prepare Data for Classification
X = engagement[['student_id', 'material_id', 'time_spent', 'quiz_score']]
y = engagement['success']

# Train a RandomForest Model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Predict and Evaluate
y_pred = model.predict(X)
print("Classification Report:\n", classification_report(y, y_pred))
