import pandas as pd
import random

data = []

for i in range(2000):
    attendance = random.randint(40, 100)
    assignment = random.randint(40, 100)
    study_hours = random.randint(1, 6)
    mid_marks = random.randint(30, 100)

    # Base logic (not perfect)
    score = (0.3 * attendance +
             0.2 * assignment +
             0.2 * study_hours * 10 +
             0.3 * mid_marks)

    # Add randomness (VERY IMPORTANT)
    noise = random.randint(-15, 15)
    score += noise

    # Decision boundary (not strict)
    if score > 60:
        result = 1  # Pass
    else:
        result = 0  # Fail

    # Add real-world exceptions
    if attendance < 50 and random.random() < 0.3:
        result = 1  # Some low attendance students still pass

    if mid_marks > 85 and random.random() < 0.2:
        result = 0  # Some high scorers fail (rare but possible)

    data.append([attendance, assignment, study_hours, mid_marks, result])

df = pd.DataFrame(data, columns=[
    "Attendance",
    "Assignment_Avg",
    "Study_Hours",
    "Mid_Marks",
    "Final_Result"
])

df.to_csv("student_data.csv", index=False)

print("Realistic dataset created successfully!")