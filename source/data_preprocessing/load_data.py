import pandas as pd
resume_df = pd.read_csv("data/raw/AI_Resume_Screening.csv")
job_df = pd.read_csv("data/raw/job description.csv")
print("Resume Dataset:")
print(resume_df.head())
print("\nColumns:", resume_df.columns)
print("\nJob Description Dataset:")
print(job_df.head())
print("\nColumns:", job_df.columns)
