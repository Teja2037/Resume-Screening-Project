import pandas as pd
resume_df = pd.read_csv("data/raw/AI_Resume_Screening.csv")
job_df = pd.read_csv("data/raw/job description.csv")
resume_df.fillna("", inplace=True)
job_df.fillna("", inplace=True)
resume_df["resume_text"] = (
    resume_df["Skills"] + " " +
    resume_df["Education"] + " " +
    resume_df["Certifications"] + " " +
    resume_df["Job Role"]
)
job_df["job_text"] = (
    job_df["Job Description"] + " " +
    job_df["skills"] + " " +
    job_df["Role"] + " " +
    job_df["Qualifications"]
)
print(resume_df[["Resume_ID", "resume_text"]].head())
print(job_df[["Job Id", "job_text"]].head())
