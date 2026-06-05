import pandas as pd
import re
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
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()
resume_df["clean_resume_text"] = resume_df["resume_text"].apply(clean_text)
job_df["clean_job_text"] = job_df["job_text"].apply(clean_text)
print(resume_df["clean_resume_text"].head())
print(job_df["clean_job_text"].head())
