import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
resume_df = pd.read_csv("data/raw/AI_Resume_Screening.csv")
job_df = pd.read_csv("data/raw/job description.csv")
resume_df.fillna("", inplace=True)
job_df.fillna("", inplace=True)
resume_df["clean_resume_text"] = (
    resume_df["Skills"] + " " +
    resume_df["Education"] + " " +
    resume_df["Certifications"] + " " +
    resume_df["Job Role"]
).str.lower()
job_df["clean_job_text"] = (
    job_df["Job Description"] + " " +
    job_df["skills"] + " " +
    job_df["Role"] + " " +
    job_df["Qualifications"]
).str.lower()
model = SentenceTransformer("all-MiniLM-L6-v2")
resume_embeddings = model.encode(
    resume_df["clean_resume_text"].tolist(),
    show_progress_bar=True
)
job_embeddings = model.encode(
    job_df["clean_job_text"].tolist(),
    show_progress_bar=True
)
print("Resume embeddings shape:", resume_embeddings.shape)
print("Job embeddings shape:", job_embeddings.shape)
