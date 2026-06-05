import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
resume_df = pd.read_csv("data/raw/AI_Resume_Screening.csv").fillna("")
job_df = pd.read_csv("data/raw/job description.csv").fillna("")
resume_texts = (
    resume_df["Skills"] + " " +
    resume_df["Education"] + " " +
    resume_df["Certifications"] + " " +
    resume_df["Job Role"]
).str.lower().tolist()
job_texts = (
    job_df["Job Description"] + " " +
    job_df["skills"] + " " +
    job_df["Role"] + " " +
    job_df["Qualifications"]
).str.lower().tolist()
model = SentenceTransformer("all-MiniLM-L6-v2")
resume_embeddings = model.encode(resume_texts)
job_embeddings = model.encode(job_texts)
job_index = 0
job_vector = job_embeddings[job_index].reshape(1, -1)
scores = cosine_similarity(job_vector, resume_embeddings)[0]
resume_df["Match Score"] = scores
ranked_resumes = resume_df.sort_values(by="Match Score", ascending=False)
ranked_resumes.to_csv("ranked_resumes_for_job_0.csv", index=False)
print("Ranked resumes saved successfully!")
