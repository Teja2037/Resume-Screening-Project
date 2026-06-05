import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="🤖",
    layout="wide"
)
st.markdown("""
<style>
.stApp {
    background-color: #f9fafb;
}
* {
    color: #111827;
}
.card {
    background: #ffffff;
    padding: 22px;
    border-radius: 14px;
    margin-bottom: 18px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}
.title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
}
.subtitle {
    text-align: center;
    font-size: 17px;
    color: #6b7280;
    margin-bottom: 35px;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="title">AI Resume Screening System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Semantic Resume–Job Matching using BERT & NLP</div>',
    unsafe_allow_html=True
)
resume_df = pd.read_csv("data/raw/AI_Resume_Screening.csv").fillna("")
job_df = pd.read_csv("data/raw/job description.csv").fillna("")
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
model = load_model()
col1, col2 = st.columns([2, 3])
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎯 Select Job")

    job_index = st.selectbox(
        "Job Index",
        job_df.index,
        format_func=lambda x: f"Job #{x}"
    )

    top_n = st.slider(
        "Select Number of Top Resumes",
        min_value=5,
        max_value=50,
        value=10
    )
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📄 Job Description Preview")
    st.write(job_df.loc[job_index, "Job Description"])
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
center = st.columns([3, 2, 3])
with center[1]:
    run = st.button("🚀 Analyze & Match Resumes", use_container_width=True)
if run:
    with st.spinner("Running BERT semantic matching..."):

        resume_texts = (
            resume_df["Skills"] + " " +
            resume_df["Education"] + " " +
            resume_df["Certifications"] + " " +
            resume_df["Job Role"]
        ).str.lower().tolist()
        job_text = (
            job_df.loc[job_index, "Job Description"] + " " +
            job_df.loc[job_index, "skills"] + " " +
            job_df.loc[job_index, "Role"] + " " +
            job_df.loc[job_index, "Qualifications"]
        ).lower()
        resume_emb = model.encode(resume_texts)
        job_emb = model.encode([job_text])

        scores = cosine_similarity(job_emb, resume_emb)[0]
        resume_df["Match Score"] = scores

        ranked = resume_df.sort_values("Match Score", ascending=False)
        top_resumes = ranked.head(top_n).copy()
        top_resumes["Match %"] = (top_resumes["Match Score"] * 100).round(2)
        top_resumes["ATS Score"] = top_resumes["Match %"].round(0)
    st.subheader("📊 Matching Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Top Match %", f"{top_resumes['Match %'].max():.2f}%")
    m2.metric("Average Match %", f"{top_resumes['Match %'].mean():.2f}%")
    m3.metric("Total Resumes", len(resume_df))
    st.subheader("📊 Match Percentage Bar Graph")
    chart_df = top_resumes[["Name", "Match %"]].set_index("Name")
    st.bar_chart(chart_df)
    st.subheader(f"🏆 Top {top_n} Matching Resumes")
    for _, row in top_resumes.iterrows():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### {row['Name']} — *{row['Job Role']}*")
        st.progress(row["Match %"] / 100)
        st.write(f"**Match Percentage:** {row['Match %']:.2f}%")
        st.write(f"**ATS Score:** {int(row['ATS Score'])}/100")
        st.caption(f"Skills: {row['Skills']}")
        st.markdown('</div>', unsafe_allow_html=True)
    st.subheader("📋 Top Matching Resumes (Table View)")
    table_cols = [
        "Resume_ID",
        "Name",
        "Job Role",
        "Match %",
        "ATS Score"
    ]
    st.dataframe(
        top_resumes[table_cols],
        use_container_width=True
    )
    st.download_button(
        "⬇️ Download Results",
        top_resumes[table_cols].to_csv(index=False),
        file_name="top_resumes_results.csv",
        mime="text/csv"
    )
