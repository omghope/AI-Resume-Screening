from flask import Flask, render_template, request
import os
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy

# ---------- Flask app ----------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# ---------- Load SpaCy ----------
nlp = spacy.load("en_core_web_sm")

# ---------- Skills list ----------
SKILLS_LIST = [
    "python", "java", "c++", "c#", "javascript", "react", "angular",
    "node.js", "sql", "mysql", "postgresql", "mongodb", "aws",
    "azure", "docker", "kubernetes", "git", "html", "css", "flask",
    "django", "tensorflow", "pytorch", "excel", "tableau", "powerbi",
    "linux", "bash", "rest api", "graphql", "machine learning", "ai",
    "data analysis", "software testing", "selenium"
]

# ---------- PDF Text Extraction ----------
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# ---------- Cosine similarity ----------
def calculate_similarity(resume_text, jobdesc_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, jobdesc_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * 100, 2)

# ---------- Text cleaning ----------
def clean_text_simple(text):
    text = text or ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------- Extract relevant skills ----------
def extract_relevant_skills(resume_text, jobdesc_text, top_n=10):
    resume_lower = clean_text_simple(resume_text)
    jobdesc_lower = clean_text_simple(jobdesc_text)

    # SpaCy NER to extract potential technical skills from JD
    doc = nlp(jobdesc_text)
    spacy_skills = set()
    for ent in doc.ents:
        # Consider organizations, products, or technologies
        if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART"):
            spacy_skills.add(ent.text.lower())

    # Combine SpaCy detected skills with curated skill list
    candidate_skills = spacy_skills.union(set(SKILLS_LIST))
    
    # Filter skills missing from resume
    missing_skills = [skill for skill in candidate_skills if skill.lower() not in resume_lower]

    # Limit to top_n suggestions
    return missing_skills[:top_n]

# ---------- Flask routes ----------
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    resume = request.files['resume']
    jobdesc = request.files['jobdesc']

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume.filename)
    jobdesc_path = os.path.join(app.config['UPLOAD_FOLDER'], jobdesc.filename)

    resume.save(resume_path)
    jobdesc.save(jobdesc_path)

    resume_text = extract_text_from_pdf(resume_path)
    jobdesc_text = extract_text_from_pdf(jobdesc_path)

    similarity_score = calculate_similarity(resume_text, jobdesc_text)
    missing_keywords = extract_relevant_skills(resume_text, jobdesc_text)

    if similarity_score >= 75:
        message = "Excellent match! Your resume aligns strongly with the job description."
        color_class = "good"
    elif similarity_score >= 50:
        message = "Moderate match. You can improve your resume to better fit the job."
        color_class = "average"
    else:
        message = "Low match. Try tailoring your resume with relevant skills."
        color_class = "poor"

    return render_template(
        "result.html",
        score=similarity_score,
        message=message,
        color_class=color_class,
        keywords=missing_keywords
    )

# ---------- Run Flask ----------
if __name__ == "__main__":
    app.run(debug=True)
