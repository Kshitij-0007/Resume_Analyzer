# =========================
# Resume Analyzer Engine
# =========================

import re
from pathlib import Path

from PyPDF2 import PdfReader
from docx import Document

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sentence_transformers import SentenceTransformer, util


# -------------------------
# NLTK SETUP
# -------------------------
# (safe to call multiple times)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

STOP_WORDS = set(stopwords.words("english"))


# -------------------------
# SKILL CONFIG
# -------------------------
SKILLS = [
    # ---------- Programming & Scripting ----------
    "python", "java", "c", "c++", "c#", "javascript", "typescript",
    "go", "rust", "php", "ruby", "scala", "kotlin",
    "bash", "shell scripting", "powershell",

    # ---------- Web & APIs ----------
    "html", "css", "rest api", "api", "graphql", "soap",
    "django", "flask", "fastapi", "node.js", "express",
    "react", "angular", "vue",

    # ---------- Databases ----------
    "sql", "mysql", "postgresql", "oracle",
    "sqlite", "mssql",
    "nosql", "mongodb", "redis", "cassandra",
    "database design", "query optimization", "indexing",

    # ---------- Data & Analytics ----------
    "data analysis", "data analytics", "data validation",
    "data cleaning", "data processing",
    "pandas", "numpy", "excel",
    "power bi", "tableau",
    "etl", "data pipelines", "reporting",

    # ---------- Machine Learning / AI ----------
    "machine learning", "deep learning", "nlp",
    "text mining", "information retrieval",
    "scikit-learn", "tensorflow", "pytorch",
    "sentence transformers",

    # ---------- Cloud & DevOps ----------
    "aws", "azure", "gcp",
    "docker", "kubernetes",
    "ci/cd", "jenkins", "github actions",
    "linux", "unix",
    "virtualization", "vmware",

    # ---------- Networking ----------
    "networking", "tcp/ip", "http", "https",
    "dns", "dhcp", "ftp", "sftp",
    "firewalls", "load balancing",
    "b2b", "edi",

    # ---------- File Formats & Payloads ----------
    "xml", "json", "csv", "yaml",
    "log files", "payload analysis",
    "message parsing",

    # ---------- QA / Support / Operations ----------
    "technical support", "production support",
    "application support", "customer support",
    "incident management", "issue tracking",
    "ticketing systems", "sla",
    "root cause analysis", "rca",
    "troubleshooting", "debugging",
    "log analysis", "monitoring",

    # ---------- Tools ----------
    "git", "github", "gitlab",
    "jira", "confluence", "servicenow",
    "postman", "swagger",
    "splunk", "elk", "grafana",

    # ---------- Security ----------
    "cybersecurity", "authentication", "authorization",
    "oauth", "sso",
    "encryption", "ssl", "tls",
    "vulnerability assessment",

    # ---------- Operating Systems ----------
    "windows", "linux administration",
    "macos", "system administration",

    # ---------- Soft / Professional Skills ----------
    "communication", "written communication",
    "verbal communication",
    "problem solving", "analytical thinking",
    "critical thinking",
    "documentation", "technical documentation",
    "cross-functional collaboration",
    "stakeholder communication",

    # ---------- Business / Enterprise ----------
    "business analysis", "requirements gathering",
    "process improvement", "workflow optimization",
    "customer success",
    "global support", "enterprise systems",
]

BASELINE_SUPPORT_SKILLS = [
    "linux", "unix", "shell", "bash",
    "sql", "data analysis",
    "xml", "json", "csv", "api",
    "troubleshooting", "technical support",
    "rca", "root cause analysis",
    "log analysis", "automation",
    "communication", "networking"
]


# -------------------------
# MODEL (LOAD ONCE)
# -------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------
# TEXT EXTRACTION
# -------------------------
def extract_resume_text(path: str) -> str:
    path = Path(path)

    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    elif path.suffix.lower() == ".docx":
        doc = Document(path)
        return " ".join(p.text for p in doc.paragraphs)

    else:
        raise ValueError("Unsupported file format. Use PDF or DOCX.")


# -------------------------
# TEXT CLEANING
# -------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    return " ".join(tokens)


# -------------------------
# SKILL EXTRACTION
# -------------------------
def extract_skills(text: str, skills: list[str]) -> list[str]:
    found = set()
    for skill in skills:
        if skill in text:
            found.add(skill)
    return sorted(found)


# -------------------------
# RESUME-ONLY MODE
# -------------------------
def evaluate_resume_only(clean_resume: str) -> dict:
    resume_skills = extract_skills(clean_resume, BASELINE_SUPPORT_SKILLS)
    coverage = (len(resume_skills) / len(BASELINE_SUPPORT_SKILLS)) * 100

    suggestions = []
    if "data analysis" not in resume_skills:
        suggestions.append("Add explicit data analysis experience.")
    if "root cause analysis" not in resume_skills:
        suggestions.append("Mention root cause analysis (RCA).")
    if "networking" not in resume_skills:
        suggestions.append("Add basic networking/B2B exposure.")

    if coverage >= 70:
        readiness = "Strong Technical Support Readiness"
    elif coverage >= 50:
        readiness = "Moderate Technical Support Readiness"
    else:
        readiness = "Basic Technical Support Readiness"

    return {
        "baseline_score": round(coverage, 2),
        "readiness_level": readiness,
        "skills_found": resume_skills,
        "suggestions": suggestions
    }


# -------------------------
# MAIN ENTRY POINT (IMPORTANT)
# -------------------------
def run_resume_analyzer(
    resume_path: str,
    jd_path: str | None = None,
    verbose: bool = True
) -> dict:
    # ---- Resume processing ----
    resume_text = extract_resume_text(resume_path)
    clean_resume = clean_text(resume_text)

    # ---- JD-based mode ----
    if jd_path:
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read()

        clean_jd = clean_text(jd_text)

        resume_skills = extract_skills(clean_resume, SKILLS)
        jd_skills = extract_skills(clean_jd, SKILLS)

        resume_emb = model.encode(clean_resume, convert_to_tensor=True)
        jd_emb = model.encode(clean_jd, convert_to_tensor=True)

        ats_score = util.cos_sim(resume_emb, jd_emb).item() * 100

        missing_skills = list(set(jd_skills) - set(resume_skills))[:10]

        result = {
            "mode": "JD-based",
            "ats_score": round(ats_score, 2),
            "matched_skills": sorted(set(resume_skills) & set(jd_skills)),
            "missing_skills": missing_skills,
        }

    # ---- Resume-only mode ----
    else:
        result = evaluate_resume_only(clean_resume)
        result["mode"] = "Resume-only"

    if verbose:
        print(result)

    return result


# -------------------------
# LOCAL TEST (OPTIONAL)
# -------------------------
if __name__ == "__main__":
    print(run_resume_analyzer("resumes/sample_resume.docx"))
