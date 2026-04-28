import os
import re
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx
except Exception:
    docx = None


# =========================================================
# CONFIG
# =========================================================
DATA_PATH = "data/applications.csv"
DOCUMENTS_PATH = "data/documents.csv"
NOTES_PATH = "data/notes.csv"

st.set_page_config(
    page_title="JobTracker Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

STATUS_OPTIONS = [
    "Applied",
    "Screening",
    "Phone Screen",
    "Interview",
    "Offer",
    "Rejected",
]

SOURCE_OPTIONS = [
    "LinkedIn",
    "Company Website",
    "Referral",
    "Indeed",
    "StepStone",
    "Glassdoor",
    "Email",
    "Other",
]

DOCUMENT_TYPES = [
    "CV",
    "Motivation Letter",
    "Cover Letter",
    "Certificate",
    "Transcript",
    "Portfolio",
    "Other",
]

DEFAULT_COLUMNS = [
    "Company",
    "Role",
    "Date Applied",
    "Status",
    "Source",
    "Location",
    "Next Step",
]

DOCUMENT_COLUMNS = [
    "Document Name",
    "Type",
    "Company",
    "Role",
    "Date Added",
    "Notes",
]

NOTES_COLUMNS = [
    "Title",
    "Company",
    "Date Added",
    "Note",
]

STATUS_COLORS = {
    "Applied": "#2563EB",
    "Screening": "#60A5FA",
    "Phone Screen": "#38BDF8",
    "Interview": "#8B5CF6",
    "Offer": "#10B981",
    "Rejected": "#F43F5E",
}

SOURCE_COLORS = [
    "#2563EB",
    "#14B8A6",
    "#8B5CF6",
    "#FB923C",
    "#CBD5E1",
    "#F43F5E",
    "#22C55E",
    "#64748B",
]

COMMON_SKILLS = [
    "python", "sql", "excel", "power bi", "tableau", "r", "pandas", "numpy",
    "scikit-learn", "machine learning", "deep learning", "statistics",
    "data analysis", "data visualization", "dashboard", "dashboards", "reporting",
    "business intelligence", "bi", "etl", "data warehouse", "spark",
    "cloud", "aws", "azure", "gcp", "docker", "git", "github",
    "nlp", "tensorflow", "pytorch", "streamlit", "plotly", "matplotlib",
    "database", "databases", "postgresql", "mysql", "snowflake", "api",
    "a/b testing", "regression", "classification", "clustering",
    "time series", "data cleaning", "data modeling", "kpi", "kpis",
    "communication", "stakeholder", "presentation", "analytics",
    "forecasting", "optimization", "requirements", "documentation",
    "agile", "scrum", "english", "german",

    "datenanalyse", "datenvisualisierung", "berichterstattung",
    "berichtswesen", "dashboard", "dashboards", "kennzahlen",
    "datenmodellierung", "datenbereinigung", "datenbank",
    "datenbanken", "maschinelles lernen", "künstliche intelligenz",
    "statistik", "optimierung", "prognose", "zeitreihen",
    "klassifikation", "regression", "clustering",
    "praktikum", "werkstudent", "werkstudentin", "trainee",
    "vollzeit", "teilzeit", "homeoffice", "remote", "hybrid",
    "berufserfahrung", "kommunikation", "präsentation",
    "anforderungen", "dokumentation", "agil", "deutsch", "englisch",
]


# =========================================================
# DATA SETUP
# =========================================================
def ensure_data_files():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(DATA_PATH):
        sample_data = [
            ["Microsoft", "Product Manager", "2024-05-20", "Interview", "LinkedIn", "Graz", "Interview - May 28, 2024"],
            ["Shopify", "Senior Product Manager", "2024-05-18", "Screening", "Company Website", "Remote", "Recruiter Follow-up"],
            ["Google", "Program Manager", "2024-05-15", "Interview", "Referral", "Vienna", "Onsite - Jun 3, 2024"],
            ["Notion", "Product Operations Manager", "2024-05-10", "Applied", "Indeed", "Remote", "—"],
            ["Airbnb", "Product Manager", "2024-05-08", "Screening", "LinkedIn", "Berlin", "Take-home Assignment"],
            ["Amazon", "Business Analyst", "2024-05-06", "Rejected", "LinkedIn", "Munich", "—"],
            ["Siemens", "Data Analyst", "2024-05-04", "Applied", "Company Website", "Graz", "Awaiting response"],
        ]
        pd.DataFrame(sample_data, columns=DEFAULT_COLUMNS).to_csv(DATA_PATH, index=False)

    if not os.path.exists(DOCUMENTS_PATH):
        pd.DataFrame(columns=DOCUMENT_COLUMNS).to_csv(DOCUMENTS_PATH, index=False)

    if not os.path.exists(NOTES_PATH):
        pd.DataFrame(columns=NOTES_COLUMNS).to_csv(NOTES_PATH, index=False)


def normalize_status(value):
    if pd.isna(value) or str(value).strip() == "":
        return "Applied"

    raw = str(value).strip().lower()

    mapping = {
        "applied": "Applied",
        "sent": "Applied",
        "submitted": "Applied",
        "waiting": "Applied",
        "pending": "Applied",
        "screening": "Screening",
        "screen": "Screening",
        "hr screen": "Phone Screen",
        "phone screen": "Phone Screen",
        "phone": "Phone Screen",
        "call": "Phone Screen",
        "interview": "Interview",
        "interviewing": "Interview",
        "technical interview": "Interview",
        "onsite": "Interview",
        "offer": "Offer",
        "accepted": "Offer",
        "rejected": "Rejected",
        "rejection": "Rejected",
        "declined": "Rejected",
        "not selected": "Rejected",
        "beworben": "Applied",
        "eingereicht": "Applied",
        "wartend": "Applied",
        "in prüfung": "Screening",
        "telefoninterview": "Phone Screen",
        "telefonat": "Phone Screen",
        "vorstellungsgespräch": "Interview",
        "angebot": "Offer",
        "zusage": "Offer",
        "absage": "Rejected",
        "abgelehnt": "Rejected",
    }

    return mapping.get(raw, str(value).strip().title())


def normalize_source(value):
    if pd.isna(value) or str(value).strip() == "":
        return "Other"

    raw = str(value).strip().lower()

    mapping = {
        "linkedin": "LinkedIn",
        "linked in": "LinkedIn",
        "company": "Company Website",
        "company website": "Company Website",
        "website": "Company Website",
        "career page": "Company Website",
        "careers": "Company Website",
        "karriereseite": "Company Website",
        "unternehmenswebsite": "Company Website",
        "referral": "Referral",
        "empfehlung": "Referral",
        "indeed": "Indeed",
        "stepstone": "StepStone",
        "glassdoor": "Glassdoor",
        "email": "Email",
        "mail": "Email",
        "e-mail": "Email",
    }

    return mapping.get(raw, str(value).strip().title())


def clean_applications(data):
    for col in DEFAULT_COLUMNS:
        if col not in data.columns:
            data[col] = ""

    data = data[DEFAULT_COLUMNS].copy()

    data["Company"] = data["Company"].fillna("").astype(str).str.strip().replace("", "Unknown Company")
    data["Role"] = data["Role"].fillna("").astype(str).str.strip().replace("", "Unknown Role")
    data["Location"] = data["Location"].fillna("").astype(str).str.strip().replace("", "—")
    data["Next Step"] = data["Next Step"].fillna("").astype(str).str.strip().replace("", "Awaiting response")

    data["Status"] = data["Status"].apply(normalize_status)
    data["Source"] = data["Source"].apply(normalize_source)

    data["Date Applied"] = pd.to_datetime(data["Date Applied"], errors="coerce")
    data["Date Applied"] = data["Date Applied"].fillna(pd.Timestamp(date.today()))
    data["Month"] = data["Date Applied"].dt.strftime("%b %Y")

    return data


def load_applications():
    ensure_data_files()
    data = pd.read_csv(DATA_PATH)
    return clean_applications(data)


def save_applications(data):
    data = clean_applications(data)
    data_to_save = data[DEFAULT_COLUMNS].copy()
    data_to_save["Date Applied"] = pd.to_datetime(data_to_save["Date Applied"]).dt.strftime("%Y-%m-%d")
    data_to_save.to_csv(DATA_PATH, index=False)


def load_documents():
    ensure_data_files()
    data = pd.read_csv(DOCUMENTS_PATH)

    for col in DOCUMENT_COLUMNS:
        if col not in data.columns:
            data[col] = ""

    data = data[DOCUMENT_COLUMNS]
    data["Date Added"] = pd.to_datetime(data["Date Added"], errors="coerce")
    return data


def save_documents(data):
    data_to_save = data[DOCUMENT_COLUMNS].copy()
    data_to_save["Date Added"] = pd.to_datetime(data_to_save["Date Added"], errors="coerce").dt.strftime("%Y-%m-%d")
    data_to_save.to_csv(DOCUMENTS_PATH, index=False)


def load_notes():
    ensure_data_files()
    data = pd.read_csv(NOTES_PATH)

    for col in NOTES_COLUMNS:
        if col not in data.columns:
            data[col] = ""

    data = data[NOTES_COLUMNS]
    data["Date Added"] = pd.to_datetime(data["Date Added"], errors="coerce")
    return data


def save_notes(data):
    data_to_save = data[NOTES_COLUMNS].copy()
    data_to_save["Date Added"] = pd.to_datetime(data_to_save["Date Added"], errors="coerce").dt.strftime("%Y-%m-%d")
    data_to_save.to_csv(NOTES_PATH, index=False)


# =========================================================
# CRUD
# =========================================================
def add_application(company, role, date_applied, status, source, location, next_step):
    data = load_applications()
    new_row = pd.DataFrame(
        [[company, role, date_applied, status, source, location, next_step]],
        columns=DEFAULT_COLUMNS,
    )
    data = pd.concat([data[DEFAULT_COLUMNS], new_row], ignore_index=True)
    save_applications(data)


def update_application(row_index, company, role, date_applied, status, source, location, next_step):
    data = load_applications()
    data.loc[row_index, "Company"] = company
    data.loc[row_index, "Role"] = role
    data.loc[row_index, "Date Applied"] = date_applied
    data.loc[row_index, "Status"] = status
    data.loc[row_index, "Source"] = source
    data.loc[row_index, "Location"] = location
    data.loc[row_index, "Next Step"] = next_step
    save_applications(data)


def delete_application(row_index):
    data = load_applications()
    data = data.drop(index=row_index).reset_index(drop=True)
    save_applications(data)


# =========================================================
# CSV IMPORT
# =========================================================
def normalize_col_name(col):
    return str(col).strip().lower().replace("_", " ").replace("-", " ")


def guess_column(uploaded_columns, target):
    aliases = {
        "Company": [
            "company", "company name", "employer", "organisation", "organization",
            "firm", "business", "unternehmen", "firma", "arbeitgeber",
        ],
        "Role": [
            "role", "job title", "position", "title", "job", "vacancy",
            "job role", "stelle", "berufsbezeichnung",
        ],
        "Date Applied": [
            "date applied", "applied date", "application date", "date",
            "applied on", "created", "created at", "submitted date",
            "bewerbungsdatum", "datum", "eingereicht am",
        ],
        "Status": [
            "status", "stage", "application status", "pipeline", "state",
            "stand", "phase", "bewerbungsstatus",
        ],
        "Source": [
            "source", "platform", "job board", "channel", "where", "origin",
            "quelle", "plattform", "kanal",
        ],
        "Location": [
            "location", "city", "country", "place", "job location",
            "ort", "stadt", "land", "standort",
        ],
        "Next Step": [
            "next step", "next action", "follow up", "follow-up", "notes",
            "note", "action", "todo", "nächster schritt", "notiz",
            "aktion", "bemerkung",
        ],
    }

    normalized_lookup = {normalize_col_name(col): col for col in uploaded_columns}

    for alias in aliases[target]:
        if alias in normalized_lookup:
            return normalized_lookup[alias]

    for normalized, original in normalized_lookup.items():
        for alias in aliases[target]:
            if alias in normalized:
                return original

    return "None"


def build_imported_applications(uploaded_df, mapping):
    imported = pd.DataFrame()

    for target_col in DEFAULT_COLUMNS:
        selected_col = mapping.get(target_col, "None")

        if selected_col != "None" and selected_col in uploaded_df.columns:
            imported[target_col] = uploaded_df[selected_col]
        else:
            defaults = {
                "Company": "Unknown Company",
                "Role": "Unknown Role",
                "Date Applied": date.today().strftime("%Y-%m-%d"),
                "Status": "Applied",
                "Source": "Other",
                "Location": "—",
                "Next Step": "Awaiting response",
            }
            imported[target_col] = defaults[target_col]

    return clean_applications(imported)


# =========================================================
# LOCAL NLP / ML
# =========================================================
@st.cache_resource
def load_embedding_model():
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None

    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def read_pdf_text(uploaded_file):
    if pdfplumber is None:
        return ""

    text_parts = []

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

    return "\n".join(text_parts)


def read_docx_text(uploaded_file):
    if docx is None:
        return ""

    document = docx.Document(uploaded_file)
    return "\n".join([para.text for para in document.paragraphs])


def read_uploaded_cv(uploaded_file):
    if uploaded_file is None:
        return ""

    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".pdf"):
            return read_pdf_text(uploaded_file)

        if file_name.endswith(".docx"):
            return read_docx_text(uploaded_file)

        if file_name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore")

    except Exception:
        return ""

    return ""


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-ZäöüÄÖÜß0-9+#./\- ]", " ", text)
    return text.strip()


def extract_skills(text):
    cleaned = clean_text(text)
    found = []

    for skill in COMMON_SKILLS:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, cleaned):
            found.append(skill)

    return sorted(set(found))


def tfidf_similarity(text_a, text_b):
    if not text_a.strip() or not text_b.strip():
        return 0

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([text_a, text_b])
    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return round(score * 100)


def semantic_similarity(text_a, text_b):
    if not text_a.strip() or not text_b.strip():
        return 0

    model = load_embedding_model()

    if model is None:
        return tfidf_similarity(text_a, text_b)

    embeddings = model.encode([text_a, text_b])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(score * 100)


def keyword_score(cv_text, job_text):
    cv_skills = set(extract_skills(cv_text))
    job_skills = set(extract_skills(job_text))

    if not job_skills:
        return 0, [], []

    matched = sorted(cv_skills.intersection(job_skills))
    missing = sorted(job_skills.difference(cv_skills))

    score = round((len(matched) / len(job_skills)) * 100)
    return score, matched, missing


def section_score(cv_text):
    cleaned = clean_text(cv_text)

    sections = {
        "Education / Ausbildung": [
            "education", "university", "master", "bachelor", "degree",
            "ausbildung", "universität", "studium", "abschluss",
        ],
        "Experience / Erfahrung": [
            "experience", "work experience", "employment", "internship",
            "erfahrung", "berufserfahrung", "praktikum", "beschäftigung",
        ],
        "Skills / Fähigkeiten": [
            "skills", "technical skills", "tools", "technologies",
            "fähigkeiten", "kenntnisse", "technologien", "werkzeuge",
        ],
        "Projects / Projekte": [
            "projects", "project", "projekte", "projekt",
        ],
        "Contact / Kontakt": [
            "email", "phone", "linkedin", "github",
            "e-mail", "telefon", "kontakt",
        ],
    }

    found_sections = []

    for section, keywords in sections.items():
        if any(keyword in cleaned for keyword in keywords):
            found_sections.append(section)

    score = round((len(found_sections) / len(sections)) * 100)
    missing_sections = sorted(set(sections.keys()) - set(found_sections))

    return score, found_sections, missing_sections


def generate_cv_suggestions(missing_keywords, missing_sections, final_score):
    suggestions = []

    if missing_keywords:
        top_missing = ", ".join(missing_keywords[:8])
        suggestions.append(f"Add or make clearer these job keywords in your CV: {top_missing}.")

    if missing_sections:
        suggestions.append(f"Consider adding or clarifying these CV sections: {', '.join(missing_sections)}.")

    if final_score < 50:
        suggestions.append("The CV seems weakly aligned with this job. Tailor the skills and experience bullets more directly to the job description.")
    elif final_score < 75:
        suggestions.append("The CV has a moderate match. Add more role-specific keywords and quantify relevant project or work experience.")
    else:
        suggestions.append("The CV looks reasonably aligned. Focus on small improvements and clearer evidence for the most important job requirements.")

    suggestions.append("Use exact wording from the job description where it truthfully matches your experience.")

    return suggestions


def analyze_job_description(job_text):
    cleaned = clean_text(job_text)
    skills = extract_skills(job_text)

    seniority = "Not clear"

    if re.search(
        r"\binternship\b|\bintern\b|\bworking student\b|\btrainee\b|"
        r"\bpraktikum\b|\bpraktikant\b|\bpraktikantin\b|\bwerkstudent\b|\bwerkstudentin\b",
        cleaned,
    ):
        seniority = "Internship / Working Student / Trainee"

    elif re.search(
        r"\bjunior\b|\bentry level\b|\bgraduate\b|"
        r"\beinsteiger\b|\bberufseinsteiger\b|\bberufseinsteigerin\b",
        cleaned,
    ):
        seniority = "Junior / Entry-level"

    elif re.search(
        r"\bmid\b|\bprofessional\b|\b2\+ years\b|\b3\+ years\b|"
        r"\b2 jahre\b|\b3 jahre\b|\bberufserfahrung\b",
        cleaned,
    ):
        seniority = "Mid-level"

    elif re.search(
        r"\bsenior\b|\blead\b|\b5\+ years\b|\b7\+ years\b|"
        r"\b5 jahre\b|\b7 jahre\b|\bleitung\b|\bteamlead\b",
        cleaned,
    ):
        seniority = "Senior"

    red_flags = []

    if re.search(
        r"\bc1\b|\bnative german\b|\bfluent german\b|"
        r"\bdeutsch c1\b|\bdeutschkenntnisse c1\b|\bmuttersprache deutsch\b|"
        r"\bverhandlungssicheres deutsch\b",
        cleaned,
    ):
        red_flags.append("Strong German requirement")

    if re.search(r"\bfull-time\b|\bfull time\b|\bvollzeit\b", cleaned):
        red_flags.append("Full-time requirement")

    if re.search(
        r"\b5\+ years\b|\b7\+ years\b|\bsenior\b|"
        r"\b5 jahre\b|\b7 jahre\b|\blangjährige berufserfahrung\b",
        cleaned,
    ):
        red_flags.append("High experience requirement")

    if re.search(
        r"\btravel\b|\brelocation\b|\breisebereitschaft\b|\bumzug\b",
        cleaned,
    ):
        red_flags.append("Travel / relocation may be required")

    responsibilities_keywords = [
        "analyze", "build", "develop", "design", "maintain", "report",
        "visualize", "communicate", "collaborate", "optimize", "automate",
        "clean", "model", "present", "support",
        "analysieren", "entwickeln", "erstellen", "gestalten", "pflegen",
        "berichten", "visualisieren", "kommunizieren", "zusammenarbeiten",
        "optimieren", "automatisieren", "bereinigen", "modellieren",
        "präsentieren", "unterstützen",
    ]

    responsibilities = []
    sentences = re.split(r"[.\n]", job_text)

    for sentence in sentences:
        s_clean = clean_text(sentence)
        if any(word in s_clean for word in responsibilities_keywords):
            if len(sentence.strip()) > 25:
                responsibilities.append(sentence.strip())

    responsibilities = responsibilities[:6]

    if len(job_text.strip()) > 0:
        simple_summary = (
            "This role appears to focus on "
            + (", ".join(skills[:6]) if skills else "general job-related responsibilities")
            + ". The main fit depends on matching your CV with the required skills, tools, and responsibilities."
        )
    else:
        simple_summary = "No job description provided."

    return {
        "summary": simple_summary,
        "skills": skills,
        "seniority": seniority,
        "red_flags": red_flags,
        "responsibilities": responsibilities,
    }


def final_match_score(semantic, keyword, section):
    return round((0.55 * semantic) + (0.30 * keyword) + (0.15 * section))


# =========================================================
# CSS
# =========================================================
st.markdown(
    """
<style>
    .block-container {
        padding-top: 1.1rem;
        padding-left: 1.4rem;
        padding-right: 1.4rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }

    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E5E7EB;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.7rem;
    }

    h1 {
        font-size: 32px !important;
        font-weight: 850 !important;
        color: #111827 !important;
        margin-bottom: 0 !important;
    }

    h2, h3 {
        color: #111827 !important;
        font-weight: 850 !important;
    }

    .brand-box {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 17px 16px;
        margin-bottom: 18px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }

    .brand-title {
        font-size: 22px;
        font-weight: 850;
        color: #111827;
    }

    .user-box {
        margin-top: 24px;
        border-top: 1px solid #E5E7EB;
        padding-top: 16px;
        color: #334155;
        font-size: 14px;
        line-height: 1.45;
    }

    .kpi-card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 18px;
        padding: 17px;
        box-shadow: 0 2px 8px rgba(15,23,42,0.04);
        min-height: 124px;
    }

    .kpi-top {
        display: flex;
        align-items: center;
        gap: 13px;
    }

    .kpi-icon {
        width: 54px;
        height: 54px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
    }

    .kpi-title {
        color: #475569;
        font-size: 15px;
        font-weight: 650;
        margin-bottom: 4px;
    }

    .kpi-value {
        color: #0F172A;
        font-size: 32px;
        font-weight: 850;
        line-height: 1;
    }

    .kpi-note-green {
        color: #059669;
        font-size: 13px;
        font-weight: 750;
        margin-top: 13px;
    }

    .kpi-note-red {
        color: #EF4444;
        font-size: 13px;
        font-weight: 750;
        margin-top: 13px;
    }

    .card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 18px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(15,23,42,0.04);
    }

    .section-title {
        color: #111827;
        font-size: 18px;
        font-weight: 850;
        margin-bottom: 4px;
    }

    .small-muted {
        color: #64748B;
        font-size: 13px;
    }

    .insight-row {
        border: 1px solid #EEF2F7;
        border-radius: 14px;
        padding: 12px 13px;
        margin-bottom: 9px;
        background: white;
    }

    .insight-title {
        font-size: 15px;
        font-weight: 850;
        color: #111827;
    }

    .insight-sub {
        font-size: 13px;
        color: #64748B;
        margin-top: 3px;
    }

    .right-value {
        float: right;
        font-weight: 850;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #E5E7EB;
    }

    .stSelectbox label,
    .stTextInput label,
    .stDateInput label,
    .stTextArea label,
    .stMultiSelect label,
    .stFileUploader label {
        font-weight: 700 !important;
        color: #334155 !important;
    }

    .stButton button {
        border-radius: 12px;
        font-weight: 750;
    }

    .mini-card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 14px;
        box-shadow: 0 1px 5px rgba(15,23,42,0.04);
        min-height: 105px;
    }

    .mini-title {
        color: #64748B;
        font-size: 13px;
        font-weight: 700;
    }

    .mini-value {
        color: #0F172A;
        font-size: 24px;
        font-weight: 850;
        margin-top: 5px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"


def go_to_page(page_name):
    st.session_state.page = page_name


# =========================================================
# LOAD DATA
# =========================================================
df = load_applications()
documents_df = load_documents()
notes_df = load_notes()


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 💼 JobTracker")

    pages = [
        ("Dashboard", "📊"),
        ("Applications", "📄"),
        ("Calendar", "📅"),
        ("Companies", "🏢"),
        ("Documents", "📁"),
        ("Analytics", "📈"),
        ("Insights", "✨"),
        ("Notes", "📝"),
    ]

    for page_name, icon in pages:
        button_type = "primary" if st.session_state.page == page_name else "secondary"

        if st.button(
            f"{icon} {page_name}",
            use_container_width=True,
            type=button_type,
        ):
            go_to_page(page_name)
            st.rerun()

    st.divider()

    st.markdown("### Mohamed Elhadidy")
    st.caption("mohamed.elhadidy@student.tugraz.at")

    st.markdown("**📌 Goal**")
    st.caption("Data / BI / ML roles")

    sidebar_total = len(df)
    sidebar_active = len(
        df[df["Status"].isin(["Applied", "Screening", "Phone Screen", "Interview"])]
    )
    sidebar_interviews = len(df[df["Status"] == "Interview"])

    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown(
            f"""
            <div style="text-align:center;">
                <div style="font-size:11px; color:#64748B;">Apps</div>
                <div style="font-size:22px; font-weight:850; color:#0F172A;">{sidebar_total}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with s2:
        st.markdown(
            f"""
            <div style="text-align:center;">
                <div style="font-size:11px; color:#64748B;">Active</div>
                <div style="font-size:22px; font-weight:850; color:#0F172A;">{sidebar_active}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with s3:
        st.markdown(
            f"""
            <div style="text-align:center;">
                <div style="font-size:11px; color:#64748B;">Int.</div>
                <div style="font-size:22px; font-weight:850; color:#0F172A;">{sidebar_interviews}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.link_button(
            "LinkedIn",
            "https://www.linkedin.com/in/elhadidy19/",
            use_container_width=True,
        )

    with col2:
        st.link_button(
            "GitHub",
            "https://github.com/elhadidy2025",
            use_container_width=True,
        )

    st.caption("Demo app_Elhadidy · CSV upload supported")


# =========================================================
# SHARED COMPONENTS
# =========================================================
def add_application_form(expanded=False):
    with st.expander("➕ Add New Application", expanded=expanded):
        with st.form("add_application_form", clear_on_submit=True):
            f1, f2, f3 = st.columns(3)

            with f1:
                company = st.text_input("Company")
                status = st.selectbox("Status", STATUS_OPTIONS)

            with f2:
                role = st.text_input("Role")
                source = st.selectbox("Source", SOURCE_OPTIONS)

            with f3:
                date_applied = st.date_input("Date Applied", value=date.today())
                location = st.text_input("Location", value="Graz")

            next_step = st.text_input("Next Step", value="Awaiting response")

            submitted = st.form_submit_button("Add Application", use_container_width=True)

            if submitted:
                if not company.strip() or not role.strip():
                    st.error("Please enter at least Company and Role.")
                else:
                    add_application(
                        company=company.strip(),
                        role=role.strip(),
                        date_applied=date_applied.strftime("%Y-%m-%d"),
                        status=status,
                        source=source,
                        location=location.strip(),
                        next_step=next_step.strip(),
                    )
                    st.success("Application added successfully.")
                    st.rerun()


def upload_csv_component(expanded=False):
    with st.expander("📤 Upload CSV and Visualize", expanded=expanded):
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is None:
            st.info("Upload a CSV file. Extra columns will be ignored.")
            return

        try:
            uploaded_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV file: {e}")
            return

        if uploaded_df.empty:
            st.warning("The uploaded CSV is empty.")
            return

        st.markdown("### Preview")
        st.dataframe(uploaded_df.head(10), use_container_width=True, hide_index=True)

        st.markdown("### Column Mapping")

        csv_columns = ["None"] + uploaded_df.columns.tolist()
        mapping = {}

        m1, m2, m3 = st.columns(3)

        with m1:
            for target_col in ["Company", "Role"]:
                guessed = guess_column(uploaded_df.columns.tolist(), target_col)
                default_index = csv_columns.index(guessed) if guessed in csv_columns else 0
                mapping[target_col] = st.selectbox(
                    f"{target_col}",
                    csv_columns,
                    index=default_index,
                    key=f"map_{target_col}",
                )

        with m2:
            for target_col in ["Date Applied", "Status"]:
                guessed = guess_column(uploaded_df.columns.tolist(), target_col)
                default_index = csv_columns.index(guessed) if guessed in csv_columns else 0
                mapping[target_col] = st.selectbox(
                    f"{target_col}",
                    csv_columns,
                    index=default_index,
                    key=f"map_{target_col}",
                )

        with m3:
            for target_col in ["Source", "Location", "Next Step"]:
                guessed = guess_column(uploaded_df.columns.tolist(), target_col)
                default_index = csv_columns.index(guessed) if guessed in csv_columns else 0
                mapping[target_col] = st.selectbox(
                    f"{target_col}",
                    csv_columns,
                    index=default_index,
                    key=f"map_{target_col}",
                )

        imported_apps = build_imported_applications(uploaded_df, mapping)

        st.markdown("### Imported Preview")
        preview = imported_apps[DEFAULT_COLUMNS].copy()
        preview["Date Applied"] = preview["Date Applied"].dt.strftime("%Y-%m-%d")
        st.dataframe(preview.head(10), use_container_width=True, hide_index=True)

        st.warning("Replace will overwrite your current applications.csv. Merge will add uploaded rows to existing data.")

        c1, c2 = st.columns(2)

        with c1:
            if st.button("Replace Current Data", use_container_width=True, type="primary"):
                save_applications(imported_apps)
                st.success("Current data replaced with uploaded CSV.")
                st.rerun()

        with c2:
            if st.button("Merge With Current Data", use_container_width=True):
                current = load_applications()
                merged = pd.concat([current[DEFAULT_COLUMNS], imported_apps[DEFAULT_COLUMNS]], ignore_index=True)
                save_applications(merged)
                st.success("Uploaded CSV merged with current data.")
                st.rerun()


def calculate_metrics(data):
    total = len(data)
    interviews = len(data[data["Status"] == "Interview"])
    offers = len(data[data["Status"] == "Offer"])
    rejections = len(data[data["Status"] == "Rejected"])

    active_statuses = ["Applied", "Screening", "Phone Screen", "Interview"]
    active = len(data[data["Status"].isin(active_statuses)])

    responded = len(data[data["Status"].isin(["Screening", "Phone Screen", "Interview", "Offer", "Rejected"])])
    response_rate = round((responded / total) * 100) if total > 0 else 0
    offer_rate = round((offers / total) * 100) if total > 0 else 0
    rejection_rate = round((rejections / total) * 100) if total > 0 else 0

    return {
        "total": total,
        "interviews": interviews,
        "offers": offers,
        "rejections": rejections,
        "active": active,
        "response_rate": response_rate,
        "offer_rate": offer_rate,
        "rejection_rate": rejection_rate,
    }


def get_dashboard_filters(data):
    months = ["All"] + sorted(data["Month"].dropna().unique().tolist())
    statuses = ["All"] + sorted(data["Status"].dropna().unique().tolist())
    sources = ["All"] + sorted(data["Source"].dropna().unique().tolist())
    locations = ["All"] + sorted(data["Location"].dropna().unique().tolist())

    header_col, month_col, status_col, source_col, location_col, reset_col = st.columns(
        [3.1, 1.05, 1.05, 1.05, 1.05, 0.8]
    )

    with header_col:
        st.markdown("<h1>Job Application Tracker Dashboard</h1>", unsafe_allow_html=True)

    with month_col:
        selected_month = st.selectbox("Month", months, index=0)

    with status_col:
        selected_status = st.selectbox("Status", statuses, index=0)

    with source_col:
        selected_source = st.selectbox("Source", sources, index=0)

    with location_col:
        selected_location = st.selectbox("Location", locations, index=0)

    with reset_col:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Reset", use_container_width=True):
            st.rerun()

    filtered = data.copy()

    if selected_month != "All":
        filtered = filtered[filtered["Month"] == selected_month]

    if selected_status != "All":
        filtered = filtered[filtered["Status"] == selected_status]

    if selected_source != "All":
        filtered = filtered[filtered["Source"] == selected_source]

    if selected_location != "All":
        filtered = filtered[filtered["Location"] == selected_location]

    return filtered


# =========================================================
# DASHBOARD PAGE
# =========================================================
def render_dashboard():
    filtered_df = get_dashboard_filters(df)
    metrics = calculate_metrics(filtered_df)

    add_application_form(expanded=False)
    upload_csv_component(expanded=False)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-top"><div class="kpi-icon" style="background:#DBEAFE;">📄</div><div><div class="kpi-title">Total Applications</div><div class="kpi-value">{metrics["total"]}</div></div></div><div class="kpi-note-green">Live from your data</div></div>""", unsafe_allow_html=True)

    with k2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-top"><div class="kpi-icon" style="background:#F3E8FF;">👥</div><div><div class="kpi-title">Interviews</div><div class="kpi-value">{metrics["interviews"]}</div></div></div><div class="kpi-note-green">Interview stage</div></div>""", unsafe_allow_html=True)

    with k3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-top"><div class="kpi-icon" style="background:#DCFCE7;">💼</div><div><div class="kpi-title">Offers</div><div class="kpi-value">{metrics["offers"]}</div></div></div><div class="kpi-note-green">Offer rate: {metrics["offer_rate"]}%</div></div>""", unsafe_allow_html=True)

    with k4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-top"><div class="kpi-icon" style="background:#FEE2E2;">❌</div><div><div class="kpi-title">Rejections</div><div class="kpi-value">{metrics["rejections"]}</div></div></div><div class="kpi-note-red">Rejected applications</div></div>""", unsafe_allow_html=True)

    with k5:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-top"><div class="kpi-icon" style="background:#FEF3C7;">%</div><div><div class="kpi-title">Response Rate</div><div class="kpi-value">{metrics["response_rate"]}%</div></div></div><div class="kpi-note-green">Non-applied responses</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    pipeline_counts = filtered_df["Status"].value_counts().reindex(STATUS_OPTIONS, fill_value=0)
    pipeline_labels = pipeline_counts.index.tolist()
    pipeline_values = pipeline_counts.values.tolist()
    pipeline_percentages = [
        f"{round((value / metrics['total']) * 100) if metrics['total'] > 0 else 0}%"
        for value in pipeline_values
    ]
    pipeline_colors = [STATUS_COLORS.get(status, "#64748B") for status in pipeline_labels]

    month_counts = (
        filtered_df.groupby(filtered_df["Date Applied"].dt.to_period("M"))
        .size()
        .reset_index(name="Applications")
    )

    if not month_counts.empty:
        month_counts["Month"] = month_counts["Date Applied"].dt.strftime("%b %Y")
    else:
        month_counts = pd.DataFrame({"Month": [], "Applications": []})

    source_counts = filtered_df["Source"].value_counts().reset_index()
    source_counts.columns = ["Source", "Applications"]

    chart_col1, chart_col2, chart_col3 = st.columns([1.25, 1.35, 1.25])

    with chart_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Application Pipeline</div>', unsafe_allow_html=True)

        fig_pipeline = go.Figure()
        fig_pipeline.add_trace(
            go.Bar(
                x=pipeline_values,
                y=pipeline_labels,
                orientation="h",
                text=[f"{v} ({p})" for v, p in zip(pipeline_values, pipeline_percentages)],
                textposition="outside",
                marker=dict(color=pipeline_colors),
                hoverinfo="skip",
            )
        )

        max_x = max(pipeline_values) if pipeline_values else 1

        fig_pipeline.update_layout(
            height=265,
            margin=dict(l=8, r=60, t=4, b=4),
            xaxis=dict(showgrid=False, visible=False, range=[0, max_x + 2]),
            yaxis=dict(autorange="reversed", showgrid=False, tickfont=dict(size=13, color="#334155")),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
        )

        st.plotly_chart(fig_pipeline, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Conversion rate (Applied → Offer): {metrics['offer_rate']}%")
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Applications by Month</div>', unsafe_allow_html=True)

        fig_month = go.Figure()
        fig_month.add_trace(
            go.Bar(
                x=month_counts["Month"],
                y=month_counts["Applications"],
                text=month_counts["Applications"],
                textposition="outside",
                marker_color="#2563EB",
                width=0.42,
            )
        )

        y_max = int(month_counts["Applications"].max()) + 2 if not month_counts.empty else 5

        fig_month.update_layout(
            height=265,
            margin=dict(l=8, r=8, t=6, b=4),
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis=dict(range=[0, y_max], gridcolor="#E5E7EB", tickfont=dict(size=12, color="#64748B")),
            xaxis=dict(showgrid=False, tickfont=dict(size=12, color="#64748B")),
            showlegend=False,
        )

        st.plotly_chart(fig_month, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Applications by Source</div>', unsafe_allow_html=True)

        fig_source = go.Figure()

        if not source_counts.empty:
            fig_source.add_trace(
                go.Pie(
                    labels=source_counts["Source"],
                    values=source_counts["Applications"],
                    hole=0.62,
                    marker_colors=SOURCE_COLORS,
                    textinfo="none",
                    sort=False,
                )
            )

        fig_source.update_layout(
            height=265,
            margin=dict(l=6, r=6, t=6, b=4),
            paper_bgcolor="white",
            annotations=[
                dict(
                    text=f"<b>{metrics['total']}</b><br>Total",
                    x=0.5,
                    y=0.5,
                    font_size=20,
                    showarrow=False,
                )
            ],
            legend=dict(orientation="v", x=1.02, y=0.86, font=dict(size=12, color="#334155")),
        )

        st.plotly_chart(fig_source, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    table_col, insight_col = st.columns([2.35, 1.15])

    with table_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recent Applications</div>', unsafe_allow_html=True)

        display_df = filtered_df.copy().sort_values("Date Applied", ascending=False)

        if not display_df.empty:
            display_df["Date Applied"] = display_df["Date Applied"].dt.strftime("%b %d, %Y")
            st.dataframe(display_df[DEFAULT_COLUMNS], use_container_width=True, hide_index=True, height=245)
        else:
            st.info("No applications match the selected filters.")

        if st.button("View all applications", use_container_width=True):
            go_to_page("Applications")
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with insight_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Insights & Next Steps ✨</div>', unsafe_allow_html=True)

        if not source_counts.empty:
            best_source = source_counts.iloc[0]["Source"]
            best_source_count = int(source_counts.iloc[0]["Applications"])
        else:
            best_source = "—"
            best_source_count = 0

        pending_followups = len(
            filtered_df[
                filtered_df["Next Step"]
                .fillna("")
                .str.lower()
                .str.contains("follow|await|response|recruiter|interview|assignment", regex=True)
            ]
        )

        upcoming_interviews = filtered_df[filtered_df["Status"] == "Interview"].copy()
        upcoming_interviews = upcoming_interviews.sort_values("Date Applied", ascending=False)

        if not upcoming_interviews.empty:
            next_interview_company = upcoming_interviews.iloc[0]["Company"]
            next_interview_role = upcoming_interviews.iloc[0]["Role"]
        else:
            next_interview_company = "No interview"
            next_interview_role = "Keep applying"

        st.markdown(
            f"""
            <div class="insight-row">
                <div class="insight-title">
                    📅 Upcoming Interview
                    <span class="right-value" style="color:#7C3AED;">{metrics["interviews"]}</span>
                </div>
                <div class="insight-sub">{next_interview_company} - {next_interview_role}</div>
            </div>

            <div class="insight-row">
                <div class="insight-title">
                    ⏰ Pending Follow-ups
                    <span class="right-value" style="color:#F59E0B;">{pending_followups}</span>
                </div>
                <div class="insight-sub">Applications that may need follow-up</div>
            </div>

            <div class="insight-row">
                <div class="insight-title">
                    🎯 Best Source
                    <span class="right-value" style="color:#10B981;">{best_source_count}</span>
                </div>
                <div class="insight-sub">{best_source}</div>
            </div>

            <div class="insight-row">
                <div class="insight-title">
                    📈 Active Applications
                    <span class="right-value" style="color:#2563EB;">{metrics["active"]}</span>
                </div>
                <div class="insight-sub">Still in process</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("View full insights", use_container_width=True):
            go_to_page("Insights")
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# APPLICATIONS PAGE
# =========================================================
def render_applications():
    st.markdown("<h1>Applications</h1>", unsafe_allow_html=True)
    st.caption("Manage all your job applications in one place.")

    add_application_form(expanded=True)
    upload_csv_component(expanded=True)

    st.markdown("## Application Details")

    search = st.text_input("Search by company, role, source, location, or next step")

    data = df.copy()

    if search.strip():
        search_lower = search.lower()
        data = data[
            data[DEFAULT_COLUMNS]
            .astype(str)
            .apply(lambda row: row.str.lower().str.contains(search_lower).any(), axis=1)
        ]

    status_filter = st.multiselect("Filter by Status", STATUS_OPTIONS, default=[])

    if status_filter:
        data = data[data["Status"].isin(status_filter)]

    display = data.copy().sort_values("Date Applied", ascending=False)

    if not display.empty:
        display["Date Applied"] = display["Date Applied"].dt.strftime("%Y-%m-%d")
        st.dataframe(display[DEFAULT_COLUMNS], use_container_width=True, hide_index=True, height=360)
    else:
        st.info("No applications found.")

    st.markdown("## Edit or Delete Application")

    if df.empty:
        st.info("No records available.")
        return

    edit_options = []

    for idx, row in df.sort_values("Date Applied", ascending=False).iterrows():
        label = f"{idx} | {row['Company']} | {row['Role']} | {row['Date Applied'].strftime('%Y-%m-%d')} | {row['Status']}"
        edit_options.append(label)

    selected_record = st.selectbox("Select application", edit_options)
    selected_index = int(selected_record.split("|")[0].strip())
    selected_row = df.loc[selected_index]

    with st.form("edit_application_form"):
        e1, e2, e3 = st.columns(3)

        with e1:
            company = st.text_input("Company", value=selected_row["Company"])
            status = st.selectbox(
                "Status",
                STATUS_OPTIONS,
                index=STATUS_OPTIONS.index(selected_row["Status"]) if selected_row["Status"] in STATUS_OPTIONS else 0,
            )

        with e2:
            role = st.text_input("Role", value=selected_row["Role"])
            source = st.selectbox(
                "Source",
                SOURCE_OPTIONS,
                index=SOURCE_OPTIONS.index(selected_row["Source"]) if selected_row["Source"] in SOURCE_OPTIONS else 0,
            )

        with e3:
            date_applied = st.date_input("Date Applied", value=selected_row["Date Applied"].date())
            location = st.text_input("Location", value=selected_row["Location"])

        next_step = st.text_input("Next Step", value=selected_row["Next Step"])

        save_col, delete_col = st.columns(2)

        with save_col:
            save_clicked = st.form_submit_button("Save Changes", use_container_width=True)

        with delete_col:
            delete_clicked = st.form_submit_button("Delete Record", use_container_width=True)

        if save_clicked:
            update_application(
                selected_index,
                company.strip(),
                role.strip(),
                date_applied.strftime("%Y-%m-%d"),
                status,
                source,
                location.strip(),
                next_step.strip(),
            )
            st.success("Application updated successfully.")
            st.rerun()

        if delete_clicked:
            delete_application(selected_index)
            st.success("Application deleted successfully.")
            st.rerun()


# =========================================================
# DOCUMENTS PAGE
# =========================================================
def render_documents():
    st.markdown("<h1>Documents</h1>", unsafe_allow_html=True)
    st.caption("Track documents, analyze job descriptions, and calculate local CV-to-job match scores in English or German.")

    tab1, tab2, tab3 = st.tabs(
        [
            "📁 Document Library",
            "🧾 Job Description Analyzer",
            "🎯 CV Match Score",
        ]
    )

    with tab1:
        with st.expander("➕ Add Document Record", expanded=True):
            with st.form("add_document_form", clear_on_submit=True):
                d1, d2, d3 = st.columns(3)

                with d1:
                    doc_name = st.text_input("Document Name", placeholder="CV_Data_Science_V1.pdf")
                    doc_type = st.selectbox("Type", DOCUMENT_TYPES)

                with d2:
                    company = st.text_input("Company", placeholder="Infineon / Siemens / General")
                    role = st.text_input("Role", placeholder="Data Science Intern")

                with d3:
                    date_added = st.date_input("Date Added", value=date.today())

                notes = st.text_area("Notes", placeholder="Used for Graz roles / tailored for BI positions")

                submitted = st.form_submit_button("Add Document", use_container_width=True)

                if submitted:
                    if not doc_name.strip():
                        st.error("Please enter the document name.")
                    else:
                        new_doc = pd.DataFrame(
                            [[doc_name.strip(), doc_type, company.strip(), role.strip(), date_added.strftime("%Y-%m-%d"), notes.strip()]],
                            columns=DOCUMENT_COLUMNS,
                        )

                        updated_docs = pd.concat([documents_df[DOCUMENT_COLUMNS], new_doc], ignore_index=True)
                        save_documents(updated_docs)
                        st.success("Document record added.")
                        st.rerun()

        st.markdown("## Document Library")

        docs = load_documents()

        if docs.empty:
            st.info("No documents added yet.")
        else:
            display_docs = docs.copy()
            display_docs["Date Added"] = display_docs["Date Added"].dt.strftime("%Y-%m-%d")
            st.dataframe(display_docs, use_container_width=True, hide_index=True, height=360)

            st.markdown("## Delete Document Record")
            delete_options = []

            for idx, row in docs.iterrows():
                date_text = row["Date Added"].strftime("%Y-%m-%d") if pd.notna(row["Date Added"]) else "No date"
                delete_options.append(f"{idx} | {row['Document Name']} | {row['Type']} | {date_text}")

            selected_doc = st.selectbox("Select document", delete_options)

            if st.button("Delete Document", use_container_width=True):
                selected_index = int(selected_doc.split("|")[0].strip())
                docs = docs.drop(index=selected_index).reset_index(drop=True)
                save_documents(docs)
                st.success("Document deleted.")
                st.rerun()

    with tab2:
        st.markdown("## Job Description Analyzer")
        st.caption("Paste a job description in English or German and get simplified insights locally.")

        job_text = st.text_area(
            "Paste Job Description",
            height=260,
            placeholder="Paste the full job description here...",
            key="jd_analyzer_text",
        )

        if st.button("Analyze Job Description", use_container_width=True):
            if not job_text.strip():
                st.error("Please paste a job description first.")
            else:
                analysis = analyze_job_description(job_text)

                a1, a2, a3 = st.columns(3)

                with a1:
                    st.markdown(f"""<div class="mini-card"><div class="mini-title">Detected Skills</div><div class="mini-value">{len(analysis["skills"])}</div></div>""", unsafe_allow_html=True)

                with a2:
                    st.markdown(f"""<div class="mini-card"><div class="mini-title">Seniority</div><div class="mini-value" style="font-size:18px;">{analysis["seniority"]}</div></div>""", unsafe_allow_html=True)

                with a3:
                    st.markdown(f"""<div class="mini-card"><div class="mini-title">Red Flags</div><div class="mini-value">{len(analysis["red_flags"])}</div></div>""", unsafe_allow_html=True)

                st.markdown("### Simple Summary")
                st.write(analysis["summary"])

                st.markdown("### Required / Important Skills")
                if analysis["skills"]:
                    st.write(", ".join(analysis["skills"]))
                else:
                    st.info("No common technical skills detected from the current skill list.")

                st.markdown("### Main Responsibilities")
                if analysis["responsibilities"]:
                    for item in analysis["responsibilities"]:
                        st.markdown(f"- {item}")
                else:
                    st.info("No clear responsibility sentences detected.")

                st.markdown("### Red Flags")
                if analysis["red_flags"]:
                    for flag in analysis["red_flags"]:
                        st.markdown(f"- ⚠️ {flag}")
                else:
                    st.success("No obvious red flags detected.")

                st.markdown("### Keywords to Consider for Your CV")
                if analysis["skills"]:
                    st.write(", ".join(analysis["skills"][:15]))
                else:
                    st.info("No keywords detected.")

    with tab3:
        st.markdown("## CV ↔ Job Match Score")
        st.caption("Upload your CV and paste a job description. English and German are supported locally.")

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            st.success("Local multilingual semantic model available: paraphrase-multilingual-MiniLM-L12-v2")
        else:
            st.warning("sentence-transformers not available. The app will fall back to TF-IDF similarity.")

        c1, c2 = st.columns([1, 1.4])

        with c1:
            cv_file = st.file_uploader(
                "Upload CV",
                type=["pdf", "docx", "txt"],
                key="cv_match_upload",
            )

        with c2:
            match_job_text = st.text_area(
                "Paste Job Description",
                height=260,
                placeholder="Paste the job description here...",
                key="cv_match_job_text",
            )

        if st.button("Calculate Match Score", use_container_width=True):
            if cv_file is None:
                st.error("Please upload your CV first.")
                return

            if not match_job_text.strip():
                st.error("Please paste the job description first.")
                return

            cv_text = read_uploaded_cv(cv_file)

            if not cv_text.strip():
                st.error("Could not extract text from the CV. Try PDF with selectable text, DOCX, or TXT.")
                return

            semantic = semantic_similarity(cv_text, match_job_text)
            keyword, matched_keywords, missing_keywords = keyword_score(cv_text, match_job_text)
            section, found_sections, missing_sections = section_score(cv_text)
            final_score = final_match_score(semantic, keyword, section)

            s1, s2, s3, s4 = st.columns(4)

            with s1:
                st.markdown(f"""<div class="mini-card"><div class="mini-title">Final Match Score</div><div class="mini-value">{final_score}%</div></div>""", unsafe_allow_html=True)

            with s2:
                st.markdown(f"""<div class="mini-card"><div class="mini-title">Semantic Score</div><div class="mini-value">{semantic}%</div></div>""", unsafe_allow_html=True)

            with s3:
                st.markdown(f"""<div class="mini-card"><div class="mini-title">Keyword Score</div><div class="mini-value">{keyword}%</div></div>""", unsafe_allow_html=True)

            with s4:
                st.markdown(f"""<div class="mini-card"><div class="mini-title">CV Section Score</div><div class="mini-value">{section}%</div></div>""", unsafe_allow_html=True)

            st.markdown("### Interpretation")

            if final_score >= 80:
                st.success("Strong match. Your CV seems well aligned with this job.")
            elif final_score >= 60:
                st.info("Moderate match. Your CV is relevant, but could be tailored better.")
            else:
                st.warning("Weak to moderate match. Consider tailoring your CV more strongly to this job.")

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("### Matched Keywords")
                if matched_keywords:
                    st.write(", ".join(matched_keywords))
                else:
                    st.info("No strong keyword overlap detected.")

                st.markdown("### Found CV Sections")
                if found_sections:
                    st.write(", ".join(found_sections))
                else:
                    st.info("No common CV sections detected.")

            with col_b:
                st.markdown("### Missing Keywords")
                if missing_keywords:
                    st.write(", ".join(missing_keywords))
                else:
                    st.success("No major missing keywords detected from the current skill list.")

                st.markdown("### Missing CV Sections")
                if missing_sections:
                    st.write(", ".join(missing_sections))
                else:
                    st.success("All basic CV sections were detected.")

            st.markdown("### Suggested Improvements")
            suggestions = generate_cv_suggestions(missing_keywords, missing_sections, final_score)

            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")

            with st.expander("Extracted CV Text Preview"):
                st.text(cv_text[:4000])


# =========================================================
# CALENDAR PAGE
# =========================================================
def render_calendar():
    st.markdown("<h1>Calendar</h1>", unsafe_allow_html=True)
    st.caption("A simple calendar-style view for interviews, follow-ups, and next steps.")

    data = df.copy().sort_values("Date Applied", ascending=True)

    event_keywords = "interview|follow|onsite|assignment|call|screen|recruiter|await|response|vorstellungsgespräch|telefon|aufgabe|rückmeldung"
    events = data[
        data["Next Step"].fillna("").str.lower().str.contains(event_keywords, regex=True)
        | data["Status"].isin(["Interview", "Phone Screen", "Screening"])
    ].copy()

    if events.empty:
        st.info("No calendar-related events found yet.")
        return

    events["Date Applied"] = events["Date Applied"].dt.strftime("%Y-%m-%d")

    st.markdown("## Upcoming / Action Items")
    st.dataframe(
        events[["Date Applied", "Company", "Role", "Status", "Next Step", "Location"]],
        use_container_width=True,
        hide_index=True,
        height=380,
    )

    st.markdown("## Timeline by Date")

    grouped = events.groupby("Date Applied")

    for event_date, group in grouped:
        st.markdown(f"### 📅 {event_date}")
        for _, row in group.iterrows():
            st.markdown(
                f"""
                <div class="card">
                    <b>{row['Company']}</b> — {row['Role']}<br>
                    <span class="small-muted">Status: {row['Status']} | Location: {row['Location']}</span><br>
                    <span>{row['Next Step']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)


# =========================================================
# COMPANIES PAGE
# =========================================================
def render_companies():
    st.markdown("<h1>Companies</h1>", unsafe_allow_html=True)
    st.caption("Company-level overview based on your applications.")

    if df.empty:
        st.info("No companies yet.")
        return

    company_summary = (
        df.groupby("Company")
        .agg(
            Applications=("Company", "count"),
            Latest_Date=("Date Applied", "max"),
            Roles=("Role", lambda x: ", ".join(sorted(set(x.astype(str))))),
            Statuses=("Status", lambda x: ", ".join(sorted(set(x.astype(str))))),
            Sources=("Source", lambda x: ", ".join(sorted(set(x.astype(str))))),
            Locations=("Location", lambda x: ", ".join(sorted(set(x.astype(str))))),
        )
        .reset_index()
        .sort_values("Applications", ascending=False)
    )

    company_summary["Latest_Date"] = company_summary["Latest_Date"].dt.strftime("%Y-%m-%d")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""<div class="mini-card"><div class="mini-title">Unique Companies</div><div class="mini-value">{df['Company'].nunique()}</div></div>""", unsafe_allow_html=True)

    with c2:
        top_company = company_summary.iloc[0]["Company"]
        st.markdown(f"""<div class="mini-card"><div class="mini-title">Most Repeated Company</div><div class="mini-value">{top_company}</div></div>""", unsafe_allow_html=True)

    with c3:
        offer_companies = df[df["Status"] == "Offer"]["Company"].nunique()
        st.markdown(f"""<div class="mini-card"><div class="mini-title">Companies with Offers</div><div class="mini-value">{offer_companies}</div></div>""", unsafe_allow_html=True)

    st.markdown("## Company Details")
    st.dataframe(company_summary, use_container_width=True, hide_index=True, height=420)

    st.markdown("## Applications by Company")

    company_counts = df["Company"].value_counts().reset_index()
    company_counts.columns = ["Company", "Applications"]

    fig = px.bar(company_counts, x="Company", y="Applications", text="Applications")
    fig.update_layout(
        height=360,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# =========================================================
# ANALYTICS PAGE
# =========================================================
def render_analytics():
    st.markdown("<h1>Analytics</h1>", unsafe_allow_html=True)
    st.caption("Deeper analytics for your job search performance.")

    if df.empty:
        st.info("No data available.")
        return

    metrics = calculate_metrics(df)

    a1, a2, a3, a4 = st.columns(4)

    with a1:
        st.markdown(f"""<div class="mini-card"><div class="mini-title">Total Applications</div><div class="mini-value">{metrics["total"]}</div></div>""", unsafe_allow_html=True)

    with a2:
        st.markdown(f"""<div class="mini-card"><div class="mini-title">Unique Companies</div><div class="mini-value">{df["Company"].nunique()}</div></div>""", unsafe_allow_html=True)

    with a3:
        st.markdown(f"""<div class="mini-card"><div class="mini-title">Sources Used</div><div class="mini-value">{df["Source"].nunique()}</div></div>""", unsafe_allow_html=True)

    with a4:
        st.markdown(f"""<div class="mini-card"><div class="mini-title">Active Applications</div><div class="mini-value">{metrics["active"]}</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Status Distribution</div>', unsafe_allow_html=True)

        status_counts = df["Status"].value_counts().reindex(STATUS_OPTIONS, fill_value=0).reset_index()
        status_counts.columns = ["Status", "Applications"]

        fig_status = px.bar(
            status_counts,
            x="Status",
            y="Applications",
            text="Applications",
            color="Status",
            color_discrete_map=STATUS_COLORS,
        )
        fig_status.update_layout(
            height=360,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_status, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Source Performance</div>', unsafe_allow_html=True)

        source_status = pd.crosstab(df["Source"], df["Status"]).reset_index()
        st.dataframe(source_status, use_container_width=True, hide_index=True, height=360)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Applications Over Time</div>', unsafe_allow_html=True)

    monthly = df.groupby(df["Date Applied"].dt.to_period("M")).size().reset_index(name="Applications")
    monthly["Month"] = monthly["Date Applied"].dt.strftime("%b %Y")

    fig_line = px.line(monthly, x="Month", y="Applications", markers=True)
    fig_line.update_layout(
        height=350,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# INSIGHTS PAGE
# =========================================================
def render_insights():
    st.markdown("<h1>Insights</h1>", unsafe_allow_html=True)
    st.caption("Full insights from your current job search data.")

    if df.empty:
        st.info("No data available.")
        return

    metrics = calculate_metrics(df)

    i1, i2, i3, i4, i5 = st.columns(5)

    with i1:
        st.markdown(f"""<div class="mini-card"><div class="mini-title">Total Applications</div><div class="mini-value">{metrics["total"]}</div></div>""", unsafe_allow_html=True)

    with i2:
        st.markdown(f"""<div class="mini-card"><div class="mini-title">Response Rate</div><div class="mini-value">{metrics["response_rate"]}%</div></div>""", unsafe_allow_html=True)

    with i3:
        st.markdown(f"""<div class="mini-card"><div class="mini-title">Offer Rate</div><div class="mini-value">{metrics["offer_rate"]}%</div></div>""", unsafe_allow_html=True)

    with i4:
        st.markdown(f"""<div class="mini-card"><div class="mini-title">Rejection Rate</div><div class="mini-value">{metrics["rejection_rate"]}%</div></div>""", unsafe_allow_html=True)

    with i5:
        st.markdown(f"""<div class="mini-card"><div class="mini-title">Active</div><div class="mini-value">{metrics["active"]}</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Best Sources</div>', unsafe_allow_html=True)

        source_counts = df["Source"].value_counts().reset_index()
        source_counts.columns = ["Source", "Applications"]

        fig = px.bar(source_counts, x="Source", y="Applications", text="Applications")
        fig.update_layout(
            height=360,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Status Summary</div>', unsafe_allow_html=True)

        status_counts = df["Status"].value_counts().reindex(STATUS_OPTIONS, fill_value=0).reset_index()
        status_counts.columns = ["Status", "Applications"]

        fig = px.pie(
            status_counts,
            names="Status",
            values="Applications",
            hole=0.55,
            color="Status",
            color_discrete_map=STATUS_COLORS,
        )
        fig.update_layout(
            height=360,
            paper_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Actionable Applications</div>', unsafe_allow_html=True)

    actionable = df[
        df["Next Step"]
        .fillna("")
        .str.lower()
        .str.contains("follow|await|response|recruiter|interview|assignment|call|onsite|rückmeldung|telefon", regex=True)
    ].copy()

    if actionable.empty:
        st.info("No actionable applications found.")
    else:
        actionable["Date Applied"] = actionable["Date Applied"].dt.strftime("%Y-%m-%d")
        st.dataframe(actionable[DEFAULT_COLUMNS], use_container_width=True, hide_index=True, height=320)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# NOTES PAGE
# =========================================================
def render_notes():
    st.markdown("<h1>Notes</h1>", unsafe_allow_html=True)
    st.caption("Keep notes about companies, interviews, follow-ups, and application strategy.")

    with st.expander("➕ Add Note", expanded=True):
        with st.form("add_note_form", clear_on_submit=True):
            n1, n2, n3 = st.columns(3)

            with n1:
                title = st.text_input("Title", placeholder="Follow-up idea")

            with n2:
                company = st.text_input("Company", placeholder="Infineon")

            with n3:
                date_added = st.date_input("Date Added", value=date.today())

            note = st.text_area("Note", placeholder="Write your note here...")

            submitted = st.form_submit_button("Add Note", use_container_width=True)

            if submitted:
                if not title.strip() or not note.strip():
                    st.error("Please enter at least a title and note.")
                else:
                    new_note = pd.DataFrame(
                        [[title.strip(), company.strip(), date_added.strftime("%Y-%m-%d"), note.strip()]],
                        columns=NOTES_COLUMNS,
                    )

                    updated_notes = pd.concat([notes_df[NOTES_COLUMNS], new_note], ignore_index=True)
                    save_notes(updated_notes)
                    st.success("Note added.")
                    st.rerun()

    st.markdown("## Notes List")

    notes = load_notes()

    if notes.empty:
        st.info("No notes added yet.")
    else:
        display_notes = notes.copy()
        display_notes["Date Added"] = display_notes["Date Added"].dt.strftime("%Y-%m-%d")
        st.dataframe(display_notes, use_container_width=True, hide_index=True, height=360)

        st.markdown("## Delete Note")
        delete_options = []

        for idx, row in notes.iterrows():
            date_text = row["Date Added"].strftime("%Y-%m-%d") if pd.notna(row["Date Added"]) else "No date"
            delete_options.append(f"{idx} | {row['Title']} | {row['Company']} | {date_text}")

        selected_note = st.selectbox("Select note", delete_options)

        if st.button("Delete Note", use_container_width=True):
            selected_index = int(selected_note.split("|")[0].strip())
            notes = notes.drop(index=selected_index).reset_index(drop=True)
            save_notes(notes)
            st.success("Note deleted.")
            st.rerun()


# =========================================================
# ROUTER
# =========================================================
if st.session_state.page == "Dashboard":
    render_dashboard()

elif st.session_state.page == "Applications":
    render_applications()

elif st.session_state.page == "Calendar":
    render_calendar()

elif st.session_state.page == "Companies":
    render_companies()

elif st.session_state.page == "Documents":
    render_documents()

elif st.session_state.page == "Analytics":
    render_analytics()

elif st.session_state.page == "Insights":
    render_insights()

elif st.session_state.page == "Notes":
    render_notes()