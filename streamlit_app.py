import re
import streamlit as st
import io
import base64
import json
import os
import random
from datetime import datetime

try:
    import PyPDF2
except ImportError:
    st.error("PyPDF2 is not installed. Please install it with: pip install PyPDF2")

try:
    import docx
except ImportError:
    st.error("python-docx is not installed. Please install it with: pip install python-docx")

from fpdf import FPDF
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    
    NLP_ENABLED = True
except ImportError as e:
    st.warning(f"NLTK not installed: {e}. NLP features will be disabled.")
    NLP_ENABLED = False

COMMON_SKILLS = {
    'programming': ['python', 'java', 'javascript', 'html', 'css', 'c++', 'c#', 'ruby', 'php', 'sql', 'r'],
    'frameworks': ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'node.js', 'express', '.net'],
    'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'sqlite', 'redis'],
    'cloud': ['aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes'],
    'tools': ['git', 'github', 'jira', 'jenkins', 'agile', 'scrum'],
    'soft_skills': ['communication', 'leadership', 'teamwork', 'problem solving', 'time management'],
}

TECHNICAL_QUESTIONS = {
    'python': [
        {"question": "Explain how you would implement a decorator in Python.", 
         "expected_keywords": ["function", "wrapper", "decorator", "@", "arguments", "return"]},
        {"question": "How would you handle exceptions in Python?", 
         "expected_keywords": ["try", "except", "finally", "raise", "error", "handling"]},
        {"question": "Describe the difference between a list and a tuple in Python.", 
         "expected_keywords": ["mutable", "immutable", "list", "tuple", "ordered", "elements"]},
        {"question": "What are generators in Python and how do they differ from regular functions?", 
         "expected_keywords": ["generator", "yield", "iterator", "memory", "lazy", "next"]},
        {"question": "Explain the Global Interpreter Lock (GIL) in Python.", 
         "expected_keywords": ["GIL", "thread", "multithreading", "lock", "CPython", "performance"]},
        {"question": "How do you manage memory in Python?", 
         "expected_keywords": ["garbage collection", "reference counting", "memory", "del", "objects", "heap"]},
        {"question": "What is the difference between deep copy and shallow copy?", 
         "expected_keywords": ["deep copy", "shallow copy", "copy", "reference", "nested", "object"]},
        {"question": "Explain list comprehensions and their advantages.", 
         "expected_keywords": ["list comprehension", "loop", "concise", "performance", "syntax", "filter"]},
    ],
    'java': [
        {"question": "Explain the concept of inheritance in Java.", 
         "expected_keywords": ["extends", "class", "parent", "child", "super", "override"]},
        {"question": "How do you handle exceptions in Java?", 
         "expected_keywords": ["try", "catch", "finally", "throw", "throws", "exception"]},
        {"question": "What is the difference between an interface and an abstract class in Java?", 
         "expected_keywords": ["implement", "extend", "methods", "abstract", "interface", "multiple"]},
        {"question": "Describe how the JVM manages memory.", 
         "expected_keywords": ["JVM", "heap", "stack", "garbage collection", "memory", "allocation"]},
        {"question": "What are Java streams and how are they used?", 
         "expected_keywords": ["stream", "lambda", "functional", "pipeline", "filter", "map"]},
        {"question": "Explain the concept of multithreading in Java.", 
         "expected_keywords": ["thread", "runnable", "synchronize", "concurrency", "lock", "executor"]},
        {"question": "What is the difference between equals() and == in Java?", 
         "expected_keywords": ["equals", "==", "reference", "value", "comparison", "object"]},
        {"question": "How does the Java Collections Framework work?", 
         "expected_keywords": ["collections", "list", "set", "map", "arraylist", "hashmap"]},
    ],
    'javascript': [
        {"question": "Explain closures in JavaScript.", 
         "expected_keywords": ["function", "scope", "variable", "closure", "lexical", "access"]},
        {"question": "How does asynchronous programming work in JavaScript?", 
         "expected_keywords": ["promise", "async", "await", "callback", "then", "event loop"]},
        {"question": "What's the difference between var, let, and const in JavaScript?", 
         "expected_keywords": ["scope", "hoisting", "reassign", "block", "function", "declaration"]},
        {"question": "What is the event loop in JavaScript?", 
         "expected_keywords": ["event loop", "call stack", "queue", "asynchronous", "browser", "execution"]},
        {"question": "Explain the concept of prototypal inheritance.", 
         "expected_keywords": ["prototype", "inheritance", "object", "chain", "__proto__", "constructor"]},
        {"question": "What are arrow functions and how do they differ from regular functions?", 
         "expected_keywords": ["arrow function", "this", "binding", "syntax", "lexical", "concise"]},
        {"question": "How do you handle errors in JavaScript?", 
         "expected_keywords": ["try", "catch", "throw", "error", "promise", "async"]},
        {"question": "What is the difference between null and undefined?", 
         "expected_keywords": ["null", "undefined", "value", "absence", "type", "comparison"]},
    ],
    'sql': [
        {"question": "Explain the difference between INNER JOIN and LEFT JOIN.", 
         "expected_keywords": ["inner", "left", "join", "matching", "all", "records"]},
        {"question": "How would you optimize a slow SQL query?", 
         "expected_keywords": ["index", "execution plan", "query", "optimize", "performance", "analyze"]},
        {"question": "What is database normalization?", 
         "expected_keywords": ["normal form", "redundancy", "dependency", "relation", "table", "normalize"]},
        {"question": "What are indexes and how do they impact performance?", 
         "expected_keywords": ["index", "performance", "lookup", "clustered", "non-clustered", "query"]},
        {"question": "Explain the ACID properties in databases.", 
         "expected_keywords": ["ACID", "atomicity", "consistency", "isolation", "durability", "transaction"]},
        {"question": "What is a stored procedure and when would you use one?", 
         "expected_keywords": ["stored procedure", "database", "execute", "logic", "performance", "security"]},
        {"question": "How do you prevent SQL injection attacks?", 
         "expected_keywords": ["SQL injection", "prepared statement", "parameterized", "sanitize", "escape", "security"]},
        {"question": "What is the difference between a view and a table?", 
         "expected_keywords": ["view", "table", "virtual", "query", "stored", "data"]},
    ],
    'react': [
        {"question": "Explain the component lifecycle in React.", 
         "expected_keywords": ["mount", "update", "unmount", "render", "effect", "component"]},
        {"question": "How do you manage state in React applications?", 
         "expected_keywords": ["useState", "useReducer", "state", "props", "context", "Redux"]},
        {"question": "What are hooks in React and why were they introduced?", 
         "expected_keywords": ["hooks", "functional", "state", "effect", "rules", "useState"]},
        {"question": "What is the virtual DOM and how does it work?", 
         "expected_keywords": ["virtual DOM", "reconciliation", "diffing", "render", "performance", "update"]},
        {"question": "How do you optimize performance in a React application?", 
         "expected_keywords": ["memo", "useCallback", "useMemo", "lazy", "performance", "render"]},
        {"question": "What is the difference between controlled and uncontrolled components?", 
         "expected_keywords": ["controlled", "uncontrolled", "state", "form", "input", "manage"]},
        {"question": "How does React Router work?", 
         "expected_keywords": ["router", "route", "navigation", "component", "path", "history"]},
        {"question": "Explain the concept of higher-order components.", 
         "expected_keywords": ["higher-order", "component", "reuse", "props", "wrapping", "logic"]},
    ],
    'aws': [
        {"question": "Explain the difference between EC2 and Lambda.", 
         "expected_keywords": ["instance", "serverless", "EC2", "Lambda", "scaling", "compute"]},
        {"question": "How do you handle security in AWS?", 
         "expected_keywords": ["IAM", "security group", "encryption", "access", "policy", "role"]},
        {"question": "Describe the AWS services you've worked with.", 
         "expected_keywords": ["S3", "EC2", "Lambda", "RDS", "CloudFront", "DynamoDB"]},
        {"question": "What is VPC and why is it used?", 
         "expected_keywords": ["VPC", "network", "subnet", "isolation", "security", "private"]},
        {"question": "How does auto-scaling work in AWS?", 
         "expected_keywords": ["auto-scaling", "EC2", "load balancer", "group", "policy", "traffic"]},
        {"question": "Explain the difference between S3 and EBS.", 
         "expected_keywords": ["S3", "EBS", "storage", "block", "object", "durability"]},
        {"question": "What is CloudFormation and how is it used?", 
         "expected_keywords": ["CloudFormation", "template", "infrastructure", "stack", "provision", "YAML"]},
        {"question": "How do you monitor AWS resources?", 
         "expected_keywords": ["CloudWatch", "metrics", "logs", "alarms", "monitoring", "dashboard"]},
    ],
}

GENERIC_QUESTIONS = [
    {"question": "Tell me about a challenging project you worked on and how you overcame obstacles.", 
     "expected_keywords": ["challenge", "project", "solution", "overcome", "team", "result"]},
    {"question": "How do you approach learning new technologies?", 
     "expected_keywords": ["learning", "research", "practice", "curiosity", "documentation", "projects"]},
    {"question": "Describe your experience with agile development methodologies.", 
     "expected_keywords": ["agile", "scrum", "sprint", "kanban", "standup", "retrospective"]},
    {"question": "How do you ensure code quality in your projects?", 
     "expected_keywords": ["testing", "review", "standards", "documentation", "refactoring", "clean"]},
    {"question": "What strategies do you use to debug complex issues?", 
     "expected_keywords": ["debug", "breakpoint", "log", "trace", "isolate", "tools"]},
    {"question": "How do you handle conflicting priorities in a project?", 
     "expected_keywords": ["prioritize", "communication", "stakeholder", "deadline", "trade-off", "plan"]},
]

WELCOME_MESSAGES = [
    "Welcome to TechInterviewBot! I'm here to help you practice your technical interview skills.",
    "Hello! I'm your Technical Interview Assistant. Let's prepare you for your next tech interview.",
    "Hi there! Ready to sharpen your technical interview skills? I'm here to help you practice.",
    "Welcome aboard! I'm your AI interview coach. Let's see how well you can handle technical questions."
]

RESUME_PROMPTS = [
    "To get started, please upload your resume or paste its content so I can tailor questions to your skills.",
    "Let's begin by analyzing your resume. Please upload it or paste the text so I can customize the interview.",
    "First, I'll need to see your resume to generate relevant questions. Upload a file or paste the text below.",
    "To create a personalized interview experience, I need to review your resume first. Upload or paste it below."
]

SKILL_MESSAGES = [
    "Great! I've analyzed your resume and identified these key skills:",
    "Thanks for sharing your resume! Based on my analysis, here are the skills I've identified:",
    "Perfect! After reviewing your resume, I've extracted these technical skills:",
    "I've processed your resume and found these skills that we can focus on:"
]

INTERVIEW_START_MESSAGES = [
    "Now let's begin the interview! I'll ask you a series of technical questions related to your skills.",
    "Ready to start? I've prepared some technical questions based on your experience.",
    "Let's dive into the technical interview! I'll ask questions related to your strongest skills.",
    "The interview is about to begin! I'll evaluate your answers to help improve your skills."
]

QUESTION_TRANSITIONS = [
    "Let's move on to the next question:",
    "Here's another question for you:",
    "Now, I'd like to ask you about:",
    "Let's continue with this question:",
    "For the next question:",
    "Moving forward:",
]

EVALUATION_POSITIVE = [
    "Great answer! You've covered the key points effectively.",
    "Excellent response! Your understanding of the concept is clear.",
    "Well done! Your explanation was thorough and accurate.",
    "Very good! You demonstrated strong knowledge in this area."
]

EVALUATION_AVERAGE = [
    "Good attempt! You covered some key points, but there's room for improvement.",
    "That's a decent answer, but you could expand on a few concepts.",
    "Not bad! You have the basic understanding, but consider adding more depth.",
    "You're on the right track, but try to be more specific in your explanations."
]

EVALUATION_NEEDS_IMPROVEMENT = [
    "You've made an attempt, but there are some key concepts missing.",
    "Your answer needs more technical depth. Let me suggest some areas to focus on.",
    "I see you have some understanding, but there are important points you didn't address.",
    "This response could be improved by including more specific technical details."
]

def preprocess_text(text):
    if not NLP_ENABLED:
        return text.lower()
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token.isalnum()]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)

def extract_skills_with_nlp(text):
    """Extract skills with very strict context-based matching."""
    if not NLP_ENABLED:
        return extract_skills_basic(text)
    
    raw_text = text.lower()
    identified_skills = {}
    debug_matches = []
    
    SKILL_ALIASES = {
        'python': ['python'],
        'java': ['java'],
        'javascript': ['javascript'],
        'sql': ['sql'],
        'aws': ['aws'],
        'react': ['react'],
        'django': ['django'],
        'flask': ['flask'],
        'git': ['git'],
        'github': ['github'],
    }
    
    context_patterns = r'(?:skills|experience|proficient in|worked with|knowledge of|using|expertise in|developed with)'
    
    for category, skill_list in COMMON_SKILLS.items():
        found_skills = set()
        for skill in skill_list:
            aliases = SKILL_ALIASES.get(skill, [skill])
            for alias in aliases:
                pattern = rf'{context_patterns}\s*[^.\n]*\b{re.escape(alias)}\b[^.\n]*'
                matches = re.finditer(pattern, raw_text)
                for match in matches:
                    found_skills.add(skill)
                    debug_matches.append(f"Matched '{skill}' in: '{match.group()}'")
        if found_skills:
            identified_skills[category] = list(found_skills)
    
    st.session_state.debug_skills = debug_matches
    st.session_state.raw_resume_text = raw_text
    return identified_skills

def extract_skills_basic(text):
    """Basic skill extraction with strict context."""
    text = text.lower()
    identified_skills = {}
    context_patterns = r'(?:skills|experience|proficient in|worked with|knowledge of|using|expertise in|developed with)'
    
    for category, skill_list in COMMON_SKILLS.items():
        found_skills = set()
        for skill in skill_list:
            pattern = rf'{context_patterns}\s*[^.\n]*\b{re.escape(skill)}\b[^.\n]*'
            if re.search(pattern, text):
                found_skills.add(skill)
        if found_skills:
            identified_skills[category] = list(found_skills)
    return identified_skills

def extract_skills(text):
    if not text:
        return {}
    return extract_skills_with_nlp(text) if NLP_ENABLED else extract_skills_basic(text)

def evaluate_answer_with_nlp(question, answer, expected_keywords):
    """Evaluate an answer using NLP techniques"""
    if not answer.strip():
        return {"score": 0, "feedback": "No answer provided.", "missing_concepts": expected_keywords}
    
    if not NLP_ENABLED:
        keyword_count = sum(1 for keyword in expected_keywords if keyword.lower() in answer.lower())
        score = min(keyword_count / len(expected_keywords), 1.0) * 100
        missing = [k for k in expected_keywords if k.lower() not in answer.lower()]
        return {"score": score, "feedback": "Basic keyword matching applied.", "missing_concepts": missing}
    
    processed_answer = preprocess_text(answer)
    processed_keywords = [preprocess_text(kw) for kw in expected_keywords]
    
    keyword_count = sum(1 for kw in processed_keywords if kw in processed_answer)
    score = min(keyword_count / len(expected_keywords), 1.0) * 100
    missing = [kw for kw in expected_keywords if preprocess_text(kw) not in processed_answer]
    
    feedback = get_feedback_message(score)
    if missing:
        feedback += f" Consider mentioning: {', '.join(missing[:3])}."
    
    sentences = answer.split('.')
    avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / max(1, len([s for s in sentences if s.strip()]))
    
    if len(answer.split()) > 100 and score > 70:
        feedback += " Your answer shows good technical depth."
    elif len(answer.split()) < 30 and score < 70:
        feedback += " Consider providing a more detailed explanation."
    
    explanation_patterns = ["because", "therefore", "means that", "this is why", "which is"]
    has_explanations = any(pattern in answer.lower() for pattern in explanation_patterns)
    
    if has_explanations and score > 50:
        feedback += " Your explanations help demonstrate understanding."
    elif not has_explanations and score > 50:
        feedback += " Consider explaining 'why' in addition to 'what'."
    
    return {
        "score": score, 
        "feedback": feedback, 
        "missing_concepts": missing
    }

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return ""

def extract_text_from_docx(docx_file):
    try:
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error processing DOCX: {e}")
        return ""

def generate_technical_questions(skills, max_questions=7):
    all_possible_questions = []
    all_skills = [skill for category, skill_list in skills.items() for skill in skill_list]
    
    for skill in set(all_skills):
        if skill in TECHNICAL_QUESTIONS:
            skill_questions = TECHNICAL_QUESTIONS[skill].copy()
            random.shuffle(skill_questions)
            all_possible_questions.extend(skill_questions)
    
    random.shuffle(all_possible_questions)
    
    unique_questions = []
    question_texts = set()
    for q in all_possible_questions:
        if q["question"] not in question_texts:
            unique_questions.append(q)
            question_texts.add(q["question"])
            if len(unique_questions) >= max_questions:
                break
    
    if len(unique_questions) < max_questions:
        generic = GENERIC_QUESTIONS.copy()
        random.shuffle(generic)
        for q in generic:
            if q["question"] not in question_texts:
                unique_questions.append(q)
                question_texts.add(q["question"])
                if len(unique_questions) >= max_questions:
                    break
    
    return unique_questions[:max_questions]

def get_download_link(text, filename, label="Download"):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{label}</a>'

def get_feedback_message(score):
    if score >= 80:
        return random.choice(EVALUATION_POSITIVE)
    elif score >= 60:
        return random.choice(EVALUATION_AVERAGE)
    else:
        return random.choice(EVALUATION_NEEDS_IMPROVEMENT)

def format_skills_message(skills):
    message = ""
    for category, skill_list in skills.items():
        message += f"**{category.capitalize()}**: {', '.join(skill_list)}\n"
    return message

def export_results_as_pdf(candidate_name, interview_date, avg_score, rating, skills, evaluations, questions):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Technical Interview Results", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Candidate: {candidate_name}", ln=True)
    pdf.cell(0, 10, f"Date: {interview_date}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Summary", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Overall Score: {avg_score:.1f}/100", ln=True)
    pdf.cell(0, 10, f"Rating: {rating}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Skills", ln=True)
    pdf.set_font("Arial", "", 12)
    for category, skill_list in skills.items():
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"{category.capitalize()}", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, ", ".join(skill_list))
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Interview Questions and Evaluations", ln=True)
    for i, q in enumerate(questions):
        if q['question'] in evaluations:
            data = evaluations[q['question']]
            evaluation = data["evaluation"]
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Question {i+1}: {q['question']}", ln=True)
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 10, f"Answer: {data['answer']}")
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Score: {evaluation.get('score', 'N/A')}/100", ln=True)
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 10, f"Feedback: {evaluation.get('feedback', 'No feedback available')}")
            missing = evaluation.get('missing_concepts', [])
            if missing:
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "Missing concepts:", ln=True)
                pdf.set_font("Arial", "", 12)
                for concept in missing:
                    pdf.cell(0, 10, f"- {concept}", ln=True)
            pdf.ln(5)
    output_path = "interview_results.pdf"
    pdf.output(output_path)
    return output_path

def generate_interview_summary(candidate_name, interview_date, avg_score, rating, skills, evaluations, questions):
    summary = []
    summary.append(f"# Technical Interview Results for {candidate_name}")
    summary.append(f"**Date:** {interview_date}")
    summary.append(f"**Overall Score:** {avg_score:.1f}/100")
    summary.append(f"**Rating:** {rating}")
    summary.append("\n## Skills Profile")
    for category, skill_list in skills.items():
        summary.append(f"**{category.capitalize()}:** {', '.join(skill_list)}")
    summary.append("\n## Question Analysis")
    for i, q in enumerate(questions):
        if q['question'] in evaluations:
            data = evaluations[q['question']]
            evaluation = data["evaluation"]
            score = evaluation.get('score', 0)
            summary.append(f"### Question {i+1}: {q['question']}")
            summary.append(f"**Score:** {score}/100")
            summary.append(f"**Feedback:** {evaluation.get('feedback', 'No feedback available')}")
            missing = evaluation.get('missing_concepts', [])
            if missing:
                summary.append("**Areas for improvement:**")
                for concept in missing:
                    summary.append(f"- {concept}")
            summary.append("")
    summary.append("## Interview Recommendation")
    if avg_score >= 85:
        summary.append("Based on your technical interview performance, you demonstrate strong technical knowledge and communication skills.")
    elif avg_score >= 70:
        summary.append("Your technical skills are solid, with some areas that could benefit from deeper understanding.")
    elif avg_score >= 50:
        summary.append("You have a good foundation of technical knowledge, but should continue to build your expertise.")
    else:
        summary.append("Consider spending more time studying the fundamentals of your technical areas.")
    return "\n".join(summary)

st.set_page_config(page_title="Technical Interview Chatbot", layout="wide")

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "skills" not in st.session_state:
    st.session_state.skills = {}
if "questions" not in st.session_state:
    st.session_state.questions = []
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0
if "evaluations" not in st.session_state:
    st.session_state.evaluations = {}
if "interview_complete" not in st.session_state:
    st.session_state.interview_complete = False
if "interview_date" not in st.session_state:
    st.session_state.interview_date = datetime.now().strftime("%Y-%m-%d %H:%M")
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [{"role": "assistant", "content": random.choice(WELCOME_MESSAGES) + " " + random.choice(RESUME_PROMPTS)}]
if "candidate_name" not in st.session_state:
    st.session_state.candidate_name = ""
if "bot_state" not in st.session_state:
    st.session_state.bot_state = "wait_for_resume"
if "max_questions" not in st.session_state:
    st.session_state.max_questions = 5
if "debug_skills" not in st.session_state:
    st.session_state.debug_skills = []
if "raw_resume_text" not in st.session_state:
    st.session_state.raw_resume_text = ""

def add_message(role, content):
    st.session_state.chat_messages.append({"role": role, "content": content})

with st.sidebar:
    st.header("Interview Bot Settings")
    if st.session_state.bot_state in ["wait_for_resume", "analyzing_resume"]:
        st.info("Please upload your resume or paste its content to begin.")
    elif st.session_state.bot_state == "interview":
        st.subheader("Interview Progress")
        progress = st.session_state.current_question_index / len(st.session_state.questions)
        st.progress(progress)
        st.write(f"Question {st.session_state.current_question_index}/{len(st.session_state.questions)}")
        if st.session_state.skills:
            st.subheader("Your Skills Focus")
            for category, skills in st.session_state.skills.items():
                with st.expander(category.capitalize()):
                    st.write(", ".join(skills))
    elif st.session_state.bot_state == "complete":
        st.success("Interview Complete!")
        evaluations = st.session_state.evaluations
        total_score = sum(data["evaluation"].get("score", 0) for data in evaluations.values())
        avg_score = total_score / len(evaluations) if evaluations else 0
        rating = "Excellent" if avg_score >= 85 else "Good" if avg_score >= 70 else "Average" if avg_score >= 50 else "Needs Improvement"
        st.metric("Overall Score", f"{avg_score:.1f}/100")
        st.metric("Rating", rating)
    
    max_q = st.slider("Number of Questions", min_value=3, max_value=10, value=st.session_state.max_questions)
    if max_q != st.session_state.max_questions:
        st.session_state.max_questions = max_q
    
    if st.button("Start New Interview"):
        st.session_state.resume_text = ""
        st.session_state.skills = {}
        st.session_state.questions = []
        st.session_state.current_question_index = 0
        st.session_state.evaluations = {}
        st.session_state.interview_complete = False
        st.session_state.bot_state = "wait_for_resume"
        st.session_state.chat_messages = [{"role": "assistant", "content": random.choice(WELCOME_MESSAGES) + " " + random.choice(RESUME_PROMPTS)}]
        st.session_state.candidate_name = ""
        st.session_state.debug_skills = []
        st.session_state.raw_resume_text = ""
        st.rerun()

st.title("Technical Interview Chatbot 🤖")

uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"], key="resume_upload")
chat_container = st.container()

def process_user_input(user_input):
    add_message("user", user_input)
    
    if st.session_state.bot_state == "wait_for_resume":
        if len(user_input.split()) <= 3 and len(st.session_state.chat_messages) <= 3:
            st.session_state.candidate_name = user_input
            add_message("assistant", f"Nice to meet you, {user_input}! Please upload your resume or paste its content so I can prepare relevant technical questions.")
            return
        
        st.session_state.resume_text = user_input
        st.session_state.bot_state = "analyzing_resume"
        add_message("assistant", "Thanks for sharing your resume! I'm analyzing it to identify your technical skills...")
        
        skills = extract_skills(user_input)
        if not skills:
            add_message("assistant", "I couldn't identify specific technical skills from your resume. Let's add some manually. What are your top technical skills? (e.g., Python, Java, AWS)")
            st.session_state.bot_state = "manual_skills"
        else:
            st.session_state.skills = skills
            skill_message = random.choice(SKILL_MESSAGES) + "\n\n" + format_skills_message(skills)
            if st.session_state.debug_skills:
                skill_message += "\n\n**Debug Info:**\n" + "\n".join(st.session_state.debug_skills)
            skill_message += f"\n\n**Raw Resume Text (first 200 chars):** {st.session_state.raw_resume_text[:200]}..."
            skill_message += "\n\nAre these skills accurate? You can add more skills if needed, or type 'start interview' when you're ready."
            add_message("assistant", skill_message)
            st.session_state.bot_state = "confirm_skills"
    
    elif st.session_state.bot_state == "manual_skills":
        skills_input = user_input.lower()
        manual_skills = {}
        for category, skill_list in COMMON_SKILLS.items():
            found_skills = []
            for skill in skill_list:
                if skill in skills_input:
                    found_skills.append(skill)
            if found_skills:
                manual_skills[category] = found_skills
        
        if not manual_skills:
            manual_skills['programming'] = ['python', 'java']
            manual_skills['tools'] = ['git']
        
        st.session_state.skills = manual_skills
        skill_message = "Thanks! I've added these skills to your profile:\n\n" + format_skills_message(manual_skills)
        skill_message += "\n\nReady to start the interview? Type 'start interview' when you're ready."
        add_message("assistant", skill_message)
        st.session_state.bot_state = "confirm_skills"
    
    elif st.session_state.bot_state == "confirm_skills":
        if "start interview" in user_input.lower() or "ready" in user_input.lower() or "yes" in user_input.lower():
            technical_questions = generate_technical_questions(st.session_state.skills, st.session_state.max_questions)
            st.session_state.questions = technical_questions
            st.session_state.current_question_index = 0
            start_message = random.choice(INTERVIEW_START_MESSAGES)
            first_question = technical_questions[0]["question"] if technical_questions else "Tell me about your background in technology."
            add_message("assistant", f"{start_message}\n\n**Question 1:** {first_question}")
            st.session_state.bot_state = "interview"
        else:
            new_skills = extract_skills(user_input)
            if new_skills:
                for category, skills_list in new_skills.items():
                    if category in st.session_state.skills:
                        for skill in skills_list:
                            if skill not in st.session_state.skills[category]:
                                st.session_state.skills[category].append(skill)
                    else:
                        st.session_state.skills[category] = skills_list
                add_message("assistant", f"I've updated your skills profile. Type 'start interview' when you're ready to begin.")
            else:
                add_message("assistant", "I'm ready whenever you are. Type 'start interview' to begin.")
    
    elif st.session_state.bot_state == "interview":
        current_index = st.session_state.current_question_index
        if current_index < len(st.session_state.questions):
            current_question = st.session_state.questions[current_index]
            evaluation = evaluate_answer_with_nlp(
                current_question["question"],
                user_input,
                current_question["expected_keywords"]
            )
            st.session_state.evaluations[current_question["question"]] = {
                "answer": user_input,
                "evaluation": evaluation
            }
            st.session_state.current_question_index += 1
            
            if st.session_state.current_question_index < len(st.session_state.questions):
                next_question = st.session_state.questions[st.session_state.current_question_index]
                add_message("assistant", f"{evaluation['feedback']}\n\n{random.choice(QUESTION_TRANSITIONS)}\n\n**Question {st.session_state.current_question_index + 1}:** {next_question['question']}")
            else:
                st.session_state.bot_state = "complete"
                evaluations = st.session_state.evaluations
                total_score = sum(data["evaluation"].get("score", 0) for data in evaluations.values())
                avg_score = total_score / len(evaluations) if evaluations else 0
                rating = "Excellent" if avg_score >= 85 else "Good" if avg_score >= 70 else "Average" if avg_score >= 50 else "Needs Improvement"
                
                summary = generate_interview_summary(
                    st.session_state.candidate_name or "Candidate",
                    st.session_state.interview_date,
                    avg_score,
                    rating,
                    st.session_state.skills,
                    st.session_state.evaluations,
                    st.session_state.questions
                )
                
                add_message("assistant", f"Interview complete! Your overall score is {avg_score:.1f}/100 ({rating}).\n\nHere's your detailed summary:\n\n{summary}")
                
                pdf_path = export_results_as_pdf(
                    st.session_state.candidate_name or "Candidate",
                    st.session_state.interview_date,
                    avg_score,
                    rating,
                    st.session_state.skills,
                    st.session_state.evaluations,
                    st.session_state.questions
                )
                
                with open(pdf_path, "rb") as f:
                    st.session_state.pdf_data = f.read()
                
                add_message("assistant", "You can download your interview results as a PDF below.")

with chat_container:
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if st.session_state.bot_state == "complete" and "pdf_data" in st.session_state:
        st.download_button(
            label="Download Interview Results (PDF)",
            data=st.session_state.pdf_data,
            file_name="interview_results.pdf",
            mime="application/pdf"
        )

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        st.session_state.resume_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        st.session_state.resume_text = extract_text_from_docx(uploaded_file)
    st.session_state.bot_state = "analyzing_resume"
    add_message("assistant", "Thanks for uploading your resume! I'm analyzing it to identify your technical skills...")
    process_user_input(st.session_state.resume_text)
    st.rerun()

if st.session_state.bot_state != "complete":
    user_input = st.chat_input("Your response...")
    if user_input:
        process_user_input(user_input)
        st.rerun()
