import re
import streamlit as st
import io
import base64
import json
import requests
import os
import random
from datetime import datetime
# import en_core_web_sm

# For PDF and DOCX processing
try:
    import PyPDF2
except ImportError:
    st.error("PyPDF2 is not installed. Please install it with: pip install PyPDF2")

try:
    import docx
except ImportError:
    st.error("python-docx is not installed. Please install it with: pip install python-docx")

# For email and PDF generation
from fpdf import FPDF
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# NLP Modules
# Prioritize NLP packages with better error handling
try:
    import spacy
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import numpy as np
    
    # Add newer NLP packages
    import torch
    from sentence_transformers import SentenceTransformer
    
    # Download resources once
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    
    # Load spaCy model - try larger model first, fall back to smaller one
    try:
        nlp = spacy.load("en_core_web_md")  # Medium model with word vectors
    except:
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
    
    # Initialize more advanced NLP components
    try:
        # Sentence embeddings model for semantic similarity
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Multiple NLP pipelines for different tasks
        sentiment_analyzer = pipeline("sentiment-analysis")
        token_classifier = pipeline("token-classification", aggregation_strategy="simple")
        text_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        ADVANCED_NLP_ENABLED = True
    except Exception as e:
        st.warning(f"Advanced NLP models couldn't be loaded: {e}. Using basic models instead.")
        ADVANCED_NLP_ENABLED = False
        
    NLP_ENABLED = True
except ImportError as e:
    st.warning(f"Some NLP packages are not installed: {e}. Using basic text processing instead.")
    NLP_ENABLED = False
    ADVANCED_NLP_ENABLED = False
# Define skills dictionary
COMMON_SKILLS = {
    'programming': ['python', 'java', 'javascript', 'html', 'css', 'c++', 'c#', 'ruby', 'php', 'sql', 'r', 'golang', 'rust', 'swift', 'kotlin'],
    'frameworks': ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'node.js', 'express', '.net', 'laravel', 'rails', 'fastapi'],
    'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'sqlite', 'redis', 'cassandra', 'dynamodb', 'couchbase'],
    'cloud': ['aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'serverless', 'lambda', 'terraform', 'devops'],
    'tools': ['git', 'github', 'jira', 'jenkins', 'agile', 'scrum', 'ci/cd', 'gitlab', 'bitbucket', 'confluence'],
    'soft_skills': ['communication', 'leadership', 'teamwork', 'problem solving', 'time management', 'collaboration', 'critical thinking'],
    'data_science': ['machine learning', 'deep learning', 'neural networks', 'nlp', 'computer vision', 'data analysis', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras'],
    'mobile_dev': ['android', 'ios', 'react native', 'flutter', 'swift', 'kotlin', 'mobile development', 'app development'],
    'testing': ['unit testing', 'integration testing', 'selenium', 'jest', 'pytest', 'jasmine', 'mocha', 'cucumber', 'qa'],
}

# Define custom technical questions based on skills
TECHNICAL_QUESTIONS = {
    'python': [
        {"question": "Explain how you would implement a decorator in Python.", 
         "expected_keywords": ["function", "wrapper", "decorator", "@", "arguments", "return"]},
        {"question": "How would you handle exceptions in Python?", 
         "expected_keywords": ["try", "except", "finally", "raise", "error", "handling"]},
        {"question": "Describe the difference between a list and a tuple in Python.", 
         "expected_keywords": ["mutable", "immutable", "list", "tuple", "ordered", "elements"]},
        {"question": "Explain Python's Global Interpreter Lock (GIL) and its implications.",
         "expected_keywords": ["thread", "lock", "interpreter", "multi-threading", "concurrency", "performance"]},
        {"question": "How would you implement a simple REST API in Python?",
         "expected_keywords": ["flask", "django", "fastapi", "routes", "http", "json", "request", "response"]}
    ],
    'java': [
        {"question": "Explain the concept of inheritance in Java.", 
         "expected_keywords": ["extends", "class", "parent", "child", "super", "override"]},
        {"question": "How do you handle exceptions in Java?", 
         "expected_keywords": ["try", "catch", "finally", "throw", "throws", "exception"]},
        {"question": "What is the difference between an interface and an abstract class in Java?", 
         "expected_keywords": ["implement", "extend", "methods", "abstract", "interface", "multiple"]},
        {"question": "Explain Java's memory management and garbage collection.",
         "expected_keywords": ["heap", "stack", "garbage collector", "JVM", "memory", "reference"]},
        {"question": "How do you achieve thread safety in Java?",
         "expected_keywords": ["synchronized", "lock", "atomic", "volatile", "concurrent", "thread-safe"]}
    ],
    'javascript': [
        {"question": "Explain closures in JavaScript.", 
         "expected_keywords": ["function", "scope", "variable", "closure", "lexical", "access"]},
        {"question": "How does asynchronous programming work in JavaScript?", 
         "expected_keywords": ["promise", "async", "await", "callback", "then", "event loop"]},
        {"question": "What's the difference between var, let, and const in JavaScript?", 
         "expected_keywords": ["scope", "hoisting", "reassign", "block", "function", "declaration"]},
        {"question": "Explain prototypical inheritance in JavaScript.",
         "expected_keywords": ["prototype", "object", "inheritance", "__proto__", "constructor", "chain"]},
        {"question": "How does the 'this' keyword work in JavaScript?",
         "expected_keywords": ["context", "bind", "apply", "call", "arrow function", "object"]}
    ],
    'sql': [
        {"question": "Explain the difference between INNER JOIN and LEFT JOIN.", 
         "expected_keywords": ["inner", "left", "join", "matching", "all", "records"]},
        {"question": "How would you optimize a slow SQL query?", 
         "expected_keywords": ["index", "execution plan", "query", "optimize", "performance", "analyze"]},
        {"question": "What is database normalization?", 
         "expected_keywords": ["normal form", "redundancy", "dependency", "relation", "table", "normalize"]},
        {"question": "Explain the concept of database transactions and ACID properties.",
         "expected_keywords": ["atomic", "consistent", "isolated", "durable", "commit", "rollback"]},
        {"question": "How would you design a database schema for a social media application?",
         "expected_keywords": ["tables", "relationships", "foreign key", "users", "posts", "comments"]}
    ],
    'react': [
        {"question": "Explain the component lifecycle in React.", 
         "expected_keywords": ["mount", "update", "unmount", "render", "effect", "component"]},
        {"question": "How do you manage state in React applications?", 
         "expected_keywords": ["useState", "useReducer", "state", "props", "context", "Redux"]},
        {"question": "What are hooks in React and why were they introduced?", 
         "expected_keywords": ["hooks", "functional", "state", "effect", "rules", "useState"]},
        {"question": "Explain the virtual DOM in React and its benefits.",
         "expected_keywords": ["virtual DOM", "reconciliation", "diff", "performance", "render", "update"]},
        {"question": "How do you optimize performance in React applications?",
         "expected_keywords": ["memo", "useMemo", "useCallback", "key", "shouldComponentUpdate", "pure component"]}
    ],
    'aws': [
        {"question": "Explain the difference between EC2 and Lambda.", 
         "expected_keywords": ["instance", "serverless", "EC2", "Lambda", "scaling", "compute"]},
        {"question": "How do you handle security in AWS?", 
         "expected_keywords": ["IAM", "security group", "encryption", "access", "policy", "role"]},
        {"question": "Describe the AWS services you've worked with.", 
         "expected_keywords": ["S3", "EC2", "Lambda", "RDS", "CloudFront", "DynamoDB"]},
        {"question": "Explain the concept of infrastructure as code in AWS.",
         "expected_keywords": ["CloudFormation", "Terraform", "template", "stack", "provision", "automation"]},
        {"question": "How do you design a highly available architecture in AWS?",
         "expected_keywords": ["availability zone", "region", "auto-scaling", "load balancer", "redundancy", "failover"]}
    ],
    'machine learning': [
        {"question": "Explain the difference between supervised and unsupervised learning.",
         "expected_keywords": ["labeled", "unlabeled", "classifier", "clustering", "training data", "patterns"]},
        {"question": "How would you handle overfitting in a machine learning model?",
         "expected_keywords": ["regularization", "cross-validation", "dropout", "pruning", "generalization", "training data"]},
        {"question": "Explain the concept of gradient descent in neural networks.",
         "expected_keywords": ["optimization", "weights", "backpropagation", "learning rate", "cost function", "minimum"]},
        {"question": "What evaluation metrics would you use for a classification problem?",
         "expected_keywords": ["accuracy", "precision", "recall", "F1", "ROC", "confusion matrix"]},
        {"question": "How do you approach feature selection and engineering?",
         "expected_keywords": ["correlation", "importance", "dimensionality", "transformation", "selection", "domain knowledge"]}
    ],
    'nlp': [
        {"question": "Explain the concept of word embeddings in NLP.",
         "expected_keywords": ["vector", "semantic", "word2vec", "GloVe", "representation", "embedding"]},
        {"question": "How would you build a simple text classification system?",
         "expected_keywords": ["tokenization", "features", "classifier", "bag of words", "preprocessing", "model"]},
        {"question": "Explain how transformer models work in NLP.",
         "expected_keywords": ["attention", "self-attention", "BERT", "GPT", "transformer", "encoder-decoder"]},
        {"question": "What are the challenges in implementing a chatbot?",
         "expected_keywords": ["context", "understanding", "generation", "intent", "entity", "dialogue"]},
        {"question": "How would you approach named entity recognition?",
         "expected_keywords": ["entities", "extraction", "recognition", "model", "tagging", "sequence"]}
    ],
}

# Add more questions for data science, cloud, etc.

# Define generic questions that can be asked for any role
GENERIC_QUESTIONS = [
    {"question": "Tell me about a challenging project you worked on and how you overcame obstacles.", 
     "expected_keywords": ["challenge", "project", "solution", "overcome", "team", "result"]},
    {"question": "How do you approach learning new technologies?", 
     "expected_keywords": ["learning", "research", "practice", "curiosity", "documentation", "projects"]},
    {"question": "Describe your experience with agile development methodologies.", 
     "expected_keywords": ["agile", "scrum", "sprint", "kanban", "standup", "retrospective"]},
    {"question": "How do you ensure code quality in your projects?", 
     "expected_keywords": ["testing", "review", "standards", "documentation", "refactoring", "clean"]},
    {"question": "How do you handle disagreements with team members?",
     "expected_keywords": ["communication", "listen", "compromise", "resolution", "perspective", "discussion"]},
    {"question": "Describe a time when you had to meet a tight deadline.",
     "expected_keywords": ["prioritize", "planning", "time management", "pressure", "delivery", "schedule"]},
    {"question": "How do you stay updated with the latest technologies?",
     "expected_keywords": ["learning", "blogs", "courses", "community", "practice", "projects"]},
    {"question": "What's your approach to debugging complex issues?",
     "expected_keywords": ["systematic", "logs", "reproduce", "isolate", "root cause", "testing"]}
]

# Chatbot welcome messages and prompts
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

# NLP Functions
def preprocess_text(text):
    """Preprocess text for NLP analysis"""
    if not NLP_ENABLED:
        return text.lower()
    
    # Convert to lowercase and tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and special characters
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)
def recognize_entities_in_answer(answer):
    """Extract named entities from the answer for context-aware feedback"""
    if not NLP_ENABLED:
        return {}
    
    try:
        doc = nlp(answer)
        entities = {}
        
        # Extract entities by type
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        
        # Look for technical terms
        technical_terms = []
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                for category, skills in COMMON_SKILLS.items():
                    if any(skill in token.text.lower() for skill in skills):
                        technical_terms.append(token.text)
                        break
        
        if technical_terms:
            entities["TECH"] = technical_terms
        
        return entities
    except Exception as e:
        st.error(f"Error in entity recognition: {str(e)}")
        return {}
def extract_entities_with_nlp(text):
    """Extract named entities and potential skills from resume text using spaCy"""
    if not NLP_ENABLED or not text:
        return {}
    
    doc = nlp(text)
    entities = {}
    
    # Extract organizations, products, and tech-related entities
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]:
            entities[ent.text.lower()] = ent.label_
    
    # Look for tech skills based on noun chunks
    tech_skills = []
    for chunk in doc.noun_chunks:
        if chunk.text.lower() in [skill for sublist in COMMON_SKILLS.values() for skill in sublist]:
            tech_skills.append(chunk.text.lower())
    
    if tech_skills:
        entities["tech_skills"] = list(set(tech_skills))
    
    return entities

def analyze_answer_sentiment(answer):
    """Analyze the sentiment and confidence of an answer"""
    if not NLP_ENABLED or sentiment_analyzer is None:
        # Simple fallback based on word count and confidence markers
        confidence_markers = ["confident", "sure", "know", "certain", "definitely"]
        uncertainty_markers = ["maybe", "perhaps", "guess", "not sure", "might", "could be"]
        
        answer_lower = answer.lower()
        confidence_count = sum(1 for marker in confidence_markers if marker in answer_lower)
        uncertainty_count = sum(1 for marker in uncertainty_markers if marker in answer_lower)
        
        if confidence_count > uncertainty_count:
            return {"sentiment": "confident", "score": 0.8}
        elif uncertainty_count > confidence_count:
            return {"sentiment": "uncertain", "score": 0.4}
        else:
            return {"sentiment": "neutral", "score": 0.6}
    
    try:
        # Use sentiment analysis pipeline
        result = sentiment_analyzer(answer[:512])[0]  # Limit text length to model max
        return {
            "sentiment": result["label"],
            "score": result["score"]
        }
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
        return {"sentiment": "neutral", "score": 0.5}

def calculate_answer_similarity(expected_keywords, answer):
    """Calculate similarity between expected keywords and answer using TF-IDF and cosine similarity"""
    if not NLP_ENABLED:
        # Simple fallback using keyword matching
        answer_lower = answer.lower()
        match_count = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
        return min(match_count / max(len(expected_keywords), 1), 1.0)
    
    try:
        # Create a document list with expected keywords and answer
        documents = [" ".join(expected_keywords), answer]
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except Exception as e:
        st.error(f"Error in similarity calculation: {str(e)}")
        return 0.5

def extract_skills_with_nlp(text):
    """Extract skills from resume using advanced NLP techniques"""
    if not NLP_ENABLED:
        return extract_skills(text)  # Fallback to basic extraction
    
    identified_skills = {}
    
    # Process text with spaCy
    doc = nlp(text)
    
    # Extract potential skill candidates
    skill_candidates = set()
    
    # Extract noun phrases, entities, and technical terms
    for chunk in doc.noun_chunks:
        if len(chunk.text.strip()) > 2:  # Avoid single letters/numbers
            skill_candidates.add(chunk.text.lower())
    
    # Add named entities of relevant types
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "GPE"]:
            skill_candidates.add(ent.text.lower())
    
    # Extract potential technical terms using POS patterns
    for token in doc:
        # Look for technical terms that are nouns or proper nouns
        if (token.pos_ in ["NOUN", "PROPN"] and 
            not token.is_stop and 
            len(token.text) > 2):
            skill_candidates.add(token.text.lower())
    
    # Use sentence transformer for semantic matching if available
    if ADVANCED_NLP_ENABLED and 'sentence_model' in globals():
        # Get embeddings for all skill candidates
        candidate_embeddings = sentence_model.encode([cand for cand in skill_candidates])
        
        # Match with known skills using semantic similarity
        for category, skill_list in COMMON_SKILLS.items():
            found_skills = []
            skill_embeddings = sentence_model.encode(skill_list)
            
            # Calculate similarities
            for i, candidate in enumerate(skill_candidates):
                candidate_emb = candidate_embeddings[i]
                similarities = cosine_similarity([candidate_emb], skill_embeddings)[0]
                
                # Find highest similarity
                max_sim_idx = np.argmax(similarities)
                max_sim = similarities[max_sim_idx]
                
                # If similarity is high enough, add the skill
                if max_sim > 0.75:  # Threshold for semantic similarity
                    found_skills.append(skill_list[max_sim_idx])
                # Also add exact matches
                elif candidate in skill_list:
                    found_skills.append(candidate)
            
            if found_skills:
                identified_skills[category] = list(set(found_skills))
    else:
        # Fallback to basic matching
        for category, skill_list in COMMON_SKILLS.items():
            found_skills = []
            for skill in skill_list:
                for candidate in skill_candidates:
                    if (skill == candidate or 
                        skill in candidate.split() or 
                        candidate in skill.split()):
                        found_skills.append(skill)
                        break
            
            if found_skills:
                identified_skills[category] = list(set(found_skills))
    
    # Look for skill context using dependency parsing
    if len(identified_skills) < 3:  # If we don't have enough skills yet
        # Look for sentences mentioning experience, knowledge, skills, etc.
        skill_context_terms = ["experience", "knowledge", "skill", "proficient", "expertise"]
        for sent in doc.sents:
            if any(term in sent.text.lower() for term in skill_context_terms):
                # Process this sentence more carefully
                sent_doc = nlp(sent.text)
                for token in sent_doc:
                    # Focus on technical-looking terms
                    if (token.pos_ in ["NOUN", "PROPN"] and 
                        not token.is_stop and 
                        len(token.text) > 2):
                        # Check if it's in any skill list
                        for category, skill_list in COMMON_SKILLS.items():
                            # Use token.lemma_ to match base forms
                            if any(skill in token.lemma_.lower() for skill in skill_list):
                                if category not in identified_skills:
                                    identified_skills[category] = []
                                for skill in skill_list:
                                    if skill in token.lemma_.lower():
                                        identified_skills[category].append(skill)
    
    return identified_skills
# Enhanced answer evaluation function 
def evaluate_answer_with_nlp(question, answer, expected_keywords):
    """
    Evaluate answer using advanced NLP techniques including:
    - Semantic similarity with expected keywords
    - Topic modeling and key concept extraction
    - Sentiment and confidence analysis
    """
    if not answer.strip():
        return {
            "score": 0,
            "feedback": "No answer provided.",
            "missing_concepts": expected_keywords
        }
    
    # Basic keyword matching (fallback if no NLP)
    if not NLP_ENABLED:
        keyword_count = sum(1 for keyword in expected_keywords if keyword.lower() in answer.lower())
        basic_score = min(keyword_count / max(len(expected_keywords), 1), 1.0) * 100
        
        missing = [k for k in expected_keywords if k.lower() not in answer.lower()]
        return {
            "score": basic_score,
            "feedback": "Based on keyword matching, your answer covers some key concepts.",
            "missing_concepts": missing
        }
    
    try:
        # Process text with spaCy
        answer_doc = nlp(answer)
        
        # Enhanced semantic similarity using sentence transformers
        if ADVANCED_NLP_ENABLED and 'sentence_model' in globals():
            # Create sentences for semantic analysis
            expected_text = " ".join(expected_keywords)
            
            # Get embeddings
            answer_embedding = sentence_model.encode(answer)
            expected_embedding = sentence_model.encode(expected_text)
            
            # Calculate semantic similarity
            semantic_similarity = cosine_similarity([answer_embedding], [expected_embedding])[0][0]
            semantic_score = semantic_similarity * 50  # 0-50 points for semantic similarity
        else:
            # Fallback to TF-IDF similarity
            documents = [" ".join(expected_keywords), answer]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(documents)
            semantic_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            semantic_score = semantic_similarity * 50
            
        # Extract key phrases and concepts from answer
        key_phrases = []
        for chunk in answer_doc.noun_chunks:
            if not all(token.is_stop for token in chunk):
                key_phrases.append(chunk.text.lower())
        
        # Extract entities
        entities = [ent.text.lower() for ent in answer_doc.ents]
        
        # Analyze technical depth
        # - Count non-stopwords
        # - Count technical terms
        tech_terms_count = 0
        technical_term_candidates = key_phrases + entities
        for term in technical_term_candidates:
            for category in COMMON_SKILLS.values():
                if any(skill in term for skill in category):
                    tech_terms_count += 1
                    break
        
        # Technical depth score based on length and keyword density
        words = [token.text for token in answer_doc if not token.is_stop]
        optimal_length = 150  # Optimal answer length (approx)
        
        # Penalize both too short and too long answers
        length_factor = min(len(words) / optimal_length, 2.0)
        if length_factor > 1:
            length_factor = 2 - length_factor  # Penalize too long answers (inverted U curve)
            
        # Calculate technical depth score
        tech_density = tech_terms_count / max(1, len(words)) * 10  # Normalize technical term density
        technical_score = (length_factor * 0.7 + tech_density * 0.3) * 25  # 0-25 points
        
        # Sentiment and confidence analysis
        if ADVANCED_NLP_ENABLED and 'sentiment_analyzer' in globals():
            # Use transformer-based sentiment analysis
            result = sentiment_analyzer(answer[:512])[0]  # Limit text length
            
            # Map sentiment to confidence
            sentiment_label = result["label"]
            sentiment_score = result["score"]
            
            if sentiment_label == "POSITIVE":
                confidence_score = sentiment_score * 25  # More positive = more confident
            else:
                confidence_score = (1 - sentiment_score) * 25  # Less negative = more confident
        else:
            # Simple confidence markers analysis
            confidence_markers = ["confident", "sure", "know", "certain", "definitely"]
            uncertainty_markers = ["maybe", "perhaps", "guess", "not sure", "might", "could be"]
            
            answer_lower = answer.lower()
            confidence_count = sum(1 for marker in confidence_markers if marker in answer_lower)
            uncertainty_count = sum(1 for marker in uncertainty_markers if marker in answer_lower)
            
            if confidence_count > uncertainty_count:
                confidence_score = 20
            elif uncertainty_count > confidence_count:
                confidence_score = 10
            else:
                confidence_score = 15
        
        # Combined score
        total_score = semantic_score + technical_score + confidence_score
        score = min(round(total_score), 100)
        
        # Identify missing concepts with enhanced semantic analysis
        missing_concepts = []
        
        # Extract key concepts from answer using NLP
        answer_concepts = set()
        for token in answer_doc:
            if not token.is_stop and token.pos_ in ["NOUN", "VERB", "ADJ"]:
                answer_concepts.add(token.lemma_.lower())
        
        for chunk in answer_doc.noun_chunks:
            if not all(token.is_stop for token in chunk):
                answer_concepts.add(chunk.text.lower())
        
        # Check for each expected keyword
        for keyword in expected_keywords:
            keyword_found = False
            keyword_lower = keyword.lower()
            
            # Direct string match
            if keyword_lower in answer.lower():
                keyword_found = True
                continue
            
            # Check for semantic similarity using spaCy
            keyword_doc = nlp(keyword_lower)
            best_similarity = 0
            
            for concept in answer_concepts:
                concept_doc = nlp(concept)
                similarity = keyword_doc.similarity(concept_doc)
                best_similarity = max(similarity, best_similarity)
                
                if similarity > 0.75:  # Threshold for semantic similarity
                    keyword_found = True
                    break
            
            # If still not found but there's some similarity, consider it "partially covered"
            if not keyword_found and best_similarity > 0.5:
                keyword_found = True  # Partial credit for related concepts
            
            if not keyword_found:
                missing_concepts.append(keyword)
        
        # Generate intelligent feedback
        if score >= 80:
            feedback = "Excellent answer that demonstrates strong understanding of the concepts. "
            if semantic_score > 40:
                feedback += "You covered all the key points effectively. "
            if technical_score > 20:
                feedback += "Your explanation has good technical depth. "
            if confidence_score > 20:
                feedback += "You presented your answer with clarity and confidence. "
        elif score >= 60:
            feedback = "Good answer that covers many of the important concepts. "
            if semantic_score < 30:
                feedback += "Try to address more of the key points directly. "
            if technical_score < 15:
                feedback += "Consider adding more technical details to strengthen your answer. "
            if confidence_score < 15:
                feedback += "Try to present your answer with more certainty. "
        else:
            feedback = "Your answer needs improvement. "
            if semantic_score < 25:
                feedback += "It doesn't adequately address the key concepts of the question. "
            if technical_score < 12:
                feedback += "Try to include more technical details and specifics. "
            if confidence_score < 10:
                feedback += "Work on presenting your answer more confidently. "
        
        if missing_concepts:
            feedback += "Consider including information about " + ", ".join(missing_concepts[:3]) + "."
        
        return {
            "score": score,
            "feedback": feedback,
            "missing_concepts": missing_concepts
        }
    except Exception as e:
        st.error(f"Error in NLP evaluation: {str(e)}")
        # Fall back to basic evaluation
        keyword_count = sum(1 for keyword in expected_keywords if keyword.lower() in answer.lower())
        basic_score = min(keyword_count / max(len(expected_keywords), 1), 1.0) * 100
        missing = [k for k in expected_keywords if k.lower() not in answer.lower()]
        return {
            "score": basic_score,
            "feedback": "Based on basic analysis, your answer covers some key concepts.",
            "missing_concepts": missing
        }
        
# Enhanced validate_answer with new NLP capabilities or fallback to Gemini
def validate_answer_with_gemini(question, answer, expected_keywords):
    """
    Use Google's Gemini API to validate a candidate's answer, with enhanced NLP fallback.
    """
    # When using NLP, we'll use our own evaluation first and only use Gemini as a backup
    if NLP_ENABLED:
        try:
            # Try our NLP evaluation first
            nlp_eval = evaluate_answer_with_nlp(question, answer, expected_keywords)
            if nlp_eval["score"] > 0:  # If valid evaluation
                return nlp_eval
        except Exception as e:
            st.warning(f"NLP evaluation error, falling back to Gemini: {str(e)}")
    
    # Use Gemini if available
    api_key = os.environ.get("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))
    
    if not api_key:
        st.warning("Gemini API key not found. Using NLP-based evaluation instead.")
        return evaluate_answer_with_nlp(question, answer, expected_keywords)
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    prompt = f"""
    Question: {question}
    
    Candidate's Answer: {answer}
    
    Expected keywords or concepts: {', '.join(expected_keywords)}
    
    Evaluate this answer based on the following criteria:
    1. Presence of expected keywords/concepts
    2. Technical accuracy
    3. Clarity of explanation
    
    Provide:
    1. A score out of 100
    2. Brief feedback (2-3 sentences)
    3. List of any missing important concepts
    
    Format as JSON with keys: "score", "feedback", "missing_concepts"
    """
    
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }
    
    data = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        response_data = response.json()
        generated_text = response_data["candidates"][0]["content"]["parts"][0]["text"]
        
        json_str = ""
        json_started = False
        
        for line in generated_text.split("\n"):
            if line.strip() == "```json":
                json_started = True
                continue
            elif line.strip() == "```" and json_started:
                break
            elif json_started:
                json_str += line + "\n"
        
        if not json_str:
            json_str = generated_text
        
        json_str = json_str.replace("```json", "").replace("```", "").strip()
        
        try:
            result = json.loads(json_str)
            if "score" not in result:
                result["score"] = 50
            if "feedback" not in result:
                result["feedback"] = "No specific feedback provided."
            if "missing_concepts" not in result:
                result["missing_concepts"] = []
            return result
        except json.JSONDecodeError:
            return evaluate_answer_with_nlp(question, answer, expected_keywords)
    except Exception as e:
        st.error(f"Error calling Gemini API: {str(e)}")
        return evaluate_answer_with_nlp(question, answer, expected_keywords)

# File processing functions
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

def extract_skills(text):
    if not text:
        return {}
    
    text = text.lower()
    identified_skills = {}
    
    for category, skill_list in COMMON_SKILLS.items():
        found_skills = []
        for skill in skill_list:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text):
                found_skills.append(skill)
        if found_skills:
            identified_skills[category] = found_skills
    
    return identified_skills


# Completing the generate_technical_questions function
def generate_technical_questions(skills, max_questions=7):
    """Generate technical questions based on extracted skills using NLP"""
    all_possible_questions = []
    
    # Flatten and count skills by frequency
    all_skills = []
    for category, skill_list in skills.items():
        all_skills.extend(skill_list)
    
    skill_frequency = {}
    for skill in all_skills:
        skill_frequency[skill] = skill_frequency.get(skill, 0) + 1
    
    # Sort skills by frequency (most mentioned first)
    sorted_skills = sorted(skill_frequency.keys(), key=lambda x: skill_frequency[x], reverse=True)
    
    if NLP_ENABLED:
        # Use NLP to find semantic matches for skills without direct question matches
        skills_without_questions = []
        
        # First pass - direct matches
        for skill in sorted_skills:
            if skill in TECHNICAL_QUESTIONS:
                # Get questions for direct skill match
                skill_questions = TECHNICAL_QUESTIONS[skill].copy()
                
                # Prioritize questions with higher keyword overlap
                if len(skills) > 1:
                    # Calculate how many other skills are mentioned in each question
                    for q in skill_questions:
                        q_text = q["question"].lower()
                        other_skill_count = sum(1 for s in all_skills if s != skill and s in q_text)
                        q["relevance_score"] = other_skill_count + 1
                    
                    # Sort by relevance
                    skill_questions.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                
                # Add questions with original order preserved
                all_possible_questions.extend(skill_questions)
            else:
                skills_without_questions.append(skill)
        
        # Second pass - semantic matching for skills without direct questions
        for skill in skills_without_questions:
            best_match = None
            best_score = 0
            
            # Create a spaCy doc for this skill
            skill_doc = nlp(skill)
            
            # Find most similar technical question category
            for tech_skill in TECHNICAL_QUESTIONS.keys():
                tech_skill_doc = nlp(tech_skill)
                similarity = skill_doc.similarity(tech_skill_doc)
                
                if similarity > best_score and similarity > 0.6:  # Threshold
                    best_score = similarity
                    best_match = tech_skill
            
            # If we found a good match, add those questions
            if best_match:
                all_possible_questions.extend(TECHNICAL_QUESTIONS[best_match])
    else:
        # Fallback to direct matching and partial matching
        for skill in sorted_skills:
            # First check for direct skill match
            if skill in TECHNICAL_QUESTIONS:
                all_possible_questions.extend(TECHNICAL_QUESTIONS[skill])
            else:
                # Check for partial matches
                for tech_skill in TECHNICAL_QUESTIONS.keys():
                    if tech_skill in skill or skill in tech_skill:
                        all_possible_questions.extend(TECHNICAL_QUESTIONS[tech_skill])
                        break
    
    # Add generic questions
    all_possible_questions.extend(GENERIC_QUESTIONS)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_questions = []
    for q in all_possible_questions:
        q_text = q["question"]
        if q_text not in seen:
            seen.add(q_text)
            unique_questions.append(q)
    
    # Shuffle and limit number of questions
    random.shuffle(unique_questions)
    return unique_questions[:max_questions]

def generate_feedback_report(interview_results):
    """Generate a comprehensive feedback report based on interview results"""
    report = FPDF()
    report.add_page()
    report.set_font("Arial", "B", 16)
    report.cell(190, 10, "Technical Interview Feedback Report", 0, 1, "C")
    report.line(10, 30, 200, 30)
    
    report.set_font("Arial", "", 12)
    report.cell(190, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
    
    # Overall score
    total_score = sum(result['score'] for result in interview_results)
    avg_score = total_score / len(interview_results) if interview_results else 0
    
    report.set_font("Arial", "B", 14)
    report.cell(190, 10, f"Overall Score: {avg_score:.1f}/100", 0, 1)
    
    # Strengths and weaknesses
    strengths = []
    areas_for_improvement = []
    
    for result in interview_results:
        if result['score'] >= 80:
            strengths.append(result['question'].split("?")[0] + "?")
        elif result['score'] <= 50:
            areas_for_improvement.append(result['question'].split("?")[0] + "?")
    
    # Add strengths section
    report.set_font("Arial", "B", 14)
    report.cell(190, 10, "Strengths:", 0, 1)
    report.set_font("Arial", "", 12)
    
    if strengths:
        for strength in strengths[:3]:  # Limit to top 3
            report.cell(190, 10, f"• {strength}", 0, 1)
    else:
        report.cell(190, 10, "• No clear strengths identified.", 0, 1)
    
    # Add areas for improvement
    report.set_font("Arial", "B", 14)
    report.cell(190, 10, "Areas for Improvement:", 0, 1)
    report.set_font("Arial", "", 12)
    
    if areas_for_improvement:
        for area in areas_for_improvement[:3]:  # Limit to top 3
            report.cell(190, 10, f"• {area}", 0, 1)
    else:
        report.cell(190, 10, "• No clear weaknesses identified.", 0, 1)
    
    # Detailed results
    report.add_page()
    report.set_font("Arial", "B", 14)
    report.cell(190, 10, "Detailed Question Analysis:", 0, 1)
    
    for i, result in enumerate(interview_results):
        report.set_font("Arial", "B", 12)
        report.cell(190, 10, f"Question {i+1}: {result['question']}", 0, 1)
        
        report.set_font("Arial", "", 12)
        report.multi_cell(190, 10, f"Your Answer: {result['answer'][:100]}..." if len(result['answer']) > 100 else f"Your Answer: {result['answer']}")
        
        report.set_font("Arial", "B", 12)
        report.cell(190, 10, f"Score: {result['score']}/100", 0, 1)
        
        report.set_font("Arial", "", 12)
        report.multi_cell(190, 10, f"Feedback: {result['feedback']}")
        
        if result['missing_concepts']:
            report.multi_cell(190, 10, f"Missing Concepts: {', '.join(result['missing_concepts'])}")
        
        report.cell(190, 5, "", 0, 1)  # Add some spacing
    
    # Final recommendations
    report.add_page()
    report.set_font("Arial", "B", 14)
    report.cell(190, 10, "Recommendations:", 0, 1)
    
    report.set_font("Arial", "", 12)
    report.multi_cell(190, 10, "Based on your performance, here are some recommendations to improve your technical interview skills:")
    
    recommendations = [
        "Practice explaining technical concepts clearly and concisely.",
        "Work on providing examples to illustrate your points.",
        "Review the core concepts of your strongest technical skills.",
        "Practice more complex technical scenarios."
    ]
    
    for rec in recommendations:
        report.cell(190, 10, f"• {rec}", 0, 1)
    
    # Return the PDF as bytes
    return report.output(dest="S").encode("latin1")

def send_feedback_email(email, report_pdf, interview_results):
    """Send interview feedback report to candidate via email"""
    sender_email = os.environ.get("EMAIL_SENDER", st.secrets.get("EMAIL_SENDER", ""))
    password = os.environ.get("EMAIL_PASSWORD", st.secrets.get("EMAIL_PASSWORD", ""))
    
    if not sender_email or not password:
        st.error("Email credentials not configured. Unable to send report.")
        return False
    
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = email
        msg['Subject'] = "Your Technical Interview Feedback Report"
        
        # Calculate overall score
        total_score = sum(result['score'] for result in interview_results)
        avg_score = total_score / len(interview_results) if interview_results else 0
        
        body = f"""
        Dear Candidate,
        
        Thank you for completing your practice technical interview with TechInterviewBot.
        
        Your overall score was {avg_score:.1f}/100.
        
        Please find attached your comprehensive feedback report.
        
        Best regards,
        TechInterviewBot
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PDF report
        attachment = MIMEApplication(report_pdf)
        attachment.add_header('Content-Disposition', 'attachment', filename="TechnicalInterviewReport.pdf")
        msg.attach(attachment)
        
        # Connect to server and send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        
        return True
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False

# Main Streamlit app
def main():
    st.set_page_config(page_title="Technical Interview Practice Bot", page_icon="🤖", layout="wide")
    
    # Custom CSS for better appearance
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7ff;
    }
    .stButton button {
        background-color: #4C6EF5;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .question-box {
        background-color: #e6f2ff;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .answer-box {
        background-color: #f0f0f0;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .feedback-positive {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 10px;
        margin: 10px 0;
    }
    .feedback-average {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 10px;
        margin: 10px 0;
    }
    .feedback-negative {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App title and description
    st.title("🤖 Technical Interview Practice Bot")
    st.markdown("""
    Prepare for your next technical interview with AI-powered practice sessions.
    Upload your resume to get personalized technical questions based on your skills and experience.
    """)
    
    # Initialize session state variables
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = ""
    if 'skills' not in st.session_state:
        st.session_state.skills = {}
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'current_question_index' not in st.session_state:
        st.session_state.current_question_index = 0
    if 'interview_results' not in st.session_state:
        st.session_state.interview_results = []
    if 'interview_started' not in st.session_state:
        st.session_state.interview_started = False
    if 'interview_completed' not in st.session_state:
        st.session_state.interview_completed = False
    if 'timer_started' not in st.session_state:
        st.session_state.timer_started = False
    if 'timer_start_time' not in st.session_state:
        st.session_state.timer_start_time = None
    
    # Display welcome message
    if not st.session_state.interview_started and not st.session_state.interview_completed:
        st.markdown(f"### {random.choice(WELCOME_MESSAGES)}")
        st.markdown(f"{random.choice(RESUME_PROMPTS)}")
        
        # Create two columns for resume upload and paste options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Resume")
            uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
            if uploaded_file is not None:
                with st.spinner("Processing resume..."):
                    # Process based on file type
                    if uploaded_file.name.endswith(".pdf"):
                        st.session_state.resume_text = extract_text_from_pdf(uploaded_file)
                    elif uploaded_file.name.endswith(".docx"):
                        st.session_state.resume_text = extract_text_from_docx(uploaded_file)
                    else:  # .txt files
                        st.session_state.resume_text = uploaded_file.getvalue().decode("utf-8")
                    
                    if NLP_ENABLED:
                        st.session_state.skills = extract_skills_with_nlp(st.session_state.resume_text)
                    else:
                        st.session_state.skills = extract_skills(st.session_state.resume_text)
        
        with col2:
            st.subheader("Paste Resume Text")
            pasted_text = st.text_area("Paste your resume text here", height=300)
            if st.button("Process Resume Text"):
                with st.spinner("Processing resume..."):
                    st.session_state.resume_text = pasted_text
                    if NLP_ENABLED:
                        st.session_state.skills = extract_skills_with_nlp(st.session_state.resume_text)
                    else:
                        st.session_state.skills = extract_skills(st.session_state.resume_text)
        
        # Display extracted skills if available
        if st.session_state.skills:
            st.markdown(f"### {random.choice(SKILL_MESSAGES)}")
            
            for category, skill_list in st.session_state.skills.items():
                st.write(f"**{category.replace('_', ' ').title()}:** {', '.join(skill_list)}")
            
            # Option to manually add skills
            st.markdown("### Add additional skills or modify extracted ones")
            
            for category in COMMON_SKILLS.keys():
                category_display = category.replace('_', ' ').title()
                
                if category in st.session_state.skills:
                    default_skills = ", ".join(st.session_state.skills[category])
                else:
                    default_skills = ""
                
                modified_skills = st.text_input(f"{category_display} Skills", value=default_skills)
                
                if modified_skills:
                    skill_list = [skill.strip().lower() for skill in modified_skills.split(",") if skill.strip()]
                    if skill_list:
                        st.session_state.skills[category] = skill_list
                    elif category in st.session_state.skills:
                        del st.session_state.skills[category]
            
            # Start interview button
            if st.button("Start Interview"):
                with st.spinner("Generating questions..."):
                    st.session_state.questions = generate_technical_questions(st.session_state.skills)
                    st.session_state.interview_started = True
                    st.session_state.current_question_index = 0
                    st.session_state.interview_results = []
                    st.experimental_rerun()
    
    # Interview in progress
    elif st.session_state.interview_started and not st.session_state.interview_completed:
        # Show progress
        progress = st.progress((st.session_state.current_question_index) / len(st.session_state.questions))
        st.markdown(f"### Question {st.session_state.current_question_index + 1} of {len(st.session_state.questions)}")
        
        # Timer display
        if not st.session_state.timer_started:
            st.session_state.timer_started = True
            st.session_state.timer_start_time = datetime.now()
        
        elapsed_time = datetime.now() - st.session_state.timer_start_time
        st.info(f"Time spent on this question: {elapsed_time.seconds // 60}m {elapsed_time.seconds % 60}s")
        
        # Display current question
        current_q = st.session_state.questions[st.session_state.current_question_index]
        st.markdown(f"<div class='question-box'><h3>{current_q['question']}</h3></div>", unsafe_allow_html=True)
        
        # Input for answer
        answer = st.text_area("Your answer:", height=200)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit Answer"):
                if not answer.strip():
                    st.warning("Please provide an answer before submitting.")
                else:
                    with st.spinner("Evaluating your answer..."):
                        # Evaluate answer
                        evaluation = validate_answer_with_gemini(
                            current_q['question'], 
                            answer, 
                            current_q['expected_keywords']
                        )
                        
                        # Save result
                        st.session_state.interview_results.append({
                            'question': current_q['question'],
                            'answer': answer,
                            'score': evaluation['score'],
                            'feedback': evaluation['feedback'],
                            'missing_concepts': evaluation['missing_concepts']
                        })
                        
                        # Display feedback
                        if evaluation['score'] >= 80:
                            st.markdown(f"<div class='feedback-positive'><strong>Score: {evaluation['score']}/100</strong><br>{evaluation['feedback']}</div>", unsafe_allow_html=True)
                        elif evaluation['score'] >= 60:
                            st.markdown(f"<div class='feedback-average'><strong>Score: {evaluation['score']}/100</strong><br>{evaluation['feedback']}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='feedback-negative'><strong>Score: {evaluation['score']}/100</strong><br>{evaluation['feedback']}</div>", unsafe_allow_html=True)
                        
                        # Move to next question or complete
                        if st.session_state.current_question_index < len(st.session_state.questions) - 1:
                            st.session_state.current_question_index += 1
                            st.session_state.timer_started = False
                            st.experimental_rerun()
                        else:
                            st.session_state.interview_completed = True
                            st.experimental_rerun()
        
        with col2:
            if st.button("Skip Question"):
                # Record as skipped with zero score
                st.session_state.interview_results.append({
                    'question': current_q['question'],
                    'answer': "[SKIPPED]",
                    'score': 0,
                    'feedback': "Question was skipped.",
                    'missing_concepts': current_q['expected_keywords']
                })
                
                # Move to next question or complete
                if st.session_state.current_question_index < len(st.session_state.questions) - 1:
                    st.session_state.current_question_index += 1
                    st.session_state.timer_started = False
                    st.experimental_rerun()
                else:
                    st.session_state.interview_completed = True
                    st.experimental_rerun()
    
    # Interview completed - show results
    elif st.session_state.interview_completed:
        st.markdown("## 🎉 Interview Completed!")
        
        # Calculate overall score
        total_score = sum(result['score'] for result in st.session_state.interview_results)
        avg_score = total_score / len(st.session_state.interview_results) if st.session_state.interview_results else 0
        
        # Display score with gauge
        st.markdown(f"### Overall Score: {avg_score:.1f}/100")
        
        # Show summary of strongest and weakest areas
        strengths = []
        areas_for_improvement = []
        
        for result in st.session_state.interview_results:
            if result['score'] >= 80:
                strengths.append(result['question'].split("?")[0] + "?")
            elif result['score'] <= 50:
                areas_for_improvement.append(result['question'].split("?")[0] + "?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Strengths")
            if strengths:
                for strength in strengths:
                    st.success(strength)
            else:
                st.info("No clear strengths identified.")
        
        with col2:
            st.markdown("### Areas for Improvement")
            if areas_for_improvement:
                for area in areas_for_improvement:
                    st.warning(area)
            else:
                st.info("No clear weaknesses identified.")
        
        # Detailed results expandable section
        with st.expander("View Detailed Results"):
            for i, result in enumerate(st.session_state.interview_results):
                st.markdown(f"#### Question {i+1}: {result['question']}")
                st.markdown(f"**Your Answer:**\n{result['answer']}")
                st.markdown(f"**Score:** {result['score']}/100")
                st.markdown(f"**Feedback:** {result['feedback']}")
                if result['missing_concepts']:
                    st.markdown(f"**Missing Concepts:** {', '.join(result['missing_concepts'])}")
                st.markdown("---")
        
        # Generate PDF report
        with st.spinner("Generating comprehensive report..."):
            report_pdf = generate_feedback_report(st.session_state.interview_results)
        
        # Download report
        st.download_button(
            label="Download Feedback Report",
            data=report_pdf,
            file_name="TechnicalInterviewReport.pdf",
            mime="application/pdf"
        )
        
        # Email report option
        st.markdown("### Send Report to Email")
        email = st.text_input("Enter your email address")
        if st.button("Send Report") and email:
            if re.match(r"[^@]+@[^@]+\.[^@]+", email):
                with st.spinner("Sending report via email..."):
                    if send_feedback_email(email, report_pdf, st.session_state.interview_results):
                        st.success("Report sent successfully to your email!")
                    else:
                        st.error("Failed to send email. Please download the report instead.")
            else:
                st.error("Please enter a valid email address.")
        
        # Start a new interview
        if st.button("Start a New Interview"):
            st.session_state.interview_started = False
            st.session_state.interview_completed = False
            st.session_state.resume_text = ""
            st.session_state.skills = {}
            st.session_state.questions = []
            st.session_state.current_question_index = 0
            st.session_state.interview_results = []
            st.session_state.timer_started = False
            st.experimental_rerun()

if __name__ == "__main__":
    main()
