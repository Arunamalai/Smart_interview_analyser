import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import speech_recognition as sr
import textstat

# SETUP
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# SPEECH TO TEXT CONVERSION
def get_audio_input(uploaded_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(uploaded_file) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except:
        return "Could not understand audio"

def record_from_mic():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Recording...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        return "Could not understand audio"

# GRAMMAR Checking
def get_grammar_score(text):
    blob = TextBlob(text)
    corrected = str(blob.correct())

    original_words = text.split()
    corrected_words = corrected.split()

    error_count = sum(1 for o, c in zip(original_words, corrected_words) if o != c)

    if error_count == 0:
        return 10
    elif error_count <= 2:
        return 8
    elif error_count <= 5:
        return 6
    else:
        return 4
    
def extract_keywords(question):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(question.lower())
    return [w for w in words if w.isalnum() and w not in stop_words]

# NLP answer analysing
def analyze_text(text, question):
    grammar = get_grammar_score(text)

    text_lower = text.lower()
    words = word_tokenize(text_lower)
    sentences = sent_tokenize(text)

    # Basic stats
    word_count = len(words)
    num_sentences = len(sentences)
    avg_len = word_count / num_sentences if num_sentences else 0

    # Keyword relevance
    keywords = extract_keywords(question)
    keyword_score = sum(1 for w in words if w in keywords)

    # Filler words
    fillers = ["um", "uh", "like", "you know", "basically"]
    filler_count = sum(text_lower.count(f) for f in fillers)

    # Clarity score (based on sentence length)
    if avg_len < 5:
        clarity_score = 4
    elif avg_len <= 20:
        clarity_score = 10
    else:
        clarity_score = 6

    # Readability score
    readability = textstat.flesch_reading_ease(text)

    return {
        "grammar": grammar,
        "word_count": word_count,
        "num_sentences": num_sentences,
        "avg_len": round(avg_len, 2),
        "keyword_score": keyword_score,
        "filler_count": filler_count,
        "clarity_score": clarity_score,
        "readability_score": round(readability, 2)
    }
# EMOTION DETECTION
def detect_emotion(text):
    text_lower = text.lower()
    words = text.split()

    fillers = ["um", "uh", "like", "you know", "basically"]
    filler_count = sum(text_lower.count(f) for f in fillers)

    confident_phrases = ["i am confident", "i can", "i will", "i have"]
    confidence_hits = sum(1 for p in confident_phrases if p in text_lower)

    repeated_words = sum(1 for i in range(1, len(words)) if words[i].lower() == words[i-1].lower())

    if filler_count > 2 or repeated_words > 2:
        return "nervous"
    elif confidence_hits > 0:
        return "confident"
    else:
        return "neutral"

# SCORING MODULE
def calculate_score(nlp, emotion):
    
    # 1. ANSWER QUALITY
    keyword_score = nlp["keyword_score"]
    word_count = nlp["word_count"]

    # Depth bonus (longer meaningful answers)
    if word_count > 40:
        depth_bonus = 3
    elif word_count > 25:
        depth_bonus = 2
    elif word_count > 15:
        depth_bonus = 1
    else:
        depth_bonus = 0

    quality = min(keyword_score + depth_bonus, 10)

    #2. COMMUNICATION
    clarity = nlp["clarity_score"]
    grammar = nlp["grammar"]
    fillers = nlp["filler_count"]
    readability = nlp["readability_score"]

    # Readability normalization (0 to 100 → 0 to 10)
    readability_score = max(min(readability / 10, 10), 0)

    # Penalize fillers
    filler_penalty = min(fillers * 0.5, 3)

    communication = (
        clarity * 0.4 +
        grammar * 0.3 +
        readability_score * 0.3
    ) - filler_penalty

    communication = max(min(communication, 10), 0)

    # 3. CONFIDENCE
    if emotion == "confident":
        confidence = 9
    elif emotion == "neutral":
        confidence = 6
    else:
        confidence = 3

    # 4. FINAL SCORE
    final = (
        quality * 0.5 +
        communication * 0.2 +
        confidence * 0.3
    )

    return {
        "answer_quality": round(quality, 2),
        "communication": round(communication, 2),
        "confidence": confidence,
        "final_score": round(final, 2)
    }

# FEEDBACK GENERATION
def generate_feedback(nlp, scores, emotion):
    feedback = []

    # ANSWER QUALITY
    if scores["answer_quality"] < 4:
        feedback.append("Your answer lacks technical depth. Try to include more relevant concepts and examples.")
    elif scores["answer_quality"] < 7:
        feedback.append("Your answer is moderately relevant, but adding more technical details would improve it.")
    else:
        feedback.append("Your answer demonstrates strong technical understanding.")

    # COMMUNICATION
    if scores["communication"] < 5:
        feedback.append("Your communication needs improvement. Try to form clearer and more structured sentences.")
    elif scores["communication"] < 8:
        feedback.append("Your communication is decent, but can be improved with better sentence structure.")
    else:
        feedback.append("Your communication is clear and well-structured.")

    # GRAMMAR
    if nlp["grammar"] < 6:
        feedback.append("Work on improving grammar and sentence formation for better clarity.")

    # FILLER WORDS
    if nlp["filler_count"] > 2:
        feedback.append("Avoid using filler words like 'um', 'uh' to sound more confident.")

    #ANSWER LENGTH 
    if nlp["word_count"] < 10:
        feedback.append("Your answer is too short. Try to elaborate more.")

    # CLARITY 
    if nlp["clarity_score"] < 6:
        feedback.append("Your sentences are too short or unclear. Try to explain ideas more clearly.")

    # READABILITY 
    if nlp["readability_score"] < 40:
        feedback.append("Your answer is difficult to understand. Try to simplify your language.")

    # CONFIDENCE
    if emotion == "nervous":
        feedback.append("You seem nervous. Try to speak more confidently and steadily.")
    elif emotion == "neutral":
        feedback.append("Try to express more confidence in your answer.")
    else:
        feedback.append("You showed good confidence while answering.")

    # FINAL SUMMARY 
    if scores["final_score"] >= 8:
        feedback.append("Overall, this is a strong answer. Keep it up!")
    elif scores["final_score"] >= 5:
        feedback.append("Overall, your answer is average. Focus on improving key areas.")
    else:
        feedback.append("Overall, your answer needs significant improvement. Practice more structured responses.")

    return feedback


#DB Connection

import mysql.connector

def get_connection():
    return mysql.connector.connect(
        host="gateway01.us-west-2.prod.aws.tidbcloud.com",
        port=4000,
        user="2EDdC8GqjrzqdnT.root",
        password="UDcr0B3tS2qEN5u5",
        database="Guvi"
    )


# SAVE RESULT to DB
def save_result(question, answer, scores):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        query = """
        INSERT INTO interview_results 
        (question, answer, answer_quality, communication, confidence, final_score, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, NOW())
        """

        values = (
            question,
            answer,
            scores["answer_quality"],
            scores["communication"],
            scores["confidence"],
            scores["final_score"]
        )

        cursor.execute(query, values)
        conn.commit()

        cursor.close()
        conn.close()

    except Exception as e:
        st.error(f"Database Error: {e}")

#  UI
st.set_page_config(page_title="Smart Interview Analyzer", layout="wide")
st.title("Smart Interview Analyzer 🚀")

interview_type = st.selectbox("Interview Type", ["Technical", "General"])

domain = st.selectbox("Domain", ["AI/ML", "Data Analyst", "Python", "General"])

question = {
    "AI/ML": "Explain supervised learning",
    "Data Analyst": "What is data cleaning",
    "Python": "Explain list vs tuple",
    "General": "Why should we hire you"
}[domain]

st.subheader("Question")
st.info(question)

input_mode = st.radio("Input Mode", ["Text", "Audio Upload", "Live Mic"])

user_text = ""

if input_mode == "Text":
    user_text = st.text_area("Enter your answer")

elif input_mode == "Audio Upload":
    audio = st.file_uploader("Upload .wav", type=["wav"])
    if audio:
        user_text = get_audio_input(audio)
        st.success("Transcribed Text:")
        st.write(user_text)

elif input_mode == "Live Mic":
    if st.button("Start Recording"):
        user_text = record_from_mic()
        st.success("Recorded Text:")
        st.write(user_text)

# ANALYZE 
if st.button("Analyze"):
    if user_text:
        nlp = analyze_text(user_text, question)
        emotion = detect_emotion(user_text)
        scores = calculate_score(nlp, emotion)

        # 👉 Generate Feedback
        feedback = generate_feedback(nlp, scores, emotion)

        # 👉 Save to DB
        save_result(question, user_text, scores)

        #  SCORE DISPLAY
        st.subheader("📊 Scores")

        col1, col2, col3 = st.columns(3)
        col1.metric("Answer Quality", scores["answer_quality"])
        col2.metric("Communication", scores["communication"])
        col3.metric("Confidence", scores["confidence"])

        st.progress(scores["final_score"] / 10)
        st.success(f"Final Score: {scores['final_score']} / 10")

        #  EMOTION 
        st.subheader("🧠 Detected Emotion")
        st.info(emotion.capitalize())

        # FEEDBACK 
        st.subheader("💡 Feedback & Suggestions")

        for f in feedback:
            st.write("•", f)

    else:
        st.warning("Please enter answer")
