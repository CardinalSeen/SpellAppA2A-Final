# SpellBee Champions: Speech-to-Speech Web-Based Spelling App (Grades 1–6)

A web-based interactive spelling application designed for elementary students (Grades 1–6) leveraging an **Agent-to-Agent (A2A)** orchestration approach and real-time speech processing. Developed as an interactive AI tutoring tool to address foundational spelling literacy gaps.

---

## 🚀 Overview & Background

Mastering foundational spelling skills is critical for elementary student literacy. Stemming from diagnostic assessments identifying spelling gaps in foundational rules, this application acts as a personal AI tutor. It provides adaptive word difficulty and game-like, interactive speech-to-speech quiz experiences to keep students engaged and challenged.

---

## 🏗️ System Architecture

The application is structured across client-side speech processing, serverless cloud backend logic, and generative AI models:

* **Client-Side (Frontend):** Built with HTML, CSS, JavaScript, and the **Web Speech API** for real-time speech recognition (listening to user spelling) and speech synthesis (reading words aloud).
* **Backend & Cloud Services (Google Cloud Platform):** Utilizes **Firebase Authentication** for secure login, **Firestore** for permanent record storage, and **GCP Cloud Functions** powered by **LangGraph** for backend agent orchestration.
* **Artificial Intelligence Services:** Powered by **Google Gemini 2.0 Flash** for dynamic age-appropriate word generation and automated student performance report analysis.

---

## ⚙️ Core Process Flow

1. **Authentication & Setup:** Users enter their name and select an elementary grade level (Grades 1–6), triggering anonymous/secure background authentication.
2. **Interactive Quiz Loop:** 
   * The system speaks a target word aloud.
   * The user spells the word aloud within a 15-second window.
   * Voice input is instantly converted to text, compared against the correct spelling, and given immediate audio-visual feedback.
3. **AI Evaluation & Reporting:** Upon completion, scores and incorrect responses are sent to a secure Cloud Function. Gemini AI generates a personalized improvement report, saving the complete summary to the database and displaying results on screen.

---

## 🏆 Hackathon Recognition & Presentation

* Developed and presented the AI solution at the **PIE & AI Manila – AI Hive’s leMNL Vibe Hack 2.0 Hackathon**, securing **4th place among 10 competing teams**.

---

## 🛠️ Tech Stack

* **Frontend:** HTML5, CSS3, JavaScript, Web Speech API
* **Backend & Cloud:** Google Cloud Platform (GCP), Cloud Functions, Firebase Authentication, Firestore
* **Orchestration & AI:** LangGraph, Google Gemini 2.0 Flash

---

## 🔮 Future Roadmap & Vision

This application serves as a foundational proof-of-concept. Future development plans include:
* Integrating **Retrieval-Augmented Generation (RAG)** with **Pinecone** vector databases and Firebase for advanced context retrieval.
* Expanding into a comprehensive pedagogical tool featuring video analysis of learners, in-school pilot testing, and a dedicated teacher dashboard.
* Full-scale development geared towards advanced academic research in a Master of Science in Computer Science program.

---

## 👤 Author

* **Marc Alvin Sandrino**
