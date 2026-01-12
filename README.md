🛡️ AI-Based Network Intrusion Detection System

VIOS Internship Project – Edunet Foundation

📌 Project Overview

This project is developed as part of the VIOS Internship Program by Edunet Foundation.
It demonstrates the application of Machine Learning in cybersecurity by detecting malicious network traffic using a supervised learning approach.

The system uses a Random Forest classifier trained on the CIC-IDS2017 dataset to classify network traffic as BENIGN or ATTACK (e.g., DDoS).
An interactive Streamlit dashboard is used to visualize predictions and simulate network packets.

An optional integration with Groq LLM provides human-readable explanations for the model’s predictions. If the Groq library or API key is unavailable, the core detection system continues to function without interruption.

🎯 Objectives

Understand network intrusion detection concepts

Apply machine learning techniques to real-world cybersecurity data

Build a simple and interactive dashboard for model visualization

Demonstrate optional use of generative AI for explainability

⚙️ Technologies Used

Python

Streamlit

Pandas, NumPy

Scikit-learn (Random Forest)

Groq LLM (optional)

📂 Project Structure
AI-NIDS-Project/
│── app.py
│── requirements.txt
│── README.md
│── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

🚀 How to Run the Project
1️⃣ Install Dependencies
python -m pip install -r requirements.txt

2️⃣ Run the Application
streamlit run app.py


The application will open in your browser at:

http://localhost:8501

🧪 How to Use

Click Train Model to train the Random Forest classifier

Capture a random network packet for simulation

View whether the traffic is classified as BENIGN or ATTACK

(Optional) Enter a Groq API key to generate an AI-based explanation

📊 Dataset

CIC-IDS2017 (Canadian Institute for Cybersecurity)

Specifically uses traffic related to DDoS attacks

📌 Internship Context

This project was developed for educational purposes during the VIOS Internship by Edunet Foundation to demonstrate practical implementation of machine learning concepts in cybersecurity.
It is not intended for real-time or production deployment.

✅ Learning Outcomes

Practical experience with ML-based classification

Exposure to cybersecurity datasets

Understanding of model explainability using LLMs

Hands-on use of Streamlit for ML applications

🔚 Conclusion

This project showcases how traditional machine learning models can be applied to cybersecurity problems and enhanced with optional AI-based explanations, making it suitable for academic and internship-level demonstrations.