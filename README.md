🧠 Mental Health Prediction App
Predict the potential risk of mental illness based on lifestyle and historical factors
This application uses a machine learning model to assess the potential risk of mental health issues. By inputting key information about your lifestyle and mental health history, the app provides a personalized prediction along with a probability score, indicating the likelihood of mental illness risk.

🔮 Features
Interactive UI: Users can input personal details and receive predictions in real-time.
Multiple Risk Levels: The app provides risk predictions in four levels (No Risk, Low Risk, Moderate Risk, High Risk).
Probability Visualization: The predicted probability is displayed visually through a bar chart.
Customized Feedback: Feedback messages are tailored based on the risk level, offering suggestions to users.
🚀 How to Run the App
To run this Streamlit app locally, follow these steps:

Prerequisites
Make sure you have the following installed:

Python 3.x
Streamlit
TensorFlow (for running the pre-trained model)
Installation
Clone the repository:

bash
Skopiuj kod
git clone https://github.com/your-username/mental-health-prediction-app.git
Navigate into the project directory:

bash
Skopiuj kod
cd mental-health-prediction-app
Install the required dependencies:

bash
Skopiuj kod
pip install -r requirements.txt
Download the pre-trained model and place it in the project directory. The model file should be named NN_via_carpatia.h5.

Running the App
Once everything is set up, you can run the app locally using Streamlit:

bash
Skopiuj kod
streamlit run app.py
Demo
After launching the app, you'll see a user-friendly interface where you can input your information. Based on your inputs, the app will predict your mental health status and display the likelihood of mental illness risk.

<!-- Add a screenshot of your app -->

🔑 Input Fields
Gender: Select your gender.
Occupation: Specify your current occupation (e.g., Student, Business, etc.).
Self-employed: Are you self-employed? (Yes/No)
Family History: Do you have a family history of mental illness? (Yes/No)
Treatment: Are you currently receiving treatment for mental health? (Yes/No)
Days Indoors: How long have you been staying indoors recently?
Stress Levels: Have your stress levels been increasing?
Changes in Habits: Have you noticed changes in your habits?
Mental Health History: Do you have a history of mental health issues?
Mood Swings: How frequently do you experience mood swings?
Coping Ability: Do you struggle to cope with daily challenges?
Interest in Work: Are you interested in your work or daily activities?
Social Weakness: Do you feel socially weak or disconnected?
Mental Health Interview: Have you ever had a mental health assessment?
Access to Mental Health Care: Do you have access to mental health care options?
🎯 Prediction Logic
The app uses a neural network model trained on a dataset of mental health records to predict the likelihood of mental illness. The model evaluates user input data and outputs a probability, which is then categorized into four levels:

No Risk: Probability < 20%
Low Risk: Probability between 20% - 50%
Moderate Risk: Probability between 50% - 80%
High Risk: Probability > 80%
📊 Visualization
A probability bar chart is displayed to give users a visual representation of their risk level.

🛠️ Tech Stack
Frontend: Streamlit
Model: TensorFlow/Keras Neural Network
Languages: Python
👨‍💻 Author
Mateusz Walo
LinkedIn
Data source: Kaggle Mental Health Dataset
🤝 Contributing
If you'd like to contribute, feel free to fork the repository and submit a pull request with your improvements. Contributions are always welcome!

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
