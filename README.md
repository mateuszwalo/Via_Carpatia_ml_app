# 🧠 Mental Health Prediction App

### Predict the potential risk of mental illness based on lifestyle and historical factors

This application uses a machine learning model to assess the potential risk of mental health issues. By inputting key information about your lifestyle and mental health history, the app provides a personalized prediction along with a probability score, indicating the likelihood of mental illness risk.

## 🔮 Features

- **Interactive UI**: Users can input personal details and receive predictions in real-time.
- **Multiple Risk Levels**: The app provides risk predictions in four levels (No Risk, Low Risk, Moderate Risk, High Risk).
- **Probability Visualization**: The predicted probability is displayed visually through a bar chart.
- **Customized Feedback**: Feedback messages are tailored based on the risk level, offering suggestions to users.

## 🚀 How to Run the App

To run this Streamlit app locally, follow these steps:

### Prerequisites

Make sure you have the following installed:
- Python 3.x
- [Streamlit](https://streamlit.io)
- TensorFlow (for running the pre-trained model)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/mental-health-prediction-app.git
    ```
   
2. Navigate into the project directory:
    ```bash
    cd mental-health-prediction-app
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the pre-trained model and place it in the project directory. The model file should be named `NN_via_carpatia.h5`.

### Running the App

Once everything is set up, you can run the app locally using Streamlit:

```bash
streamlit run app.py
