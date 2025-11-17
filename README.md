# Heart Disease Detection System

A real-time web application for detecting heart disease using machine learning. This project uses Python (Flask) for the backend, JavaScript for real-time interactions, and HTML/CSS for a modern user interface.

## Features

- 🎯 Real-time heart disease prediction
- 🎨 Modern and responsive UI design
- 🤖 Machine Learning model (Random Forest Classifier)
- 📊 Risk probability visualization
- ✅ Form validation and user feedback

## Project Structure

```
.
├── app.py                 # Flask backend server
├── model.py              # ML model training and prediction
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html       # Frontend HTML
├── static/
│   ├── style.css        # Styling
│   └── script.js        # JavaScript for real-time interactions
└── README.md            # This file
```

## Installation

1. **Clone or navigate to the project directory**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:5000`

3. **Enter patient information**
   - Fill in all the required fields with patient data
   - Click "Predict Heart Disease" button
   - View the real-time prediction result

## Input Parameters

The model requires the following 13 parameters:

1. **Age** - Patient's age in years
2. **Sex** - Gender (0 = Female, 1 = Male)
3. **Chest Pain Type** - Type of chest pain (0-3)
4. **Resting Blood Pressure** - In mm Hg
5. **Serum Cholesterol** - In mg/dl
6. **Fasting Blood Sugar** - > 120 mg/dl (0 = No, 1 = Yes)
7. **Resting ECG** - Resting electrocardiographic results (0-2)
8. **Maximum Heart Rate** - Maximum heart rate achieved
9. **Exercise Induced Angina** - (0 = No, 1 = Yes)
10. **ST Depression** - Old peak value
11. **Slope** - Slope of peak exercise ST segment (0-2)
12. **Number of Major Vessels** - Colored by fluoroscopy (0-3)
13. **Thalassemia** - (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)

## Model Information

- **Algorithm**: Random Forest Classifier
- **Training**: Synthetic dataset with 1000 samples
- **Features**: 13 medical parameters
- **Output**: Binary classification (0 = No Heart Disease, 1 = Heart Disease) with probability score

## Important Disclaimer

⚠️ **This is a demonstration tool for educational purposes only. It should NOT be used for actual medical diagnosis. Always consult with qualified medical professionals for real health concerns.**

## Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, pandas, numpy
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Styling**: Modern CSS with gradients and animations

## License

This project is for educational purposes only.

## Author

Created as a capstone project for heart disease detection demonstration.

