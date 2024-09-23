# Voltage Sag Prediction

Predict voltage sag using Random Forest and LSTM models.

## Requirements

- Python 3.7+
- [NumPy](https://numpy.org/)
- [PyTorch](https://pytorch.org/)
- [joblib](https://joblib.readthedocs.io/)

Install dependencies using:

```bash
pip install numpy torch joblib
```
## Setup

Ensure the following model files are in the same directory as the script:

- random_forest_model_no_duty.pkl
- scaler_X_no_duty.pkl
- scaler_y_no_duty.pkl
- best_lstm_model_no_duty.pth

You can download them from this google drive: [Get the models](https://drive.google.com/drive/folders/1fD-Nq-UCI3341Zggia2z34q3k1s1e1nX?usp=sharing)
## Usage

1. Clone the repo
```
git clone https://github.com/yourusername/voltage-sag-prediction.git
cd voltage-sag-prediction
```

2. Inputs should follow the structure:
```
['Actual_DCCurrent', 'Actual_ERPM', 'Actual_InputVoltage']
```

3. The script includes examples for both Random Forest and LSTM predictions.

