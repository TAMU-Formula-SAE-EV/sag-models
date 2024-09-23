import numpy as np
import torch
import torch.nn as nn
import joblib

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=4, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, 1)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = torch.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out.squeeze()

def predict_voltage_sag_rf(input_features):
    model = joblib.load('random_forest_model_no_duty.pkl')
    input_features = np.array(input_features)
    if input_features.ndim == 1:
        input_features = input_features.reshape(1, -1)
    predicted_sag = model.predict(input_features)
    return predicted_sag

def predict_voltage_sag_lstm(input_sequence):
    scaler_X = joblib.load('scaler_X_no_duty.pkl')
    scaler_y = joblib.load('scaler_y_no_duty.pkl')
    num_sequences, seq_length, num_features = input_sequence.shape
    input_size = num_features
    hidden_size = 256
    model = LSTMModel(input_size, hidden_size, num_layers=4, dropout=0.2)
    model.load_state_dict(torch.load('best_lstm_model_no_duty.pth'))
    model.eval()
    input_sequence_2d = input_sequence.reshape(-1, num_features)
    input_sequence_scaled_2d = scaler_X.transform(input_sequence_2d)
    input_sequence_scaled = input_sequence_scaled_2d.reshape(num_sequences, seq_length, num_features)
    input_tensor = torch.Tensor(input_sequence_scaled)
    with torch.no_grad():
        predicted_sag = model(input_tensor)
    predicted_sag = scaler_y.inverse_transform(predicted_sag.numpy().reshape(-1, 1)).flatten()
    return predicted_sag


# Input structure is ['Actual_DCCurrent', 'Actual_ERPM', 'Actual_InputVoltage']

# Random Forest Prediction
# Single sample
rf_input = np.array([57.8, 8614, 422.0])  # Replace with actual values
rf_predicted_sag = predict_voltage_sag_rf(rf_input)
print("Random Forest Predicted Voltage Sag (Single Sample):", rf_predicted_sag)

# Multiple samples
rf_inputs = np.array([
    [57.8, 8614, 422.0],
    [50.7, 9040, 424.0],
    [48.5, 9858, 426.0]
])
rf_predicted_sags = predict_voltage_sag_rf(rf_inputs)
print("Random Forest Predicted Voltage Sags (Multiple Samples):", rf_predicted_sags)

# LSTM Prediction
# Single sequence
sequence_length = 10
n_features = 3
lstm_input_sequence = np.array([
    [57.8, 8614, 422.0],
    [50.7, 9040, 424.0],
    [48.5, 9858, 426.0],
    [48.7, 10537, 424.0],
    [57.3, 11227, 422.0],
    [56.2, 12019, 421.0],
    [54.4, 12672, 422.0],
    [53.9, 13231, 422.0],
    [51.7, 13745, 421.0],
    [55.2, 14590, 421.0]
])

lstm_input_sequence = lstm_input_sequence.reshape(1, sequence_length, n_features)
lstm_predicted_sag = predict_voltage_sag_lstm(lstm_input_sequence)
print("LSTM Predicted Voltage Sag (Single Sequence):", lstm_predicted_sag)

# # Multiple sequences
n_sequences = 5
lstm_input_sequences = np.random.rand(n_sequences, sequence_length, n_features)  # Replace with actual sequence data
lstm_predicted_sags = predict_voltage_sag_lstm(lstm_input_sequences)
print("LSTM Predicted Voltage Sags (Multiple Sequences):", lstm_predicted_sags)
