import torch
import torch.nn as nn
import numpy as np
import os

class ListNet(nn.Module):
    def __init__(self, input_size=6, hidden_size=256):
        super(ListNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 22)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        scores = self.scorer(attended_features)
        return scores

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "listnet_core.pth")
    model = ListNet()
    device = torch.device("cpu")  # Render free tier uses CPU
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    model.to(device)
    model.eval()
    return model

def predict_and_rank(model, X_new, crop_labels):
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_new)
        scores = model(X_tensor)
        probabilities = torch.softmax(scores, dim=1)
        min_prob = torch.min(probabilities)
        max_prob = torch.max(probabilities)
        if min_prob == max_prob:
            scaled_probs = torch.full_like(probabilities, 5)
        else:
            scaled_probs = 1 + 9 * (probabilities - min_prob) / (max_prob - min_prob)
        scaled_probs_np = scaled_probs.numpy()
        ranked_crops = np.argsort(-scaled_probs_np[0])
        return scaled_probs_np, ranked_crops