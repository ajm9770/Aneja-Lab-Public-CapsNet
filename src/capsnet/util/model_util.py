import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from capsnet.engine.loss_functions import DiceLoss, DiceBCELoss, DiceReconLoss
from torchvision import transforms
import pandas as pd

# Example: Dummy model class (replace with your actual model class if necessary)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define your model architecture
        self.layer = nn.Linear(10, 2)  # Example layer; adjust as needed

    def forward(self, x):
        return self.layer(x)

# Load the model
def load_model(model_type: nn.Module, model_path: Path):
    model = MyModel()  # Replace with your model class
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()  # Set the model to evaluation mode
    return model

# Make a prediction
def predict(model, input_data):
    # Preprocess the input data here (assuming input_data is already a tensor or needs basic conversion)
    # input_data should be transformed to a tensor if it’s not already one
    with torch.no_grad():
        output = model(input_data)
        predicted_class = torch.argmax(output, dim=1)
    return predicted_class.item()

def evaluate_model(model, test_loader):
    """
    Evaluate the model on a test dataset and return accuracy and other metrics.
    :param model: Loaded PyTorch model
    :param test_loader: DataLoader containing test data
    :return: Dictionary with metrics
    """
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # No gradient tracking needed
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    losses = pd.load_csv(
            Path().home() / 'src/capsnet/data/results/temp/niftis' / "scans_losses.csv"
    )
    
    # Calculate metrics
    mean_loss = losses['loss'].mean()
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return {
        'mean loss': mean_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }