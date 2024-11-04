from flask import Flask, render_template
from capsnet.model.capsnet import CapsNet3D
from capsnet.model.unet import UNet3D
from capsnet.model.train.capsnet_train import TrainCapsNet3D
from capsnet.model.train.unet_train import TrainUNet3D
from capsnet.engine.data_loader import AdniDataset
from capsnet.util.model_util import load_model, evaluate_model
from pathlib import Path
from torch.utils.data import DataLoader



test_dataset = AdniDataset(
            self.inputs_paths,
            self.outputs_paths,
            maskcode=self.output_code,
            crop=self.crop,
            cropshift=self.cropshift,
            testmode=True,
        )
test_loader = DataLoader(test_dataset, batch_size=16)

app = Flask(__name__)

@app.route('/example')
def model_comparison():
    # Mock data for models and predictions
    model1 = {
        'name': 'Model A',
        'metrics': {
            'accuracy': 0.95,
            'precision': 0.92,
            'recall': 0.90,
            'f1_score': 0.91
        }
    }

    model2 = {
        'name': 'Model B',
        'metrics': {
            'accuracy': 0.92,
            'precision': 0.89,
            'recall': 0.88,
            'f1_score': 0.89
        }
    }

    predictions = [
        {'input': 'Example 1', 'model1_prediction': 'Positive', 'model2_prediction': 'Negative', 'actual': 'Positive'},
        {'input': 'Example 2', 'model1_prediction': 'Negative', 'model2_prediction': 'Negative', 'actual': 'Negative'},
        {'input': 'Example 3', 'model1_prediction': 'Positive', 'model2_prediction': 'Positive', 'actual': 'Positive'},
    ]

    # Pass the data to the template
    return render_template('model_comparison.html', model1=model1, model2=model2, predictions=predictions)

@app.route('/capsnet')
def unet_model_comparison():
    '''Route for testing results. Not to be used for production'''
    # load unet/capsnet files

    unet_loc = Path().home() / "src/capsnet/data/results/temp/saved_unet.pth.tar"
    capsnet_loc = Path().home() / "src/capsnet/data/results/temp/saved_capsnet.pth.tar"
    unet_model = load_model(UNet3D, unet_loc)
    capsnet_model = load_model(CapsNet3D, capsnet_loc)


    model1 = {
        'name': 'UNet',
        'metrics': {
            'accuracy': 0.95,
            'precision': 0.92,
            'recall': 0.90,
            'f1_score': 0.91
        }
    }

    model2 = {
        'name': 'CapsNet',
        'metrics': {
            'accuracy': 0.92,
            'precision': 0.89,
            'recall': 0.88,
            'f1_score': 0.89
        }
    }

    predictions = [
            {'input': 'Example 1', 'model1_prediction': 'Positive', 'model2_prediction': 'Negative', 'actual': 'Positive'},
            {'input': 'Example 2', 'model1_prediction': 'Negative', 'model2_prediction': 'Negative', 'actual': 'Negative'},
            {'input': 'Example 3', 'model1_prediction': 'Positive', 'model2_prediction': 'Positive', 'actual': 'Positive'},
        ]

    return render_template('model_comparison.html', model1=model1, model2=model2, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
