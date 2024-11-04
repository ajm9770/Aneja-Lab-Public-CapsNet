from flask import Flask, render_template
from capsnet.model.capsnet import CapsNet3D
from capsnet.model.unet import UNet3D
from capsnet.model.train.capsnet_train import TrainCapsNet3D
from capsnet.model.train.unet_train import TrainUNet3D
from capsnet.engine.data_loader import AdniDataset, make_image_list
from capsnet.util.model_util import load_model, evaluate_model
from pathlib import Path
from torch.utils.data import DataLoader

import os

file_path = os.path.realpath(__file__)

# Get only the directory part of the path
project_root = Path(os.path.dirname(file_path))

images_csv = "data/datasets_local/train_inputs.csv"
masks_csv = "data/datasets_local/train_outputs.csv"

images_path = project_root/images_csv
masks_path = project_root/masks_csv

image_list = make_image_list(images_path)
mask_list = make_image_list(masks_path)

adni = AdniDataset(
    image_list,
    mask_list,
    maskcode=14,
    crop=(64, 64, 64),
    cropshift=(0, 7, 0),
    testmode=False,
)
dataloader = DataLoader(dataset=adni, batch_size=4, shuffle=True)

app = Flask(__name__)

@app.route('/example')
def model_comparison():
    # Mock data for models and predictions
    model1 = {
        'name': 'Model A',
        'metrics': {
            'losses': 0.0,
            'accuracy': 0.95,
            'precision': 0.92,
            'recall': 0.90,
            'f1_score': 0.91
        }
    }

    model2 = {
        'name': 'Model B',
        'metrics': {
            'losses': 0.0,
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

    unet_loc = project_root / "data/results/temp/saved_unet.pth.tar"
    capsnet_loc = project_root / "/data/results/temp/saved_capsnet.pth.tar"
    unet_model = load_model(UNet3D, unet_loc)
    capsnet_model = load_model(CapsNet3D, capsnet_loc)


    model1 = {
        'name': 'UNet',
        'metrics': evaluate_model(unet_model, dataloader)
    }

    model2 = {
        'name': 'CapsNet',
        'metrics': evaluate_model(unet_model, dataloader)
    }

    predictions = [
            {'input': 'Example 1', 'model1_prediction': 'Positive', 'model2_prediction': 'Negative', 'actual': 'Positive'},
            {'input': 'Example 2', 'model1_prediction': 'Negative', 'model2_prediction': 'Negative', 'actual': 'Negative'},
            {'input': 'Example 3', 'model1_prediction': 'Positive', 'model2_prediction': 'Positive', 'actual': 'Positive'},
        ]

    return render_template('model_comparison.html', model1=model1, model2=model2, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
