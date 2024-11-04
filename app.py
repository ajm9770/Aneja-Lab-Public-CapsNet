from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
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
        # Add more examples as needed
    ]

    # Pass the data to the template
    return render_template('model_comparison.html', model1=model1, model2=model2, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
