# Fine Tuning BERT on Stanford Sentiment Tree Bank

## Requirements
- Python 3.6
- Pytorch 1.2.0
- Transformers 2.0.0

Run the following command to install all required packages:
```
pip install -r requirements.txt
```
Also create a Models directory to save your trained models.
```
mkdir Models
```
## Data
Download the data from this [link](https://gluebenchmark.com/tasks). There will be a main zip file download option at the right side of the page. Extract the contents of the zip file and place them in data/SST/

## Training the model
To train the model with fixed weights of BERT layers, execute the following command from the project directory
```
python -m src.main -freeze_bert -gpu <gpu to use> -maxlen <maximum sequence length> -batch_size <batch size to use> -lr <learning rate> -maxeps <number of epochs>
```
To train the entire model i.e. both BERT layers and the classification layer just skip the -freeze_bert flag
```
python -m src.main -gpu <gpu to use> -maxlen <maximum sequence length> -batch_size <batch size to use> -lr <learning rate> -maxeps <number of epochs>
```

## Results
|Model Variant|Accuracy on Dev Set|
|-------------|-------------------|
|BERT (no finetuning)|82.59%|
|BERT (with finetuning)|88.29%|
