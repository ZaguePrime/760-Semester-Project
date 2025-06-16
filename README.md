# Important Compatibility Information
- These transformer models are quite large and require a good GPU with a decent amount of video memory. We used an RTX4060Ti that has 16GiB of video memory.
- This is not set up to be run on Google Colabs, thus do not expect it to work flawlessly if you use Google Colabs.
- All relevant dependencies are included in the requirements.txt.
- Due to the already large model size, we reduced the scope of languages that would be used in this model. As of now the following languages are supported: 
    - Amharic (`am`)
    - Moroccan Arabic (`ary`)
    - Igbo (`ibo`)
    - Swahili (`swa`)
    - Yoruba (`yor`)
# How to run
- To run the code make sure you have all the dependencies installed, and be in the 'src' directory when running code.
- The language datasets are already on the github thus there is no need to use the loader.py and mixer.py. These files are used to download datasets, shuffle and merge them all into one.
- To run the baseline and get f-1 score evaluation, run baseline.py and the same goes to see our implementation with pipeline.py.
- To use your own input and see the results, run pipeline_predictor.py and paste your text when prompted.
- The train_langid.py and train_sentiment.py files were used as proof of concept for putting the pipeline together, and are not neccessary to see the implementation in action.
- If you would like to view the cross-validation, you may run baseline_cross.py and pipeline_cross.py.
# Folder Breakdown
## baseline_model
- This folder will be created and will contain the baseline model after it is trained, this can be used to see how it predicts on user input.
## baseline_model_cv
- This folder will be created when cross-validation is run, and it will contain the data for each fold.
## langid_model
- This contains the stand-alone language identification model.
## language_datasets
- This folder contains datasets for the languages chosen to work with. It also contains a .tsv file that has all language datasets shuffled into one for training and test purposes.
## pipeline_model
- This contains the full pipeline model, including both language identification and sentiment analysis.
## pipeline_cv
- This will contain the cross-fold validation folds and the model information used after training the pipeline model.
## sentiment model
- This contains the stand-alone sentiment analysis model.
## src
- This folder contains the .py or python files needed to run the model training and predicting. It is expected that you run pipeline.py to train and save the models locally as the files are too large to be uploaded to github.
