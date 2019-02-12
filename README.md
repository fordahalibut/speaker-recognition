# Speaker Recognition using Neural Networks and the Critical Role podcast

## Problem Statement

Using neural networks and a custom dataset, can we build an accurate speaker recognition model?

## Executive Summary

Speaker recognition is a hot topic of research in deep learning, but many of the available datasets do not closely resemble the real world. Accounting for multiple speakers, music and other background noise, and differences in recording quality greatly impacts the performance of recognition models.

In this project I wanted to explore the use of neural networks in speaker recognition using a real, messy, unlabelled dataset available to the public.

[The Critical Role podcast](https://critrole.com/) (copyright Geek and Sundry) is a longrunning podcast featuring a cast of professional voice actors playing Dungeons and Dragons together. Episodes have music and background noise, non-speech segments such as laughter, and frequently overlapping voices.

Initially I wanted to explore unsupervised clustering techniques described in recent papers by Google researchers, but due to time constraints settled for manually labelling a portion of the dataset using tools on hand. This is an area I would like to explore further in the future.

Selecting a Long Short Term Memory network, a type of recurrent neural network, I achieved 87% accuracy on binary classification of the main cast character and host of the show, Matthew Mercer. To generalize to classification of the other cast members I would need more labelled data to even out the class distributions. To aquire this data I investigated force-fit methods of matching the podcast transcripts to the audio timestamps, but didn't achieve good enough results for use in a model. Future investigation might yield better results, giving access to a much larger set of labelled data.

## Data Sources

The data was collected by scraping the feed xml (which contains links to episode downloads):
https://pbcdn1.podbean.com/criticalrolepodcast.geekandsundry.com/feed.xml

Transcripts were evaluated for use in a force-fitted labelling scenario, but ultimately were not utilized in the final model:
https://crtranscript.tumblr.com/transcripts

## Audio Processing

The collected audio files were first downsampled to 16kHz, converted from stereo to mono and saved as `.wav` files for further processing using the `librosa` library.

Labels were generated using the [Hipstas Audio Labeler](https://github.com/hipstas/audio-labeler), which randomly samples 1 second clips from provided audio files, appending labels and timestamps to a `.csv` file.

Using these labels, the audio files were split into 1 second segments for feature extraction using the `pydub` library.

A 3 minute validation audio clip was also processed into 1 second segments for easy visualization of the model results.

## Feature Extraction

The raw audio segment waveforms were transformed to Mel spectrograms (using 128 Mels) and converted to decibel scales, from which features were extracted.

24 Mel-frequency cepstral coefficients (MFCCs) and first and second order derivatives were extracted from each log-scaled Mel spectrogram using the `librosa` library (for a total of 24 x 24 x 24 = 72 features) to generate characteristic i-vectors.

The same transformations and feature extractions were applied to the validation audio segments.

## Modelling

Two neural network models were evaluated:

- Feed forward neural network
Resulted in ~85% accuracy and ~.375 binary cross entropy on the evaluation data. High variability between train and test metrics indicated overfitting was a problem which dropout did not resolve. Probably a result of the relatively small dataset.

- Long Short Term Memory neural network
Initialized and trained an LSTM neural network with 128 neurons with L2 recurrence regularization, one hidden dense layer, and a dropout layer. 
Resulted in ~89% accuracy and ~.305 binary cross entropy on the evaluation data. Metrics between train and test were very similar, indicating that overfitting was not a problem.

## Results

- Predictions were generated against an unseen 3 minute audio clip with a good distribution of speakers, instances of multiple speakers, and non-speech segments. Results were in line with predicted performance of the model, although it tended to overpredict the positive class.

## Conclusion

- A LSTM model achieved good performance for the relatively limited available data. To further pursue this topic, I would suggest gathering / labelling more data from the podcast and incorporating a universal background model to pad out the other speaker classes. With enough data this model could be generalized from a binary classification to a multiclass classification to identify each of the primary speakers in the podcast.

- I would like to pursue the topic of unsupervised learning for this dataset, taking into account many of the recent advances in speaker recognition and diarization to generate predicted class labels for use in a model similar to the one above, without the need for the time-consuming manual labelling of data.