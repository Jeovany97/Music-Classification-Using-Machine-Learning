# Music Composition Classifier: A Machine Learning Pipeline 🎶
Project Overview

This project presents a machine learning pipeline designed to categorize music compositions based on their structural features. Specifically, it employs a Recurrent Neural Network (RNN), utilizing Bidirectional LSTMs, to learn patterns from MIDI files and classify them by composer.
The pipeline successfully extracts musical features (notes, chords) and translates them into a numerical sequence format for model consumption. The final model achieves an accuracy of at least 90% on the test set, demonstrating a robust ability to distinguish between the styles of different composers.

## Dependencies

| Category          | Library/Tool         | Purpose                    |
|-------------------|--------------------- |----------------------------|
| Data Processing	  | music21              | Feature extraction and MIDI parsing |                             
| Machine Learning  | TensorFlow/Keras     | Building and training the Bidirectional LSTM model. 
| Data Handling     | NumPy, scikit-learn  | Numerical operations, data splitting, and performance metrics.
| Environment       | Python               | Core programming language.

## Model Architecture & Training

The classification task is treated as a sequence-to-label problem, ideally suited for an RNN. The model architecture is:

| Layer | Type                | Output Dimension     | Notes 
|-----------------------------|--------------------- |----------------------------|----------|
| 1	    | Embedding           | 128                  | Converts integer-encoded notes into dense vectos
| 2     | Bidirectional(LSTM) | 128                  | Processes sequence data in both forward and backward directions, capturing more context. Dropout: 0.3.
| 3     | LSTM                | 64                   | Sequential processing layer. Dropout: 0.3.
| 4     | Dense               | 128                  | Fully connected layer with relu activation.
| 5     | Dropout             | N/A                  | Regularization layer to prevent overfitting. Rate: 0.4.
| 6     | Dense               | 6 (Number of classes)| Output layer with softmax activation for classification.

Compilation and Training
Optimizer: adam

Loss Function: categorical_crossentropy (suitable for multi-class classification).

Metrics: accuracy

Callbacks:

EarlyStopping: Monitors validation loss (val_loss) with a patience of 3 epochs to prevent overfitting.

ReduceLROnPlateau: Reduces the learning rate by a factor of 0.5 if the validation loss doesn't improve after 3 epochs, aiding convergence.



