# Spam Message Classifier

A machine learning project that classifies SMS messages as either spam or ham (legitimate messages) using Natural Language Processing (NLP) and the Multinomial Naive Bayes algorithm.

## Overview

This project implements a spam detection system that:
- Preprocesses text data using NLTK
- Uses TF-IDF vectorization for feature extraction
- Trains a Multinomial Naive Bayes classifier
- Achieves high accuracy in spam detection
- Provides a prediction function for new messages

## Dataset

The model is trained on the SMS Spam Collection Dataset from GitHub:
- Source: `https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv`
- Format: TSV file with columns ['label', 'message']
- Contains SMS messages labeled as 'ham' (legitimate) or 'spam'

## Installation & Dependencies

To run this project, you'll need to install the following dependencies:

```bash
pip install nltk pandas scikit-learn
```

The notebook will automatically download required NLTK data:
- stopwords
- punkt tokenizer
- punkt_tab

## Project Structure

```
spam-classifier/
├── spam.ipynb          # Main Jupyter notebook with complete implementation
├── README.md           # This file
```

## How It Works

### 1. Data Preprocessing
- Convert text to lowercase
- Remove punctuation
- Tokenize messages into words
- Remove stopwords
- Clean and prepare text for vectorization

### 2. Feature Extraction
- Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Limited to 3000 most important features (words)
- Converts text data into numerical features

### 3. Model Training
- Uses Multinomial Naive Bayes classifier
- 80/20 train-test split with random state 42 for reproducibility
- Labels converted to numerical values (ham=0, spam=1)

### 4. Prediction Function
```python
def predict_spam(message):
    # Returns 'Spam' or 'Ham' classification
```

## Performance

The model achieves excellent performance metrics:

- **Accuracy**: High accuracy score (exact value shown when running the notebook)
- **Precision**: High precision for both spam and ham detection
- **Recall**: Excellent recall rates for both classes
- **F1-Score**: Strong F1 scores indicating good balance between precision and recall

## Usage Examples

```python
# Test the classifier
print(predict_spam("Win a free iPhone now!!!"))  # Output: Spam
print(predict_spam("Hey, let's meet for lunch tomorrow."))  # Output: Ham
```

## Key Features

- **Text Cleaning**: Comprehensive preprocessing pipeline
- **TF-IDF Vectorization**: Effective feature extraction for text data
- **Naive Bayes**: Efficient and effective for text classification
- **Real-time Prediction**: Ready-to-use prediction function
- **High Accuracy**: Well-performing model with detailed evaluation metrics

## Running the Project

1. Open the `spam.ipynb` file in Jupyter Notebook
2. Run all cells sequentially
3. The notebook will:
   - Install required packages
   - Download NLTK data
   - Load and preprocess the dataset
   - Train the model
   - Evaluate performance
   - Provide prediction examples

## Model Evaluation

The classification report includes:
- Precision, recall, and F1-score for both classes
- Support (number of instances)
- Macro and weighted averages
- Overall accuracy

## Future Enhancements

Potential improvements could include:
- Experimenting with other algorithms (SVM, Random Forest, Neural Networks)
- Adding more sophisticated text preprocessing (lemmatization, stemming)
- Implementing cross-validation
- Creating a web interface for real-time predictions
- Adding model persistence (saving/loading trained model)

## License

This project is for educational purposes. The dataset is publicly available for research and learning.

## Acknowledgments

- Dataset source: justmarkham/DAT8 repository
- Built with Python, NLTK, scikit-learn, and pandas
