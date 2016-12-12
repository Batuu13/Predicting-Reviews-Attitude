# Predicting-Reviews-Attitude
Predicting reviews attitude using Bag of Words and Naive Bayes with Python

In this project, I implemented an algorithm using python to predict if the given review is positive or negative. I used "Bag of words" method to get the frequency of the words and Naive Bayes to determine the class.

I used this project for my Machine Learning Class.

## Usage 
- _main.py_ this script contains all the code. You can download and show the data path to use it. Edit for your own projects.

In the report, you can find my comments and my results for this implementation.

## Imports

- You need to use sklearn.feature_extraction.text, glob, sys and re. (you can see it in _main.py_)

## Dataset

Movie review dataset[1] contains positive and negative reviews with a rating score (a negative review has a score smaller than or equal to 4 out of 10, while a positive review has a score bigger than or equal to 7 out of 10). You will try to implement Naive Bayes algorithm to predict the sentiment of the movie review.

- It contains 50,000 classified reviews as a text file in separate folders (25,000 for training and 25,000 for validation).
- Both of the training and validation sets include 12,500 positive reviews and 12,500 negative reviews). Each text file’s name includes their id and rating (id rating.txt).
- You can download the dataset from ftp://ftp.cs.hacettepe.edu.tr/pub/dersler/ BBM4XX/BBM409_ML/Assignment_2/MRDataset.zip.

### References

- [1]  Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts, Learning Word Vectors for Sentiment Analysis, The 49th Annual Meeting of the Association for Computational Linguistics (ACL), 2011

