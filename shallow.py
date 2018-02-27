import csv
import re
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import random
from shallow.shallow_data_processor import ShallowDataProcessor
from shallow.classifier import NaiveBayesProcessor


def main():
    dp = ShallowDataProcessor()
    filename = 'dataset.csv'
    dp.read_dataset_from_file(filename)
    dataset = dp.process()
    
    # Split Data
    pivot = int(0.5 * len(dataset))

    train_dataset = dataset[:pivot]
    test_dataset = dataset[pivot+1:]


    # Train, and test
    classifier = NaiveBayesProcessor()
    classifier.train(train_dataset)    
    print("Test result: " + str(classifier.test_model(test_dataset)))

if __name__ == '__main__':
    main()