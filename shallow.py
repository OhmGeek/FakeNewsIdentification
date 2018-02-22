import csv
import re
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import random
from shallow.shallow_data_processor import ShallowDataProcessor


class NaiveBayesProcessor(object):

    def train(self, training_data):

        parameters = {}


        self.model = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer(use_idf=False)),
            ('clf', MultinomialNB()),
        ])

        # fit data, using the words, targeting the real/fake
        self.model.fit(training_data[:][1], training_data[:][2])

    def get_output(self, input):
        pass

    def test_model(self, test_data):
        return self.model.score(test_data[:][1], test_data[:][2])
def main():
    dp = ShallowDataProcessor()
    filename = 'dataset.csv'
    dp.read_dataset_from_file(filename)
    dataset = dp.process()
    
    train_dataset = random.sample(dataset, int(len(dataset) * 0.65))
    test_dataset = random.sample(dataset, int(len(dataset) * 0.35))


    classifier = NaiveBayesProcessor()
    classifier.train(train_dataset)    
    print("Test result: " + str(classifier.test_model(test_dataset)))

if __name__ == '__main__':
    main()