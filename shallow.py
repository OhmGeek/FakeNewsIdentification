import csv
import re
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import random
from shallow.shallow_data_processor import ShallowDataProcessor

DEFAULT_MODEL_SETTINGS = {
    'use_tfidf': True
}

class NaiveBayesProcessor(object):
    def __init__(self, model_settings=DEFAULT_MODEL_SETTINGS):
        settings = []
        settings.append(('vect', CountVectorizer()))

        if(model_settings.get('use_tf')):
            settings.append(('tfidf', TfidfTransformer(use_idf=False)))
        elif(model_settings.get('use_tfidf')):
            settings.append(('tfidf', TfidfTransformer(use_idf=True)))

        settings.append(('clf', MultinomialNB()))
        
        self.model = Pipeline(settings)

    def train(self, training_data):
        """
            Train the Naive Bayes processor on a set of training data

            Arguments:
                training_data -- a list of lists containing the training data, in the form of ['ID', 'passage', 'classification']
        """
        parameters = {}
        passages = [passage[1] for passage in training_data]
        target_class = [val[2] for val in training_data]
        
        
        # fit data, using the words, targeting the real/fake
        self.model.fit(passages, target_class)
    def get_output(self, input):
        pass

    def test_model(self, test_data):
        """
            Test the trained model on a set of test data.

            Arguments:
                test_data -- a list of lists containing the training data, in the form of ['ID', 'passage', 'classification']
        """
        test_data_passages = [val[1] for val in test_data]
        test_data_classif = [val[2] for val in test_data]
        return self.model.score(test_data_passages, test_data_classif)

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