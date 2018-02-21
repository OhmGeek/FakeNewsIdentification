import csv
from nltk.tokenize import word_tokenize

DEFAULT_SETTINGS = {
    "convert_to_lowercase": True,
    "remove_punctuation": True
}


class ShallowDataProcessor(object):
    def __init__(self, settings=DEFAULT_SETTINGS):
        self.settings = settings
        self.data_str = ""

    def read_dataset_from_string(self, data_str):
        self.data_str = data_str

    def read_dataset_from_file(self, filename):
        with open(filename) as f:
            self.data_str = f.read()
    
    def __parse_csv(self, csv_data):
        csv_data = csv_data.strip() # Remove any loose whitespace on either side
        rows = csv_data.split('\n')

        csv_reader = csv.reader(rows)
        return list(csv_reader) # Read into a list, such that dataset[ROW][col] displays data.
        
    def __get_word_features(self, passage):
        # FROM https://streamhacker.com/2010/05/10/text-classification-sentiment-analysis-naive-bayes-classifier/
        return dict([(word, True) for word in passage])

    def process(self):
        # First, go through each row.
        # Filter the text of the article:
        #   - remove punctuation
        #   - all to lower case
        # 

        # COL 0: ID
        # Col 1: TExt
        # Col 2: Label
        
        dataset = self.__parse_csv(self.data_str)
        
        # Filter into fake and real
        real_data = [data for data in dataset if '0' in data[2]]
        fake_data = [data for data in dataset if '1' in data[2]]
        
        real_features = [(self.__get_word_features(data[1]), '0') for data in real_data]
        fake_features = [(self.__get_word_features(data[1]), '1') for data in fake_data]

        


class NaiveBayesProcessor(object):
    def __init__(self):
        pass

    def train(self, data):
        pass

    def get_output(self, input):
        pass


def main():
    dp = ShallowDataProcessor()
    filename = 'dataset.csv'
    dp.read_dataset_from_file(filename)
    dp.process()

if __name__ == '__main__':
    main()