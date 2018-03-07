import csv
import re
from collections import Counter

DEFAULT_SETTINGS = {
    "convert_to_lowercase": True,
    "remove_punctuation": True,
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
        

    def __filter_non_words(self, dataset):
        for row in dataset:
            row[1] = re.sub('\W+', ' ', row[1])
        return dataset

    def __set_all_lowercase(self, dataset):
        for row in dataset:
            row[1] = row[1].lower()
        return dataset

    def __filter_stop_words(self, dataset, stopwords):
        for row in dataset:
            words = row[1].split(" ")
            words = [word for word in words if word.lower() not in stopwords]
            row[1] = ' '.join(words)
        return dataset
    
    def __filter_out_uncommon_words(self, dataset, top_words):
        word_counts = {}
        for row in dataset:
            words = row[1].split(" ")
            for word in words:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
        
        # Now we have counts, return
        counter = Counter(word_counts)
        for row in dataset:
            words = row[1].split(" ")
            words = [word for word in words if word in counter.most_common(top_words)]
            row[1] = ' '.join(words)
        return dataset

    def process(self,stopwords=[],get_most_common_words=-1):
        # First, go through each row.
        # Filter the text of the article:
        #   - remove punctuation
        #   - all to lower case
        # 

        # COL 0: ID
        # Col 1: TExt
        # Col 2: Label
        
        dataset = self.__parse_csv(self.data_str)

        if self.settings['remove_punctuation']: dataset = self.__filter_non_words(dataset)
        if self.settings['convert_to_lowercase']: dataset = self.__set_all_lowercase(dataset)
        if(get_most_common_words > 0):
            dataset = self.__filter_out_uncommon_words(dataset, get_most_common_words)

        dataset = self.__filter_stop_words(dataset, stopwords)
        # Filter into fake and real
        return dataset