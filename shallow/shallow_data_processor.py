import csv
import re
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

        if self.settings['remove_punctuation']: dataset = self.__filter_non_words(dataset)
        if self.settings['convert_to_lowercase']: dataset = self.__set_all_lowercase(dataset)
        # Filter into fake and real
        return dataset