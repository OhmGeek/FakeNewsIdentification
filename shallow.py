import csv

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
        
    def process(self):
        # First, go through each row.
        # Filter the text of the article:
        #   - remove punctuation
        #   - all to lower case
        # 
        # Build up a dictionary of words
        dataset = self.__parse_csv(self.data_str)
        for row in dataset:
            print(row)



class NaiveBayesProcessor(object):
    def __init__(self):
        pass

    def train(self, data):
        pass

    def get_output(self, input):
        pass


def main():
    dp = ShallowDataProcessor()
    csv_string = "1,2,3\n4,5,6\n"
    dp.read_dataset_from_string(csv_string)
    dp.process()

if __name__ == '__main__':
    main()