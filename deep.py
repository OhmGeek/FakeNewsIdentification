from shallow.shallow_data_processor import ShallowDataProcessor
from deep.word_representation import Word2Vec
def main():
    dp = ShallowDataProcessor()
    filename = 'dataset.csv'
    dp.read_dataset_from_file(filename)
    dataset = dp.process()
    
    wtv = Word2Vec()
    wtv.corpus_to_vec_list(dataset[1][1])

if __name__ == '__main__':
    main()