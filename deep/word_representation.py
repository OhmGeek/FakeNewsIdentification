import gensim

class Word2Vec(object):
    def __init__(self):
        pass
    def corpus_to_vec_list(self, corpus):
        # WORD2VEC pretrained model from http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
        vec_model = gensim.models.KeyedVectors.load_word2vec_format('./deep/model/gnews.bin/data', binary=True)
        vec_list = []
        for word in corpus.split(" "):
            # If in vocab, add to list, otherwise ignore.
            if(word in vec_model.vocab):
                vec_list.append(vec_model[word])
                print("Word: " + word + " rep: " + str(vec_model[word]))
        return vec_list