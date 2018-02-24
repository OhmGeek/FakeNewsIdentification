import gensim

class Word2Vec(object):
    def __init__(self):
        pass
    def corpus_to_vec_list(self, corpus):
        # WORD2VEC pretrained model from http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/

        # TODO: load models dynamically using the module path, rather than hard coded.
        # TODO: remove limit when running on Lexo. Limit is only so that Lino doesn't crash badly.
        vec_model = gensim.models.KeyedVectors.load_word2vec_format('./deep/model/gnews-slim/data', binary=True, limit=1)
        vec_list = []
        for word in corpus.split(" "):
            # If in vocab, add to list, otherwise ignore.
            if(word in vec_model.vocab):
                vec_list.extend(vec_model[word])
        return vec_list