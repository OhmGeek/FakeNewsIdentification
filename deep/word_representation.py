import gensim
import numpy as np
class Word2Vec(object):
    def __init__(self):
        self.vec_model = gensim.models.KeyedVectors.load_word2vec_format('./deep/model/gnews-slim/data', binary=True)
    def corpus_to_vec_list(self, corpus):
        # WORD2VEC pretrained model from http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/

        # TODO: load models dynamically using the module path, rather than hard coded.
        # TODO: remove limit when running on Lexo. Limit is only so that Lino doesn't crash badly.
        first = True
        vec_list = None
        word_counter = 0
        return np.nanmean([self.vec_model[word] for word in corpus.split(" ") if word in self.vec_model.vocab], dtype=np.float64, axis=0)