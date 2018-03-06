import gensim
import numpy as np
class Word2Vec(object):
    def __init__(self):
        pass
    def corpus_to_vec_list(self, corpus):
        # WORD2VEC pretrained model from http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/

        # TODO: load models dynamically using the module path, rather than hard coded.
        # TODO: remove limit when running on Lexo. Limit is only so that Lino doesn't crash badly.
        vec_model = gensim.models.KeyedVectors.load_word2vec_format('./deep/model/gnews-slim/data', binary=True)
        first = True
        vec_list = None
        word_counter = 0
        return np.nanmean([vec_model[word] for word in corpus.split(" ") if word in vec_model.vocab], dtype=np.float64, axis=0)
        # for word in corpus.split(" "):
        #     # If in vocab, add to list, otherwise ignore.
        #     if(word in vec_model.vocab):
        #         word_counter += 1.0
        #         arr = np.array(vec_model[word])
        #         arr = arr.astype(np.float64)
        #         arr.setflags(write=1)
        #         if(first):
        #             vec_list = arr
        #         else:
        #             vec_list = np.add(vec_list, arr)
        #
        # if(vec_list is None):
        #     return np.array([])
        # vec_list = np.divide(vec_list, word_counter)
        # return vec_list
