"""
    This file contains various methods for prefiltering
    text. 


"""
import re
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
#---------- HELPER FUNCTIONS ---------------
def filter_urls(text):
    # use Django URL check
    # TODO: reference the fact we are using this.
    regex = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    )

    # Replace any string matching a URL
    # with the empty string.
    return regex.sub(text, '')

def filter_stop_words(text):
    pass




class PrefilterManager(object):
    def __init__(self, text):
        self.text = text



