# Copyright (C) 2020  Panagiotis Deligiannis
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
import importlib
import inspect
import json
import logging
import string
import sys
from abc import ABCMeta, abstractmethod

import nltk

from src.util import functions
from src.util import stopwords


class BasePreprocessorFactory(metaclass=ABCMeta):

    @abstractmethod
    def create_preprocessor(self, text_data):
        pass


class DefaultPreprocessorFactory(BasePreprocessorFactory):
    """
    A default implementation of a preprocessor factory.

    The proper format for defining a preprocessor of this factory is "default [*MINIMUM_TERMS_PER_DOCUMENT]
    [*MINIMUM_CHARACTERS_PER_TERM]". The two arguments are optional and if omitted they get values
    MINIMUM_TERMS_PER_DOCUMENT=10 and MINIMUM_CHARACTERS_PER_TERM=3.

    Each preprocessor created performs the following transformations on each text
        1. Remove punctuation except the dash('-')
        2. Turn every text to lowercase
        3. Lemmatize then stem each term

    The preprocessor also makes sure that each document given has a non-empty string text representation. It also checks
    whether a document has at least MINIMUM_TERMS_PER_DOCUMENT terms and each term has at least
    MINIMUM_CHARACTERS_PER_TERM characters
    """

    stopwords_set = stopwords.stopwords
    lemmatizer = nltk.stem.WordNetLemmatizer()
    stemmer = nltk.stem.SnowballStemmer("english")

    def __init__(self, *args, **kwargs):
        super(DefaultPreprocessorFactory, self).__init__(*args, **kwargs)
        logger = kwargs.get("logger", None)
        functions.nltk_verify_resource("tokenizers/punkt", "punkt", logger=logger)
        functions.nltk_verify_resource("corpora/wordnet", "wordnet", logger=logger)

    def create_preprocessor(self, *args, **kwargs):
        try:
            minimum_terms_per_document = args[0]
        except IndexError:
            minimum_terms_per_document = 10
        try:
            minimum_term_length = args[1]
        except IndexError:
            minimum_term_length = 3
        logger = kwargs.get("worker_logger", None)
        if not logger:
            logger = logging.getLogger(__name__)
            logger.addHandler(logging.NullHandler())

        # def _filter(input_article_tuple, **kwargs):
        #     logger.debug("{}: Got article {}".format(kwargs["worker_id"], input_article_tuple[0]))
        #     try:
        #         estimated_lang = langdetect.detect(input_article_tuple[1])
        #         if estimated_lang == language:
        #             return [input_article_tuple]
        #     except Exception:
        #         pass
        #     return None

        def _preprocess(input_article_tuple, **kwargs):
            nonstemmed = dict()
            logger.debug("{}: Got article {}".format(kwargs["worker_id"], input_article_tuple[0]))
            text = input_article_tuple[1]
            assert type(text) is str, "Document \"{}\" does not have a string text".format(input_article_tuple[0])
            assert text != "", "Document \"{}\" has no text".format(input_article_tuple[0])

            table = str.maketrans('', '', string.punctuation.replace('-', ''))
            no_punctuation = text.lower().translate(table)
            terms = nltk.tokenize.word_tokenize(no_punctuation)
            stemmed_terms = list()
            for term in terms:
                if term not in self.stopwords_set and term.isalpha():
                    stemmed = self.stemmer.stem(self.lemmatizer.lemmatize(term))
                    if len(stemmed) >= minimum_term_length:
                        stemmed_terms.append(stemmed)
                        if stemmed not in nonstemmed:
                            nonstemmed[stemmed] = dict()
                        if term not in nonstemmed[stemmed]:
                            nonstemmed[stemmed][term] = 1
                        else:
                            nonstemmed[stemmed][term] += 1

            if len(stemmed_terms) >= minimum_terms_per_document:
                output_article_tuple = *input_article_tuple[:1], " ".join(stemmed_terms), *input_article_tuple[2:]
                return [output_article_tuple, json.dumps(nonstemmed)]
            else:
                raise Exception("Too few terms for document \"{}\"".format(input_article_tuple[0]))

        return _preprocess


available_preprocessors = {
    (cl[0].lower()[:-len("PreprocessorFactory")] if cl[0].endswith("PreprocessorFactory") else cl[0].lower()): cl[1]
    for cl in inspect.getmembers(sys.modules[__name__], inspect.isclass) if
    (issubclass(cl[1], BasePreprocessorFactory) and cl[0] != "BasePreprocessorFactory")
}

custom_preprocessors = importlib.util.find_spec("application.preprocessors")
if custom_preprocessors is not None:
    loader = custom_preprocessors.loader
    custom_module = loader.load_module()
    for ccl in inspect.getmembers(custom_module, inspect.isclass):
        if issubclass(ccl[1], BasePreprocessorFactory) and ccl[0] != "BasePreprocessorFactory":
            if ccl[0].endswith("PreprocessorFactory"):
                description_string = ccl[0][:-len("PreprocessorFactory")].lower()
            else:
                description_string = ccl[0].lower()

            if description_string in available_preprocessors:
                description_string = "app_" + description_string
                if description_string in available_preprocessors:
                    continue
            available_preprocessors[description_string] = ccl[1]

# available_preprocessors = {
#     (cl[0].lower()[:-len("Preprocessor")] if cl[0].endswith("Preprocessor") else cl[0].lower()): cl[1]
#     for cl in inspect.getmembers(sys.modules[__name__], inspect.isclass) if
#     (issubclass(cl[1], BasePreprocessor) and cl[0] != "BasePreprocessor" and cl[0].endswith("Preprocessor"))
# }
#
# custom_preprocessors = importlib.util.find_spec("application.preprocessors ")
# if custom_preprocessors is not None:
#     loader = custom_preprocessors.loader
#     custom_module = loader.load_module()
#     for ccl in inspect.getmembers(custom_module, inspect.isclass):
#         if issubclass(ccl[1], BasePreprocessor) and ccl[0] != "BasePreprocessor":
#             if ccl[0].endswith("Preprocessor"):
#                 description_string = ccl[0][:-len("Preprocessor")].lower()
#             else:
#                 description_string = ccl[0].lower()
#
#             if description_string in available_preprocessors:
#                 description_string = "app_" + description_string
#                 if description_string in available_preprocessors:
#                     continue
#             available_preprocessors[description_string] = ccl[1]
