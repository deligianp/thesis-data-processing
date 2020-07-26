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
import importlib.util
import inspect
import logging
import math
import sys
from abc import ABCMeta, abstractmethod

import langdetect
import nltk


class BaseFilter(metaclass=ABCMeta):

    @abstractmethod
    def perform_filter(self, input_document_tuple, **kwargs):
        pass


class LanguageFilter(BaseFilter):
    """
    It creates a language filter based on the given language argument.
    
    The proper format for defining a language is "language [LANGUAGE] [*ITERATIONS]" where LANGUAGE is one of the
    following:
        * da
        * de
        * el
        * en
        * es
        * fi
        * fr
        * nl
        * no
        * pt
        * ru
        * sv

    Iterations refer to the time the estimation method must classify the document's text in the defined language. It is
    used to compensate for the indeterministic nature of the language estimating method. A document is accepted if its
    text is classified as the defined language in all of the ITERATIONS. ITERATIONS argument is optional; if omitted the
    estimation will only run once per text.
    """
    implemented_languages = {"da", "de", "el", "en", "es", "fi", "fr", "nl", "no", "pt", "ru", "sv"}

    def __init__(self, *args, **kwargs):
        super(LanguageFilter, self).__init__()
        self.language = args[0]
        try:
            self.iterations = int(args[1])
        except IndexError:
            self.iterations = 1
        assert self.language in self.implemented_languages, "Unknown language defined. Supported languages: " + \
                                                            ", ".join(self.implemented_languages)
        # logger = kwargs.get("worker_logger", None)
        # if not logger:
        #     logger = logging.getLogger(__name__)
        #     logger.addHandler(logging.NullHandler())
        # self.logger = logger

    def perform_filter(self, input_document_tuple, **kwargs):
        if "worker_logger" in kwargs:
            if "worker_id" in kwargs:
                kwargs["worker_logger"].debug("{}: Got article {}".format(kwargs["worker_id"], input_document_tuple[0]))
            else:
                kwargs["worker_logger"].debug("Got article {}".format(input_document_tuple[0]))
        try:
            language_estimation_iterations = self.iterations
            while True:
                estimated_lang = langdetect.detect(input_document_tuple[1])
                language_estimation_iterations -= 1
                assert estimated_lang == self.language, "Document \"{}\" failed to classify as a text in language " \
                                                        "\"{}\"".format(input_document_tuple[0], self.language)
                if language_estimation_iterations == 0:
                    return [input_document_tuple]
        except Exception:
            pass
        return None


class WordCountFilter(BaseFilter):
    """
    It creates a filter that accepts a document text only when the number of words is within a predefined range.

    The proper format for defining a language is "wordcount [MINIMUM_AMOUNT] [*MAXIMUM_AMOUNT]". The filter uses the
    nltk library in order to tokenize the document text and count the number of resulting tokens.

    MAXIMUM_AMOUNT is an optional argument; if omitted, no upper range exists for the text. The range defined by
    MINIMUM_AMOUNT and MAXIMUM_AMOUNT is considered inclusive for the edge values.
    """

    def __init__(self, *args, **kwargs):
        self.minimum_amount = int(args[0])
        try:
            self.maximum_amount = int(args[1])
        except IndexError:
            self.maximum_amount = math.inf
        assert self.minimum_amount < self.maximum_amount, "Minimum amount of terms must be different and less than " \
                                                          "the maximum amount of terms"

    def perform_filter(self, input_document_tuple, **kwargs):
        if "worker_logger" in kwargs:
            if "worker_id" in kwargs:
                kwargs["worker_logger"].debug("{}: Got article {}".format(kwargs["worker_id"], input_document_tuple[0]))
            else:
                kwargs["worker_logger"].debug("Got article {}".format(input_document_tuple[0]))
        document_terms = nltk.tokenize.word_tokenize(input_document_tuple[1])
        assert self.minimum_amount <= len(document_terms) <= self.maximum_amount, "Document \"{}\" amount of terms " \
                                                                                  "not between the acceptable " \
                                                                                  "range".format(
            input_document_tuple[0]
        )
        return [input_document_tuple]


available_filters = {
    (cl[0].lower()[:-len("Filter")] if cl[0].endswith("Filter") else cl[0].lower()): cl[1]
    for cl in inspect.getmembers(sys.modules[__name__], inspect.isclass) if
    (issubclass(cl[1], BaseFilter) and cl[0] != "BaseFilter" and cl[0].endswith("Filter"))
}

custom_filters = importlib.util.find_spec("application.filters")
if custom_filters is not None:
    loader = custom_filters.loader
    custom_module = loader.load_module()
    for ccl in inspect.getmembers(custom_module, inspect.isclass):
        if issubclass(ccl[1], BaseFilter) and ccl[0] != "BaseFilter":
            if ccl[0].endswith("Filter"):
                description_string = ccl[0][:-len("Filter")].lower()
            else:
                description_string = ccl[0].lower()

            if description_string in available_filters:
                description_string = "app_" + description_string
                if description_string in available_filters:
                    continue
            available_filters[description_string] = ccl[1]
