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
import argparse
import bz2
import datetime
import logging
import os
import pickle

from gensim.corpora import Dictionary
from gensim.models import TfidfModel

from src.core.file.readers import Bz2BagReader
from src.util import functions

reporting_batch = 10000


def vectorize_preprocessed(dictionary_size, reader, output_dir_path, tfidf=False, output_name=None,
                           logger=None, least_documents=1, max_document_ratio=1.0, keep_terms=None,
                           file_overwrite_confirmation_function=lambda path: True):
    if not logger:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    logger.debug("Checking that dictionary size, the least amount of document occurrences and maximum document "
                 "occurrence ratio are valid natural numbers.")
    logger.debug("Checking if the least amount of document occurrences is at least 1.")
    assert least_documents > 0, "The minimum amount of documents for an acceptable term must be at least 1"
    logger.debug("The least amount of document occurrences is at least 1.")
    logger.debug("Checking if the least amount of document occurrences is an integer.")
    assert type(least_documents) is int, "The minimum amount of documents for an acceptable term must be an integer"
    logger.debug("The least amount of document occurrences is an integer.")
    logger.debug("Checking if the max document occurrence ratio is a valid real number in range (0,1.0].")
    assert 0 < max_document_ratio <= 1.0, "The maximum ratio of documents for an acceptable term must greater than 0 " \
                                          "and less than or equal to 1.0"
    logger.debug("The max document occurrence ratio is a valid real number in range (0,1.0].")
    logger.debug("Checking if the dictionary size is at least 1.")
    assert dictionary_size > 0, "The maximum amount of dictionary terms must be at least 1"
    logger.debug("Dictionary size is at least 1.")
    logger.debug("Checking if the dictionary size is an integer.")
    assert type(dictionary_size) is int, "The maximum amount of dictionary terms must be an integer"
    logger.debug("The least amount of document occurrences is an integer.")

    # Get output directory path
    logger.debug("Attempting to get full absolute path of defined output directory's path.")
    output_dir_path = os.path.abspath(os.path.expanduser(output_dir_path))

    # Create output directory path
    # ! May raise:
    #   - PermissionError: when output directory cannot be created due to limited permissions over the output directory
    #   path
    logger.debug("Creating output directory if it does not exist.")
    os.makedirs(output_dir_path, mode=0o744, exist_ok=True)

    logger.debug("Checking whether a name for the output files is defined.")
    if not output_name:
        current_timestamp = datetime.datetime.today()
        logger.debug("Name for the output files was not defined.")
        logger.debug("Generating a name based on the current timestamp: {}.".format(
            datetime.datetime.strftime(current_timestamp, "%d-%m-%Y %H:%M:%S")
        ))
        output_name = "vectorize_{}{}{}{}{}{}{}".format(
            current_timestamp.year,
            current_timestamp.month,
            current_timestamp.day,
            current_timestamp.hour,
            current_timestamp.minute,
            current_timestamp.second,
            int(current_timestamp.microsecond / 1000)
        )
        logger.debug("Assigned name for output files: {}.".format(output_name))

    output_file_name = output_name + ".vectorized"
    logger.debug("Constructed vectorized corpus output file name: {}.".format(output_file_name))
    output_file_path = os.path.join(output_dir_path, output_file_name)
    logger.debug("Vectorized corpus output file will be saved in: {}.".format(output_file_path))
    output_dictionary_name = output_name + ".dictionary"
    logger.debug("Constructed inferred dictionary file name: {}.".format(output_dictionary_name))
    output_dictionary_path = os.path.join(output_dir_path, output_dictionary_name)
    logger.debug("Dictionary file will be saved in: {}.".format(output_dictionary_path))

    # Call the confirmation function for overwriting a file, if the output file already exists
    logger.debug("Ensuring that no files, under the paths \"{}\" and already exist.".format(
        "\", \"".join((output_file_path, output_dictionary_path))
    ))
    logger.debug("In case any of the files exist, a confirmation for overwritting the file is required.")
    if not file_overwrite_confirmation_function(output_file_path):
        return
    if not file_overwrite_confirmation_function(output_dictionary_path):
        return

    logger.debug("Executing vectorization {}. Reading from files \"{}\"".format(
        "tfidf" if tfidf else "bag-of-words",
        "\", \"".join(reader.file_paths)
    ))
    logger.debug(
        "Vocabulary to be created: MinDF: {}, MaxDFRatio: {}, MaxTerms: {}".format(least_documents,
                                                                                   max_document_ratio,
                                                                                   max_dictionary_size))
    logger.debug("Vocabulary will maintain these terms: {}".format(", ".join(keep_terms)))
    logger.debug("Result will be saved on \"{}\" and the inferred dictionary in \"{}\"".format(
        output_file_path, output_dictionary_path
    ))

    logger.info("Creating dictionary from preprocessed corpus files.")
    dictionary = Dictionary()
    for document in reader:
        dictionary.add_documents([document[1].split(" ")])
    logger.debug("Created dictionary.")
    logger.info("Reducing dictionary based on the given parameters")
    dictionary.filter_extremes(least_documents, max_document_ratio, max_dictionary_size, keep_terms)

    logger.info("Saving dictionary at \"{}\".".format(output_dictionary_path))
    dictionary.save(output_dictionary_path)

    if tfidf:
        tfidf_model = TfidfModel(dictionary=dictionary)
    with bz2.BZ2File(output_file_path, "wb") as bf_handle:
        pickler = pickle.Pickler(bf_handle, protocol=pickle.HIGHEST_PROTOCOL)
        documents_read = 0
        documents_vectorized = 0
        logger.info("Vectorizing preprocessed documents.")
        for document in reader:
            documents_read += 1
            if tfidf:
                vector = document[0], tfidf_model[dictionary.doc2bow(document[1].split(" "))], *document[2:]
            else:
                vector = document[0], dictionary.doc2bow(document[1].split(" ")), *document[2:]
            if len(vector[1]) > 0:
                documents_vectorized += 1
                pickler.dump(vector)
            else:
                logger.debug("Document \"{}\" has no resulting entries in the reduced dictionary".format(vector[0]))
            if documents_read % reporting_batch == 0:
                logger.info("Read {} preprocessed documents".format(documents_read))
        bf_handle.close()
        logger.info("Read {} preprocessed documents".format(documents_read))
    logger.info("Generated vectorized corpus, using {}, from {} preprocessed documents out of {} totally "
                "provided".format("tfidf" if tfidf else "bag-of-words", documents_vectorized, documents_read))
    logger.info("Vectorized corpus saved in \"{}\".".format(output_file_path))


if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser(description="Vectorization script")
    argument_parser.add_argument("dictionary_size", type=int, help="Maximum size of dictionary")
    argument_parser.add_argument("input_file_paths", nargs="+", help="One or more path(s) to the corpus files or "
                                                                     "configuration files for each reader")
    argument_parser.add_argument("output_dir_path", help="Path to a directory where the preprocessed dataset will be "
                                                         "saved")
    argument_parser.add_argument("-o", "--output-name", help="Name to be associated with the output files. If omitted "
                                                             "the name of the produced files is inferred as a "
                                                             "timestamp in the form of "
                                                             "\"[task_name]_[day]_[month]_[year]_[hour]_[minute]_"
                                                             "[second]_[millis]\" and refers to the time the process "
                                                             "of a given task is initiated")
    argument_parser.add_argument("--log-level", type=int, choices=(0, 1, 2), default=1,
                                 help="Logging level controls the verbosity of the module. Level 0 is a relatively "
                                      "silent execution that only produces error logs in the form of a file. This file "
                                      "is located under the \"logs\" directory in the given output directory, and is "
                                      "named after the given or inferred output name. Level 1 also yields informing "
                                      "messages in stdout. Level 2 produces debugging level messages in a file, "
                                      "located in the \"logs\" directory specified above with the name "
                                      "\"debug_[output_name]\", where output_name refers to the given or inferred "
                                      "output name. Default: 1")
    argument_parser.add_argument("-v", "--vectorization", type=lambda arg: arg.lower(), choices=("bow", "tfidf"),
                                 help="The vectorization method to be used. Can be either \"bow\" (for bag-of-words) "
                                      "or \"tfidf\". Default: \"bag-of-words\"")
    argument_parser.add_argument("--least-documents", type=int, help="Minimum number of documents a term must occur in "
                                                                     "order to get accepted in the resulting "
                                                                     "dictionary. Default: 1", default=1)
    argument_parser.add_argument("--max-document-ratio", type=float, help="A float representing the maximum ratio of "
                                                                          "documents to corpus size, in which an "
                                                                          "acceptable term occurs. Default: 1.0",
                                 default=1.0)
    argument_parser.add_argument("--keep-terms", nargs="+", default=[], help="A set of terms that should be "
                                                                             "in the resulting dictionary and not "
                                                                             "dropped")
    args_namespace = argument_parser.parse_args()
    functions.cli_print_license()

    max_dictionary_size = args_namespace.dictionary_size
    input_file_paths = args_namespace.input_file_paths
    output_dir_path = args_namespace.output_dir_path
    output_name = args_namespace.output_name
    if not output_name:
        current_timestamp = datetime.datetime.today()
        output_name = "vectorize_{}{}{}{}{}{}{}".format(
            current_timestamp.year,
            current_timestamp.month,
            current_timestamp.day,
            current_timestamp.hour,
            current_timestamp.minute,
            current_timestamp.second,
            int(current_timestamp.microsecond / 1000)
        )
    logger = functions.construct_logger(__name__, output_dir_path, output_name, args_namespace.log_level)
    file_overwrite_confirmation_function = functions.confirm_file_write
    use_tfidf = args_namespace.vectorization == "tfidf"
    least_documents = args_namespace.least_documents
    max_document_ratio = args_namespace.max_document_ratio
    keep_terms = args_namespace.keep_terms

    logger.debug("Attempting to create a BZ2BagReader for reading from preprocessed corpus files at \"{}\"".format(
        "\", \"".join(input_file_paths)
    ))
    reader = Bz2BagReader(*input_file_paths, logger=logger)

    try:
        vectorize_preprocessed(max_dictionary_size, reader, output_dir_path, tfidf=use_tfidf, output_name=output_name,
                               logger=logger, least_documents=least_documents, max_document_ratio=max_document_ratio,
                               keep_terms=keep_terms,
                               file_overwrite_confirmation_function=file_overwrite_confirmation_function)
    except Exception as ex:
        logger.critical(str(ex))
        exit(0)
