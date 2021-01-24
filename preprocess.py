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
"""
This module aims to provide a handle for applying preprocessing techniques on text corpora prior to topic modeling.

It a accepts a target corpus which it will to try to preprocess and prepare it for topic modelling. This corpus can be
either in its published format or have been passed through one or more filtering steps, by the "filter" module. A
non-filtered corpus requires a corresponding reader implementation that it can read the corpus and is defined either in
src.file.readers or application.readers. If a corpus has been filtered by the "filter" module, then the "bz2bag" reader
is appropriate.

The preprocessing module also allows for the use of customized preprocessing methods. Since preprocessing greatly
affects the performance of a produced models, it is imperative that different users can easily use different
preprocessing approaches that fit their needs. In order to define a new preprocessing method, one must define a
preprocessor class inside application.preprocessors module, that extends the class
src.core.train.preprocessors.BasePreprocessor.

The module can yield up to two files:
    * a file with the preprocessed corpus as a text of terms delimited by space (will always be produced)
    * a file which, for each preprocessed term records the initial term structure before preprocessing and the number of
    occurences
        e.g. measur->{
            measure - 4 times found,
            Measurement - 2 times found,
            measures - 5 times found
        }
    This file serves the purpose of providing a handle for retrieving a more readable representation of each
    preprocessed term.
"""

import argparse
import bz2
import datetime
import json
import logging
import os
import pickle

from src.core.file import readers as corpus_readers
from src.core.file import writers
from src.core.train import preprocessors
from src.util import docfetch
from src.util import functions
from src.util import parallel

reporting_batch = 10000
output_extensions = ("preprocessed", "nonstemmed")

PREPROCESSORS_DICT = preprocessors.available_preprocessors


def preprocess_corpus(preprocessor_ref, output_dir_path, *input_file_paths, output_name=None,
                      logger=None, workers=1, max_file_objects=-1,
                      file_overwrite_confirmation_function=lambda path: True):
    # If given logger is None then use a logger with a null handler
    if not logger:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    logger.debug("Checking that number of workers is a valid natural number.")
    logger.debug("Checking if number of workers is at least 1.")
    assert workers > 0, "Number of workers must be at least 1"
    logger.debug("Number of workers is at least 1.")
    logger.debug("Checking if number of workers is an integer.")
    assert type(workers) is int, "Number of workers must be an integer"
    logger.debug("Number of workers is an integer.")

    # If preprocessor_ref is not a function then treat it as preprocessor factory configuration list and attempt to
    # decode it
    logger.debug("Attempting to infer preprocessor from preprocessor_ref argument.")
    if not hasattr(preprocessor_ref, '__call__'):
        logger.debug("Argument preprocessor_ref was not a callable. Attempting to infer preprocessor by treating "
                     "preprocessor_ref as a preprocessor configuration string.")
        preprocessor_name = preprocessor_ref[0]
        if preprocessor_name not in PREPROCESSORS_DICT:
            logger.debug("Argument preprocessor_ref was neither a callable nor could it be broken down to a "
                         "preprocessor configuration string.")
            logger.debug("Raising error.")
            raise ValueError(
                "Preprocessor must be defined either in src.core.train.preprocessors or in application.preprocessors. "
                "Given preprocessor '{}' was not found".format(preprocessor_name)
            )
        # preprocessor_factory = PREPROCESSORS_DICT[preprocessor_name]()
        preprocessor_arguments = []
        if len(preprocessor_ref) > 0:
            preprocessor_arguments = preprocessor_ref[1:]
        logger.debug("Argument preprocessor_ref successfully identified as a preprocessor configuration string.")
        logger.debug("Attempting to create a preprocessor using any defined arguments.")
        preprocessor_obj = PREPROCESSORS_DICT[preprocessor_name](*preprocessor_arguments)
    else:
        logger.debug("Argument preprocessor_ref successfully identified as a callable.")
        preprocessor_obj = preprocessor_ref

    # Get output directory path
    logger.debug("Attempting to get full absolute path of defined output directory's path.")
    output_dir_path = os.path.abspath(os.path.expanduser(output_dir_path))

    # Create output directory path
    # ! May raise:
    #   - PermissionError: when output directory cannot be created due to limited permissions over the output directory
    #   path
    logger.debug("Creating output directory if it does not exist.")
    os.makedirs(output_dir_path, mode=0o744, exist_ok=True)

    # If no name was given for the produced files, create a name with a timestamp that describes the current time the
    # function was called
    logger.debug("Checking whether a name for the output files is defined.")
    if not output_name:
        current_timestamp = datetime.datetime.today()
        logger.debug("Name for the output files was not defined.")
        logger.debug("Generating a name based on the current timestamp: {}.".format(
            datetime.datetime.strftime(current_timestamp, "%d-%m-%Y %H:%M:%S")
        ))
        output_name = "preprocess_{:04}{:02}{:02}{:02}{:02}{:02}{:03}".format(
            current_timestamp.year,
            current_timestamp.month,
            current_timestamp.day,
            current_timestamp.hour,
            current_timestamp.minute,
            current_timestamp.second,
            int(current_timestamp.microsecond / 1000)
        )
        logger.debug("Assigned name for output files: {}.".format(output_name))

    output_name_template = output_name + ".part{}"

    output_file_name_templates = [output_name_template + "." + extension for extension in output_extensions[:1]]
    output_file_name_templates.append(output_name + "." + output_extensions[1])

    logger.debug("Preprocessed corpus files' name pattern: {}.".format(output_name_template).format("[NUMBER]"))
    logger.debug("Nonstemmed words map file pattern: {}".format(output_name + "." + output_extensions[1]))

    logger.debug("Output files will be saved in: {}.".format(output_dir_path))

    # Call the confirmation function for overwriting files, if similar files already exist
    logger.debug("Checking whether files with the same naming patterns already exist in \"{}\"".format(
        output_dir_path
    ))
    logger.debug("In case files exist, a confirmation for overwriting the files is required.")
    for extension in output_extensions:
        if not file_overwrite_confirmation_function(output_dir_path, (output_name, extension)):
            return

    reader_obj = corpus_readers.JSONReader(*input_file_paths, logger=logger)

    documents_loaded = 0

    # Initialize a multiprocessor that will execute preprocessing over the specified amount of processes
    logger.debug("Creating multiprocessor.")
    multiprocessor = parallel.Multiprocessor(preprocessor_obj.perform_preprocess, workers, output_dir_path,
                                             output_name, output_extensions[0], "nstemmed", logger=logger,
                                             buffer_size=10000,
                                             max_objects_per_output_file=max_file_objects)
    logger.debug("Initiating workers. Workers on stand-by.")
    multiprocessor.start()
    for record in reader_obj:
        multiprocessor.feed(record)
        documents_loaded += 1
        if documents_loaded % reporting_batch == 0:
            logger.info("{} documents passed to preprocessor".format(documents_loaded))
    logger.info("{} documents passed to preprocessor".format(documents_loaded))
    number_of_items_processed = multiprocessor.close()

    logger.info("Successfully preprocessed {} documents out of {} totally provided".format(number_of_items_processed[0],
                                                                                           documents_loaded))
    files_to_fetch_for_sample = 3
    preprocessed_files_sample = multiprocessor.joined_files_writers[0].created_files[:files_to_fetch_for_sample - 1]
    if len(multiprocessor.joined_files_writers[0].created_files) >= files_to_fetch_for_sample:
        if len(multiprocessor.joined_files_writers[0].created_files) > files_to_fetch_for_sample:
            preprocessed_files_sample.append("...")
        preprocessed_files_sample.append(multiprocessor.joined_files_writers[0].created_files[-1])

    logger.info("Preprocessed corpus was saved in:\n\t\"{}\"".format("\"\n\t\"".join(
        (os.path.join(output_dir_path, file_name) for file_name in preprocessed_files_sample)
    )))

    accumulated_non_stemmmed_mappings = dict()
    nstemmed_files_reader = corpus_readers.JSONReader(*multiprocessor.joined_files_writers[1].created_files)
    for stemmed_mappings in nstemmed_files_reader:
        for stemmed_term in stemmed_mappings:
            if stemmed_term not in accumulated_non_stemmmed_mappings:
                accumulated_non_stemmmed_mappings[stemmed_term] = dict()
            for nonstemmed_term in stemmed_mappings[stemmed_term]:
                if nonstemmed_term in accumulated_non_stemmmed_mappings[stemmed_term]:
                    accumulated_non_stemmmed_mappings[stemmed_term][nonstemmed_term] += stemmed_mappings[stemmed_term][
                        nonstemmed_term]
                else:
                    accumulated_non_stemmmed_mappings[stemmed_term][nonstemmed_term] = stemmed_mappings[stemmed_term][
                        nonstemmed_term]

    nonstemmed_term_writer = writers.JSONWriter(output_name, output_directory=output_dir_path,
                                                output_files_extension="nonstemmed", max_file_objects_amount=-1)
    for stemmed_term in accumulated_non_stemmmed_mappings:
        words_statistics = accumulated_non_stemmmed_mappings[stemmed_term]
        nonstemmed_mapping_object = {
            "stemmed": stemmed_term,
            "original_word_statistics": tuple(sorted([
                (word, words_statistics[word]) for word in words_statistics
            ], key=lambda x: x[1], reverse=True))
        }
        nonstemmed_term_writer.write_object(nonstemmed_mapping_object)
    nonstemmed_term_writer.close()
    logger.info("Non-stemmed statistics map was saved in \"{}\".".format(os.path.join(
        output_dir_path, output_file_name_templates[1]
    )))

    for file in multiprocessor.joined_files_writers[1].created_files:
        os.remove(file)


if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser(description="\n".join(docfetch.sanitize_docstring(__doc__)))
    argument_parser.add_argument("input_file_paths", nargs="+", help="One or more path(s) to .corpus or .filtered "
                                                                     "files")
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
    argument_parser.add_argument("-w", "--workers", type=int, help="Number of processes to be used for the task. "
                                                                   "(NOTE: While an increased number of processes will "
                                                                   "speed up the task, it is advised not to define the "
                                                                   "number of processes to more or equal to the amount "
                                                                   "of available processors, as this leads to time "
                                                                   "costs due to Inter Process Communications). "
                                                                   "Default: 1", default=1)
    argument_parser.add_argument("-p", "--preprocessor", nargs="+", default=(tuple(PREPROCESSORS_DICT.keys())[0],),
                                 help="The preprocessor which will be for the given corpus. If not the script will use "
                                      "a default preprocessor defined in src.core.train.preprocessors. For more "
                                      "information about preprocessors use \"python info.py list-preprocessors\".")
    argument_parser.add_argument("-m", "--max-file-objects", type=int, default=-1,
                                 help="It controls the maximum amount of objects writen into a JSON output file. Any "
                                      "negative value or 0 means that the script will save all of the output in a "
                                      "single file. However this is discouraged for large datasets since certain file "
                                      "systems have limitations on the maximum size of a file")

    args_namespace = argument_parser.parse_args()
    functions.cli_print_license()

    max_file_objects = args_namespace.max_file_objects
    output_dir_path = args_namespace.output_dir_path
    input_file_paths = args_namespace.input_file_paths
    output_name = args_namespace.output_name
    if not output_name:
        current_timestamp = datetime.datetime.today()
        output_name = "preprocess_{:04}{:02}{:02}{:02}{:02}{:02}{:03}".format(
            current_timestamp.year,
            current_timestamp.month,
            current_timestamp.day,
            current_timestamp.hour,
            current_timestamp.minute,
            current_timestamp.second,
            int(current_timestamp.microsecond / 1000)
        )
    workers = args_namespace.workers

    logger = functions.construct_logger(__name__, output_dir_path, output_name, args_namespace.log_level)

    file_overwrite_confirmation_function = functions.confirm_file_write

    preprocessor_ref = args_namespace.preprocessor

    try:
        preprocess_corpus(preprocessor_ref, output_dir_path, *input_file_paths, output_name=output_name,
                          logger=logger, workers=workers, max_file_objects=max_file_objects,
                          file_overwrite_confirmation_function=functions.confirm_batch_file_write)
    except Exception as ex:
        logger.critical(str(ex))
        exit(0)
