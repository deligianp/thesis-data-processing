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
from src.core.train import preprocessors
from src.util import docfetch
from src.util import functions
from src.util import parallel

reporting_batch = 10000

READERS_DICT = corpus_readers.available_readers
PREPROCESSORS_DICT = preprocessors.available_preprocessors


def preprocess_corpus(preprocessor_ref, reader_ref, output_dir_path, *input_file_paths, output_name=None,
                      logger=None, workers=1, file_overwrite_confirmation_function=lambda path: True):
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
        preprocessor_factory = PREPROCESSORS_DICT[preprocessor_name]()
        preprocessor_arguments = []
        if len(preprocessor_ref) > 0:
            preprocessor_arguments = preprocessor_ref[1:]
        logger.debug("Argument preprocessor_ref successfully identified as a preprocessor configuration string.")
        logger.debug("Attempting to create a preprocessor using any defined arguments.")
        preprocessor_function = preprocessor_factory.create_preprocessor(*preprocessor_arguments)
    else:
        logger.debug("Argument preprocessor_ref successfully identified as a callable.")
        preprocessor_function = preprocessor_ref

    # If reader_ref is not a corpus_readers.BaseReader subclass, treat it as the lowercase name of a reader class,
    # without the "Reader" ending
    logger.debug("Attempting to infer reader from reader_ref argument.")
    if not issubclass(type(reader_ref), corpus_readers.BaseReader):
        logger.debug(
            "Argument reader_ref was not a valid readers.BaseReader extension. Attempting to infer reader by"
            "treating reader_ref as a readers.BaseReader extension class name.")
        # Initialize reader
        if reader_ref not in READERS_DICT:
            logger.debug("Argument reader_ref was neither a readers.BaseReader extension instance nor a "
                         "readers.BaseReader extension class name.")
            logger.debug("Raising error.")
            raise ValueError(
                "Reader must be defined either in src.core.file.readers or in application.readers. Given reader '{}' "
                "was not found".format(reader_ref)
            )
        else:
            logger.debug(
                "Argument reader_ref successfully identified as a readers.BaseReader extension class name.")
            logger.debug("Attempting to get full absolute paths of defined input files' paths.")
            input_file_paths = (os.path.abspath(os.path.expanduser(input_file_path)) for input_file_path in
                                input_file_paths)
            logger.debug(
                "Initializing an object of the defined reader class, based on the produced input_file_paths.")
            reader_obj = READERS_DICT[str(reader_ref)](*input_file_paths, logger=logger)
    else:
        logger.debug("Argument reader_ref successfully identified as readers.BaseReader extension instance.")
        reader_obj = reader_ref

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
        output_name = "preprocess_{}{}{}{}{}{}{}".format(
            current_timestamp.year,
            current_timestamp.month,
            current_timestamp.day,
            current_timestamp.hour,
            current_timestamp.minute,
            current_timestamp.second,
            int(current_timestamp.microsecond / 1000)
        )
        logger.debug("Assigned name for output files: {}.".format(output_name))

    # Add a .preprocessed extension for the file that will hold the preprocessed corpus and a .nonstemmed extension that
    # will hold the term aliases before stemming. .ns is an intermediate file where nostemmed aliases are accumulated
    # from all the workers
    output_file_name = output_name + ".preprocessed"
    logger.debug("Constructed preprocessed corpus output file name: {}.".format(output_file_name))
    output_file_path = os.path.join(output_dir_path, output_file_name)
    logger.debug("Preprocessed corpus output file will be saved in: {}.".format(output_file_path))
    nonstemmed_file_name = "." + output_name + ".ns"
    logger.debug("Constructed temporary nonstemmed terms output file name: {}.".format(nonstemmed_file_name))
    nonstemmed_file_path = os.path.join(output_dir_path, nonstemmed_file_name)
    logger.debug("Temporary nonstemmed terms file will be saved in: {}.".format(nonstemmed_file_path))
    nonstemmed_file_name_combined = output_name + ".nonstemmed"
    logger.debug("Constructed nonstemmed terms output file name: {}.".format(nonstemmed_file_name_combined))
    nonstemmed_combined_file_path = os.path.join(output_dir_path, nonstemmed_file_name_combined)
    logger.debug("Nonstemmed terms output file will be saved in: {}.".format(nonstemmed_combined_file_path))

    # Call the confirmation function for overwriting a file, if the output file already exists
    logger.debug("Ensuring that no files, under the paths \"{}\" and already exist.".format(
        "\", \"".join((output_file_path, nonstemmed_combined_file_path))
    ))
    logger.debug("In case any of the files exist, a confirmation for overwritting the file is required.")
    if not file_overwrite_confirmation_function(output_file_path):
        return
    if not file_overwrite_confirmation_function(nonstemmed_file_path):
        return

    documents_loaded = 0

    # Initialize a multiprocessor that will execute preprocessing over the specified amount of processes
    logger.debug("Creating multiprocessor.")
    multiprocessor = parallel.Multiprocessor(preprocessor_function, workers, output_dir_path, output_file_name,
                                             nonstemmed_file_name, logger=logger, buffer_size=10000)
    logger.debug("Initiating workers. Workers on stand-by.")
    multiprocessor.start()
    for record in reader_obj:
        multiprocessor.feed(record)
        documents_loaded += 1
        if documents_loaded % reporting_batch == 0:
            logger.info("Attempted to preprocess {} documents".format(documents_loaded))
    logger.info("Attempted to preprocess {} documents".format(documents_loaded))
    number_of_items_processed = multiprocessor.close()

    logger.info("Successfully preprocessed {} documents out of {} totally provided".format(number_of_items_processed[0],
                                                                                           documents_loaded))
    logger.info("Preprocessed corpus was saved in \"{}\"".format(output_file_path))
    if len(number_of_items_processed) > 1 and number_of_items_processed[1] != 0:
        nonstemmed_fh = bz2.BZ2File(nonstemmed_file_path, "rb")
        nonstemmed_unpickler = pickle.Unpickler(nonstemmed_fh)

        combined_dictionary = dict()
        while True:
            try:
                document_nonstemmed = json.loads(nonstemmed_unpickler.load())
            except EOFError:
                break

            for stemmed_key in document_nonstemmed:
                if stemmed_key not in combined_dictionary:
                    combined_dictionary[stemmed_key] = dict()
                for nonstemmed in document_nonstemmed[stemmed_key]:
                    if nonstemmed in combined_dictionary[stemmed_key]:
                        combined_dictionary[stemmed_key][nonstemmed] += 1
                    else:
                        combined_dictionary[stemmed_key][nonstemmed] = 1

        nonstemmed_fh.close()
        nonstemmed_combined_fh = bz2.BZ2File(nonstemmed_combined_file_path, "wb")
        nonstemmed_pickler = pickle.Pickler(nonstemmed_combined_fh, protocol=pickle.HIGHEST_PROTOCOL)
        nonstemmed_pickler.dump(combined_dictionary)
        nonstemmed_combined_fh.close()
        os.remove(nonstemmed_file_path)
        logger.info("Mapped nonstemmed term aliases for {} documents to file \"{}\"".format(
            number_of_items_processed[1],
            nonstemmed_combined_file_path
        ))


if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser(description="\n".join(docfetch.sanitize_docstring(__doc__)))
    argument_parser.add_argument("reader", choices=tuple(READERS_DICT.keys()),
                                 help="An available dataset reader that should handle the dataset of each "
                                      "case. For more information regarding the available readers you can "
                                      "use the argument \"python info.py list-readers\". If the target corpus has been "
                                      "passed from one or more filtering steps, \"bz2bag\" reader should be used.")
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

    args_namespace = argument_parser.parse_args()
    functions.cli_print_license()

    reader_ref = args_namespace.reader
    output_dir_path = args_namespace.output_dir_path
    input_file_paths = args_namespace.input_file_paths
    output_name = args_namespace.output_name
    if not output_name:
        current_timestamp = datetime.datetime.today()
        output_name = "preprocess_{}{}{}{}{}{}{}".format(
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
        preprocess_corpus(preprocessor_ref, reader_ref, output_dir_path, *input_file_paths, output_name=output_name,
                          logger=logger, workers=workers,
                          file_overwrite_confirmation_function=file_overwrite_confirmation_function)
    except Exception as ex:
        logger.critical(str(ex))
        exit(0)
