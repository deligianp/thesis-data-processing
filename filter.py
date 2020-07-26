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
This module can be used in order to filter out texts of a corpus that don't match certain filtering conditions.

The module can use the package-provided readers, as well any other readers defined inside application.readers
module. Any reader focuses on managing different types of sources and formats of data.

The filters being used by the module can be any of package-provided filters, as well as any other filter defined inside
application.filters module. Different filters focus on different text metadata and attributes in order to decide whether
the text should be accepted
"""
import argparse
import datetime
import logging
import os

from src.core.file import readers as corpus_readers
from src.core.train import filters
from src.util import docfetch
from src.util import functions
from src.util import parallel

READERS_DICT = corpus_readers.available_readers
FILTERS_DICT = filters.available_filters


def filter_corpus(filter_ref, reader_ref, output_dir_path, *input_file_paths, output_name=None, logger=None,
                  workers=1, file_overwrite_confirmation_function=lambda path: True):
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

    # If filter_ref is not a function then treat it as filter factory configuration list and attempt to decode it
    logger.debug("Attempting to infer filter from filter_ref argument.")
    if not hasattr(filter_ref, '__call__'):
        logger.debug("Argument filterer_ref was not a callable. Attempting to infer filter by treating filter_ref as a "
                     "filter configuration string.")
        filter_name = filter_ref[0]
        if filter_name not in FILTERS_DICT:
            logger.debug("Argument filter_ref was neither a callable nor could it be broken down to a filter "
                         "configuration string.")
            logger.debug("Raising error.")
            raise ValueError(
                "Filter must be defined either in src.core.train.filters or in application.filters. Given filter '{}' "
                "was not found".format(filter_name)
            )
        # filter_factory = FILTERS_DICT[filter_name]()
        filter_arguments = []
        if len(filter_ref) > 0:
            filter_arguments = filter_ref[1:]
        logger.debug("Argument filter_ref successfully identified as a filter configuration string.")
        logger.debug("Attempting to create a filter using any defined arguments.")
        filter_obj = FILTERS_DICT[filter_name](*filter_arguments)
    else:
        logger.debug("Argument filter_ref successfully identified as a callable.")
        filter_obj = filter_ref

    # If reader_ref is not a corpus_readers.BaseReader subclass, treat it as the lowercase name of a reader class,
    # without the "Reader" ending
    logger.debug("Attempting to infer reader from reader_ref argument.")
    if not issubclass(type(reader_ref), corpus_readers.BaseReader):
        logger.debug("Argument reader_ref was not a valid readers.BaseReader extension. Attempting to infer reader by"
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
            logger.debug("Argument reader_ref successfully identified as a readers.BaseReader extension class name.")
            logger.debug("Attempting to get full absolute paths of defined input files' paths.")
            input_file_paths = (os.path.abspath(os.path.expanduser(input_file_path)) for input_file_path in
                                input_file_paths)
            logger.debug("Initializing an object of the defined reader class, based on the produced input_file_paths.")
            reader_obj = READERS_DICT[str(reader_ref)](*input_file_paths, logger=logger)
    else:
        logger.debug("Argument reader_ref successfully identified as readers.BaseReader extension instance.")
        reader_obj = reader_ref

    # Get output directory path
    logger.debug("Attempting to get full absolute path of defined output directory's path.")
    output_dir_path = os.path.abspath(os.path.expanduser(output_dir_path))

    # Create output directory path
    # ! May raise:
    #   - PermissionError: when output directory cannot be created due to limited permissions over the
    #   output directory path
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
        output_name = "filter_{}{}{}{}{}{}{}".format(
            current_timestamp.year,
            current_timestamp.month,
            current_timestamp.day,
            current_timestamp.hour,
            current_timestamp.minute,
            current_timestamp.second,
            int(current_timestamp.microsecond / 1000)
        )
        logger.debug("Assigned name for output files: {}.".format(output_name))

    # Add a .filtered extension for the file that will hold the filtered corpus
    output_file_name = output_name + ".filtered"
    logger.debug("Constructed filtered corpus output file name: {}.".format(output_file_name))

    # Create the output file path
    output_file_path = os.path.join(output_dir_path, output_file_name)
    logger.debug("Filtered corpus output file will be saved in: {}.".format(output_file_path))

    # Call the confirmation function for overwriting a file, if the output file already exists
    logger.debug("Ensuring that no file, under the path \"{}\" already exists.".format(output_file_path))
    logger.debug("In case the file exists, a confirmation for overwritting the file is required.")
    if not file_overwrite_confirmation_function(output_file_path):
        return

    documents_loaded = 0

    logger.debug("Creating multiprocessor.")
    multiprocessor = parallel.Multiprocessor(filter_obj.perform_filter, workers, output_dir_path, output_file_name,
                                    logger=logger, buffer_size=10000)
    logger.debug("Initiating workers. Workers on stand-by.")
    multiprocessor.start()
    for record in reader_obj:
        multiprocessor.feed(record)
        documents_loaded += 1
        if documents_loaded % reporting_batch == 0:
            logger.info("Attempted to filter {} documents".format(documents_loaded))
    logger.info("Attempted to filter {} documents".format(documents_loaded))
    number_of_items_processed = multiprocessor.close()

    logger.info("Successfully filtered and accepted {} documents out of {} totally provided".format(
        number_of_items_processed[0],
        documents_loaded
    ))
    logger.info("Filtered corpus was saved in \"{}\"".format(output_file_path))


if __name__ == "__main__":
    reporting_batch = 10000

    argument_parser = argparse.ArgumentParser(description="\n".join(docfetch.sanitize_docstring(__doc__)))
    # argument_parser.add_argument("property", choices=functions_dict.keys(), help="Target property to filter")
    argument_parser.add_argument("filter", help="A filter configuration corresponding to one of the available filters. "
                                                "For more information, regarding the available filters you can use the "
                                                "call \"python info.py list-filters\"", nargs="+")
    argument_parser.add_argument("reader", help="An available dataset reader that should handle the dataset of each "
                                                "case. For more information regarding the available readers you can "
                                                "use the argument \"python info.py list-readers\"",
                                 choices=tuple(READERS_DICT.keys()),
                                 default=tuple(READERS_DICT.keys())[0])
    argument_parser.add_argument("input_file_paths", nargs="+", help="One or more path(s) to the corpus files or "
                                                                     "configuration files for each reader")
    argument_parser.add_argument("output_dir_path", help="Path to a directory where the filtered dataset will be "
                                                         "saved", )
    argument_parser.add_argument("-o", "--output-name", help="Name to be associated with the output files. If "
                                                             "omitted the name of the produced files is inferred as a "
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

    args_namespace = argument_parser.parse_args()
    functions.cli_print_license()

    filter_ref = args_namespace.filter
    reader_ref = args_namespace.reader
    output_dir_path = args_namespace.output_dir_path
    input_file_paths = args_namespace.input_file_paths
    output_name = args_namespace.output_name
    if not output_name:
        current_timestamp = datetime.datetime.today()
        output_name = "filter_{}{}{}{}{}{}{}".format(
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

    try:
        filter_corpus(filter_ref, reader_ref, output_dir_path, *input_file_paths, output_name=output_name,
                      logger=logger,
                      workers=workers, file_overwrite_confirmation_function=file_overwrite_confirmation_function)
    except Exception as ex:
        logger.critical(str(ex))
        exit(0)
