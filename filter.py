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

FILTERS_DICT = filters.available_filters
output_extensions = ("filtered",)


def filter_corpus(filter_ref, output_dir_path, *input_file_paths, output_name=None, logger=None,
                  workers=1, max_file_objects=-1, file_overwrite_confirmation_function=lambda *args: True):
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
        filter_configuration_string_segments = filter_ref.split()
        filter_name = filter_configuration_string_segments[0]
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
            filter_arguments = filter_configuration_string_segments[1:]
        logger.debug("Argument filter_ref successfully identified as a filter configuration string.")
        logger.debug("Attempting to create a filter using any defined arguments.")
        filter_obj = FILTERS_DICT[filter_name](*filter_arguments)
    else:
        logger.debug("Argument filter_ref successfully identified as a callable.")
        filter_obj = filter_ref

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
        output_name = "filter_{:04}{:02}{:02}{:02}{:02}{:02}{:03}".format(
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

    # Add a .filtered extension for the file that will hold the filtered corpus
    output_file_name_templates = (output_name_template + "." + extension for extension in output_extensions)

    logger.debug("Filtered corpus output files' name pattern: {}.".format(output_name_template).format("[NUMBER]"))

    logger.debug("Corpus output files will be saved in: {}.".format(output_dir_path))

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

    logger.debug("Creating multiprocessor.")
    multiprocessor = parallel.Multiprocessor(filter_obj.perform_filter, workers, output_dir_path, output_name,
                                             *output_extensions, logger=logger, buffer_size=100000,
                                             max_objects_per_output_file=max_file_objects)
    logger.debug("Initiating workers. Workers on stand-by.")
    multiprocessor.start()
    for record in reader_obj:
        multiprocessor.feed(record)
        documents_loaded += 1
        if documents_loaded % reporting_batch == 0:
            logger.info("{} documents passed to the filter".format(documents_loaded))
    logger.info("{} documents passed to the filter".format(documents_loaded))
    number_of_items_processed = multiprocessor.close()

    logger.info("Successfully filtered and accepted {} documents out of {} totally provided".format(
        number_of_items_processed[0],
        documents_loaded
    ))
    # logger.info("Filtered corpus was saved in \"{}\". File names are in the form \"{}\"".format(
    #     output_dir_path,
    #     "\", \"".join(output_file_form.format("[NUMBER]") for output_file_form in output_file_name_templates)
    # ))

    files_to_fetch_for_sample = 3
    filtered_files_sample = multiprocessor.joined_files_writers[0].created_files[:files_to_fetch_for_sample - 1]
    if len(multiprocessor.joined_files_writers[0].created_files) >= files_to_fetch_for_sample:
        if len(multiprocessor.joined_files_writers[0].created_files) > files_to_fetch_for_sample:
            filtered_files_sample.append("...")
        filtered_files_sample.append(multiprocessor.joined_files_writers[0].created_files[-1])

    logger.info("Filtered corpus was saved in:\n\t\"{}\"".format("\"\n\t\"".join(
        (os.path.join(output_dir_path, file_name) for file_name in filtered_files_sample)
    )))


if __name__ == "__main__":
    reporting_batch = 10000

    argument_parser = argparse.ArgumentParser(description="\n".join(docfetch.sanitize_docstring(__doc__)))
    # argument_parser.add_argument("property", choices=functions_dict.keys(), help="Target property to filter")
    argument_parser.add_argument("filter", help="A filter configuration corresponding to one of the available filters. "
                                                "For more information, regarding the available filters you can use the "
                                                "call \"python info.py list-filters\"")
    argument_parser.add_argument("input_file_paths", nargs="+", help="One or more path(s) to the .corpus or .filtered "
                                                                     "files")
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
    argument_parser.add_argument("-m", "--max-file-objects", type=int, default=-1,
                                 help="It controls the maximum amount of objects writen into a JSON output file. Any "
                                      "negative value or 0 means that the script will save all of the output in a "
                                      "single file. However this is discouraged for large datasets since certain file "
                                      "systems have limitations on the maximum size of a file")
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
    output_dir_path = args_namespace.output_dir_path
    input_file_paths = args_namespace.input_file_paths
    output_name = args_namespace.output_name
    if not output_name:
        current_timestamp = datetime.datetime.today()
        output_name = "filter_{:04}{:02}{:02}{:02}{:02}{:02}{:03}".format(
            current_timestamp.year,
            current_timestamp.month,
            current_timestamp.day,
            current_timestamp.hour,
            current_timestamp.minute,
            current_timestamp.second,
            int(current_timestamp.microsecond / 1000)
        )
    workers = args_namespace.workers

    max_file_objects = args_namespace.max_file_objects

    logger = functions.construct_logger(__name__, output_dir_path, output_name, args_namespace.log_level)

    file_overwrite_confirmation_function = functions.confirm_batch_file_write

    try:
        filter_corpus(filter_ref, output_dir_path, *input_file_paths, output_name=output_name,
                      logger=logger, max_file_objects=max_file_objects,
                      workers=workers, file_overwrite_confirmation_function=file_overwrite_confirmation_function)
    except Exception as ex:
        logger.critical(str(ex))
        exit(0)
