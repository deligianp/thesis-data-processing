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
This module can be used in order to normalize a corpus to a text-based JSON file format.

The module can use the package-provided readers, as well any other readers defined inside application.readers
module. The readers are used to retrieve the corpus documents from their sources.

The module saves the corpus to a set of output JSON files which can in turn be used from the rest of the scripts for a
topic modeling task
"""
import argparse
import datetime
import logging
import os

from src.core.file import readers, writers
from src.util import docfetch
from src.util import functions

reporting_batch = 10000
output_extensions = ("corpus",)


def normalize_corpus(reader_ref, output_dir_path, *input_file_paths, output_name=None, logger=None,
                     file_overwrite_confirmation_function=lambda *args: True, max_file_objects=-1):
    # If given logger is None then use a logger with a null handler
    if not logger:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    # If reader_ref is not a corpus_readers.BaseReader subclass, treat it as the lowercase name of a reader class,
    # without the "Reader" ending
    logger.debug("Attempting to infer reader from reader_ref argument.")
    if not issubclass(type(reader_ref), readers.BaseReader):
        logger.debug("Argument reader_ref was not a valid readers.BaseReader extension. Attempting to infer reader by "
                     "treating reader_ref as a readers.BaseReader extension class name.")
        # Initialize reader
        if reader_ref not in readers.available_readers:
            logger.debug("Argument reader_ref was neither a readers.BaseReader extension instance nor had a "
                         "readers.BaseReader extension class name.")
            logger.debug("Raising error.")
            raise ValueError(
                "Reader must be defined either in src.core.file.readers or in application.readers. Given reader '{}' "
                "was not found.".format(reader_ref)
            )
        else:
            logger.debug("Argument reader_ref \"{}\"successfully identified as a readers.BaseReader extension class. "
                         "name.".format(reader_ref))
            logger.debug("Attempting to get full absolute paths of defined input files' paths.")
            input_file_paths = (os.path.abspath(os.path.expanduser(input_file_path)) for input_file_path in
                                input_file_paths)
            logger.debug("Initializing an object of the defined reader class, based on the produced input_file_paths.")
            reader_obj = readers.available_readers[str(reader_ref)](*input_file_paths, logger=logger)
    else:
        logger.debug("Argument reader_ref successfully identified as readers.BaseReader extension instance.")
        reader_obj = reader_ref

    # Get output directory path
    logger.debug("Attempting to get full absolute path of defined output directory's path.")
    output_dir_path = os.path.abspath(os.path.expanduser(output_dir_path))

    # Create output directory path
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
        output_name = "corpus_{:04}{:02}{:02}{:02}{:02}{:02}{:03}".format(
            current_timestamp.year,
            current_timestamp.month,
            current_timestamp.day,
            current_timestamp.hour,
            current_timestamp.minute,
            current_timestamp.second,
            int(current_timestamp.microsecond / 1000)
        )
        logger.debug("Assigned name for output files: {}.".format(output_name))

    # Output name template
    output_name_template = output_name + ".part{}"

    logger.debug("Corpus output files' name pattern: {}.".format(output_name_template).format("[NUMBER]"))
    logger.debug("Corpus output files will be saved in: {}.".format(output_dir_path))

    # Call the confirmation function for potentially overwriting files, if similar output files already exist
    logger.debug("Checking whether files with the same naming patterns already exist in \"{}\".".format(
        output_dir_path
    ))
    logger.debug("In case files exist, a confirmation for overwriting the files is required.")
    for extension in output_extensions:
        if not file_overwrite_confirmation_function(output_dir_path, (output_name, extension)):
            return

    documents_loaded = 0

    # Creating a JSONWriter for storing/normalizing the corpus
    logger.debug("Initializing JSONWriter for storing corpus in \"{}\".".format(output_dir_path))
    with writers.JSONWriter(output_name, output_directory=output_dir_path, max_file_objects_amount=max_file_objects,
                            output_files_extension=output_extensions[0], buffer_size=100000) as corpus_writer:
        logger.debug("Loading corpus.")
        # Reading through initial corpus source
        for document in reader_obj:
            corpus_writer.write_object(document)
            documents_loaded += 1

            # In case [reporting_batch] documents have been read, notify the user
            # (it is used to control the informational output of the function)
            if documents_loaded % reporting_batch == 0:
                logger.info("Loaded {} documents.".format(documents_loaded))
        # Inform the user for the last number of documents read
        if documents_loaded % reporting_batch != 0:
            logger.info("Loaded {} documents.".format(documents_loaded))

    logger.info("Successfully loaded {} documents.".format(documents_loaded))

    # Print a sample of the produced output files
    # If up to 3 files were created, then all of them are printed
    # If more than 3 files were created, then the first 2 and last file are printed, along with a string (...), between
    # the 2nd and the last file name, to indicate that the number of files created is greater than the number of files
    # being displayed
    files_to_fetch_for_sample = 3
    corpus_files_sample = corpus_writer.created_files[:files_to_fetch_for_sample - 1]
    if len(corpus_writer.created_files) >= files_to_fetch_for_sample:
        if len(corpus_writer.created_files) > files_to_fetch_for_sample:
            corpus_files_sample.append("...")
        corpus_files_sample.append(corpus_writer.created_files[-1])

    logger.info("Corpus was saved in:\n\t\"{}\"".format("\"\n\t\".".join(
        (os.path.join(output_dir_path, file_name) for file_name in corpus_files_sample)
    )))
    logger.info("Corpus normalization completed!")


if __name__ == "__main__":
    # Argument parser when corpus_normalize is ran as a script
    argument_parser = argparse.ArgumentParser(description="\n".join(docfetch.sanitize_docstring(__doc__)))
    argument_parser.add_argument("reader", help="An available dataset reader that should handle the dataset of each "
                                                "case. For more information regarding the available readers you can "
                                                "use the argument \"python info.py list-readers\"",
                                 choices=tuple(readers.available_readers.keys()),
                                 default=tuple(readers.available_readers.keys())[0])
    argument_parser.add_argument("input_file_paths", nargs="+", help="One or more path(s) to the corpus files or "
                                                                     "configuration files for the reader")
    argument_parser.add_argument("output_dir_path", help="Path to a directory where the filtered dataset will be "
                                                         "saved", )
    argument_parser.add_argument("-o", "--output-name", help="Name to be associated with the output files. If "
                                                             "omitted the name of the produced files is inferred as a "
                                                             "timestamp in the form of "
                                                             "\"[task_name]_[day]_[month]_[year]_[hour]_[minute]_"
                                                             "[second]_[millis]\" and refers to the time the script "
                                                             "execution was initiated")
    argument_parser.add_argument("-m", "--max-file-objects", type=int, default=-1,
                                 help="It controls the maximum amount of objects writen into a JSON output file. Any "
                                      "negative value or 0 means that the script will save all of the output in a "
                                      "single file. However this is discouraged for large datasets since certain file "
                                      "systems have limitations on the maximum size of a file")
    argument_parser.add_argument("--log-level", type=int, choices=(0, 1, 2), default=1,
                                 help="Logging level controls the verbosity of the module. Level 0 is a relatively "
                                      "silent execution that only produces error logs in the form of a file. This file "
                                      "is located under the \"logs\" directory in the given output directory, and is "
                                      "named after the given or inferred output name. Level 1 also yields informing "
                                      "messages in stdout. Level 2 produces debugging level messages in a file, "
                                      "located in the \"logs\" directory specified above with the name "
                                      "\"debug_[output_name]\", where output_name refers to the given or inferred "
                                      "output name. Default: 1")

    args_namespace = argument_parser.parse_args()

    # Printing license
    functions.cli_print_license()

    # Retrieving parsed arguments
    reader_ref_script = args_namespace.reader
    output_dir_path_script = args_namespace.output_dir_path
    input_file_paths_script = args_namespace.input_file_paths
    output_name_script = args_namespace.output_name
    max_file_objects_script = args_namespace.max_file_objects

    # If no output name was given, use a timestamp of the current time
    if not output_name_script:
        current_timestamp_script = datetime.datetime.today()
        output_name_script_script = "corpus_{:04}{:02}{:02}{:02}{:02}{:02}{:03}".format(
            current_timestamp_script.year,
            current_timestamp_script.month,
            current_timestamp_script.day,
            current_timestamp_script.hour,
            current_timestamp_script.minute,
            current_timestamp_script.second,
            int(current_timestamp_script.microsecond / 1000)
        )

    # Constructing logger
    logger_script = functions.construct_logger(
        __name__, output_dir_path_script, output_name_script, args_namespace.log_level
    )

    # Defining function that will be used for checking and prompting whether to overwrite existing files
    # For script execution, use a CLI function
    file_overwrite_confirmation_function_script = functions.confirm_batch_file_write

    # Calling corpus normalization function
    try:
        normalize_corpus(
            reader_ref_script, output_dir_path_script,
            *input_file_paths_script,
            output_name=output_name_script,
            logger=logger_script, max_file_objects=max_file_objects_script,
            file_overwrite_confirmation_function=file_overwrite_confirmation_function_script
        )
    except Exception as ex:
        logger_script.critical(str(ex))
        exit(0)
