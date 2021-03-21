import argparse
import sys
import os
import datetime

from src.util.functions import construct_logger, confirm_batch_file_write
from src.core.file import readers, writers, split
from statistics import mode


def _parse_args(args):
    parser = argparse.ArgumentParser(description='description')
    subparsers = parser.add_subparsers(dest='method')
    cv_parser = subparsers.add_parser('crossvalidation')
    cv_parser.add_argument('folds', type=int)

    proportional_parser = subparsers.add_parser('proportional')
    proportional_parser.add_argument('proportion', type=float)

    parser.add_argument("input_file_paths", nargs="+", help="One or more path(s) to the .corpus or .filtered "
                                                            "files")
    parser.add_argument("output_dir_path", help="Path to a directory where the filtered dataset will be "
                                                "saved", )
    parser.add_argument("-o", "--output-name", help="Name to be associated with the output files. If "
                                                    "omitted the name of the produced files is inferred as a "
                                                    "timestamp in the form of "
                                                    "\"[task_name]_[day]_[month]_[year]_[hour]_[minute]_"
                                                    "[second]_[millis]\" and refers to the time the process "
                                                    "of a given task is initiated")
    parser.add_argument("--log-level", type=int, choices=(0, 1, 2), default=1,
                        help="Logging level controls the verbosity of the module. Level 0 is a relatively "
                             "silent execution that only produces error logs in the form of a file. This file "
                             "is located under the \"logs\" directory in the given output directory, and is "
                             "named after the given or inferred output name. Level 1 also yields informing "
                             "messages in stdout. Level 2 produces debugging level messages in a file, "
                             "located in the \"logs\" directory specified above with the name "
                             "\"debug_[output_name]\", where output_name refers to the given or inferred "
                             "output name. Default: 1")
    parser.add_argument("-m", "--max-file-objects", type=int, default=-1,
                        help="It controls the maximum amount of objects writen into a JSON output file. Any "
                             "negative value or 0 means that the script will save all of the output in a "
                             "single file. However this is discouraged for large datasets since certain file "
                             "systems have limitations on the maximum size of a file")

    argparse_namespace = parser.parse_args(args[1:])

    if argparse_namespace.method == 'crossvalidation' and not argparse_namespace.folds > 0:
        print('Cross-validation folds must be a positive, non-zero integer')
        return None

    if argparse_namespace.method == 'proportional' and not 0 < argparse_namespace.proportion <= 1:
        print('Split proportion must be a decimal number greater than 0 and less than or equal to 1')
        return None

    return argparse_namespace


def split_coprus_crossvalidation(*args, **kwargs):
    pass


def split_corpus_proportionally(proportion, output_dir_path, *input_file_paths, output_name=None, log_level=None,
                                max_file_objects=-1, logger=None,
                                file_overwrite_confirmation_function=lambda *args: True):
    if not logger:
        logger = construct_logger(__name__, output_dir_path, output_name, log_level)

    logger.debug("Checking that the proportion is a valid decimal number in the range (0,1]")
    assert 0 < proportion < 1, "Split proportion must be a decimal number greater than 0 and less than or equal to 1"

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
        output_name = "split_{:04}{:02}{:02}{:02}{:02}{:02}{:03}".format(
            current_timestamp.year,
            current_timestamp.month,
            current_timestamp.day,
            current_timestamp.hour,
            current_timestamp.minute,
            current_timestamp.second,
            int(current_timestamp.microsecond / 1000)
        )
        logger.debug("Assigned name for output files: {} for training set, {} for test set.".format(
            output_name + "-training", output_name + "-test")
        )

    # Detect the most common file extension among the files that were given
    # The splitted corpus will use the same file extension as the input files
    current_file_extension = mode([(os.path.basename(file).split('.')[-1]).lower() for file in input_file_paths])

    training_output_name = output_name + "-training"
    test_output_name = output_name + "-test"
    training_output_name_template = training_output_name + ".part{}"
    test_output_name_template = test_output_name + ".part{}"

    # Add the corresponding detected file extension on the output files
    training_output_file_name_templates = training_output_name_template + "." + current_file_extension
    test_output_file_name_templates = test_output_name_template + "." + current_file_extension

    logger.debug(
        "Training corpus files' name pattern: {}.".format(training_output_file_name_templates).format("[NUMBER]"))
    logger.debug(
        "Test corpus files' name pattern: {}.".format(test_output_file_name_templates).format("[NUMBER]"))

    logger.debug("Corpus output files will be saved in: {}.".format(output_dir_path))

    # Call the confirmation function for overwriting files, if similar files already exist
    logger.debug("Checking whether files with the same naming patterns already exist in \"{}\"".format(
        output_dir_path
    ))
    logger.debug("In case files exist, a confirmation for overwriting the files is required.")
    if not file_overwrite_confirmation_function(output_dir_path, (output_name, current_file_extension)):
        return
    print('hi')
    print(input_file_paths)
    reader_obj = readers.JSONReader(*input_file_paths, logger=logger)

    writer_objs = [
        writers.JSONWriter(training_output_name, output_dir_path, max_file_objects_amount=max_file_objects,
                           output_files_extension=current_file_extension),
        writers.JSONWriter(test_output_name, output_dir_path, max_file_objects_amount=max_file_objects,
                           output_files_extension=current_file_extension)
    ]

    split.proportional_split(reader_obj, writer_objs, proportion)

    for writer_obj in writer_objs:
        writer_obj.close()


if __name__ == '__main__':
    args = sys.argv
    parsed_namespace = _parse_args(args)
    if not parsed_namespace:
        exit(0)
    if parsed_namespace.method == 'proportional':
        split_corpus_proportionally(parsed_namespace.proportion, parsed_namespace.output_dir_path,
                                    *parsed_namespace.input_file_paths, output_name=parsed_namespace.output_name,
                                    log_level=parsed_namespace.log_level,
                                    max_file_objects=parsed_namespace.max_file_objects,
                                    file_overwrite_confirmation_function=confirm_batch_file_write)
