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
import datetime
import logging
import os

import numpy as np
from gensim.models import LdaModel

from src.core.file import readers, writers
from src.util import functions

output_extensions = ("json",)


def dump_corpus_topics(model_path, output_dir_path, *vectorized_paths, output_name=None, logger=None, top_n=10,
                       file_overwrite_confirmation_function=lambda path: True, max_file_objects=-1):
    if not logger:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    output_dir_path = os.path.abspath(os.path.expanduser(output_dir_path))
    os.makedirs(output_dir_path, exist_ok=True)

    if not output_name:
        current_timestamp = datetime.datetime.today()
        output_name = "corpus_topics_{:04}{:02}{:02}{:02}{:02}{:02}{:03}".format(
            current_timestamp.year,
            current_timestamp.month,
            current_timestamp.day,
            current_timestamp.hour,
            current_timestamp.minute,
            current_timestamp.second,
            int(current_timestamp.microsecond / 1000)
        )
    output_name_template = output_name + ".part{}"

    output_file_name_templates = [output_name_template + "." + extension for extension in output_extensions]

    logger.debug("Corpus topics files' name pattern: {}.".format(output_name_template).format("[NUMBER]"))

    logger.debug("Output files will be saved in: {}.".format(output_dir_path))

    # Call the confirmation function for overwriting files, if similar files already exist
    logger.debug("Checking whether files with the same naming patterns already exist in \"{}\"".format(
        output_dir_path
    ))
    logger.debug("In case files exist, a confirmation for overwriting the files is required.")
    for extension in output_extensions:
        if not file_overwrite_confirmation_function(output_dir_path, (output_name, extension)):
            return

    reader = readers.JSONReader(*vectorized_paths)

    model_path = os.path.abspath(os.path.expanduser(model_path))
    model_ref = LdaModel.load(model_path)

    with writers.JSONWriter(output_name, output_directory=output_dir_path, max_file_objects_amount=max_file_objects,
                            output_files_extension=output_extensions[0]) as writer:
        for doc in reader:
            corpus_topics_obj = {
                "document_identifier": doc["id"],
                "top_topics": [
                    {
                        "topic_index": topic[0],
                        "probability": float(topic[1])
                    } for topic in
                    sorted(model_ref.get_document_topics(doc["content"], minimum_probability=0.0),
                           key=lambda val: val[1],
                           reverse=True)[:top_n]
                ]
            }
            writer.write_object(corpus_topics_obj)
        writer.close()

    files_to_fetch_for_sample = 3
    corpus_topics_files_sample = writer.created_files[:files_to_fetch_for_sample - 1]
    if len(writer.created_files) >= files_to_fetch_for_sample:
        if len(writer.created_files) > files_to_fetch_for_sample:
            corpus_topics_files_sample.append("...")
        corpus_topics_files_sample.append(writer.created_files[-1])

    logger.info("Corpus topics were saved in:\n\t\"{}\"".format("\"\n\t\"".join(
        (os.path.join(output_dir_path, file_name) for file_name in corpus_topics_files_sample)
    )))


def dump_model(model_path, output_dir_path, output_name=None, logger=None, top_n=10,
               file_overwrite_confirmation_function=lambda path: True, max_file_objects=-1):
    if not logger:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    output_dir_path = os.path.abspath(os.path.expanduser(output_dir_path))
    os.makedirs(output_dir_path, exist_ok=True)

    if not output_name:
        current_timestamp = datetime.datetime.today()
        output_name = "topic_terms_{:04}{:02}{:02}{:02}{:02}{:02}{:03}".format(
            current_timestamp.year,
            current_timestamp.month,
            current_timestamp.day,
            current_timestamp.hour,
            current_timestamp.minute,
            current_timestamp.second,
            int(current_timestamp.microsecond / 1000)
        )
    output_name_template = output_name + ".part{}"

    output_file_name_templates = [output_name_template + "." + extension for extension in output_extensions]

    logger.debug("Topics files' name pattern: {}.".format(output_name_template).format("[NUMBER]"))

    logger.debug("Output files will be saved in: {}.".format(output_dir_path))

    # Call the confirmation function for overwriting files, if similar files already exist
    logger.debug("Checking whether files with the same naming patterns already exist in \"{}\"".format(
        output_dir_path
    ))
    logger.debug("In case files exist, a confirmation for overwriting the files is required.")
    for extension in output_extensions:
        if not file_overwrite_confirmation_function(output_dir_path, (output_name, extension)):
            return

    model_path = os.path.abspath(os.path.expanduser(model_path))
    model_ref = LdaModel.load(model_path)
    topic_terms_matrix = model_ref.get_topics()
    top_n_topic_terms_matrix = np.flip(np.argsort(topic_terms_matrix, axis=1), axis=1)[:, :top_n]

    with writers.JSONWriter(output_name, output_directory=output_dir_path, max_file_objects_amount=max_file_objects,
                            output_files_extension=output_extensions[0]) as writer:
        for i in range(len(top_n_topic_terms_matrix)):
            topic_object = {
                "topic_index": i,
                "top_terms": [
                    {
                        "term": model_ref.id2word[j],
                        "probability": float(topic_terms_matrix[i, j])
                    } for j in top_n_topic_terms_matrix[i]
                ]
            }
            writer.write_object(topic_object)
        writer.close()

        files_to_fetch_for_sample = 3
        topics_files_sample = writer.created_files[:files_to_fetch_for_sample - 1]
        if len(writer.created_files) >= files_to_fetch_for_sample:
            if len(writer.created_files) > files_to_fetch_for_sample:
                topics_files_sample.append("...")
            topics_files_sample.append(writer.created_files[-1])

        logger.info("Topics were saved in:\n\t\"{}\"".format("\"\n\t\"".join(
            (os.path.join(output_dir_path, file_name) for file_name in topics_files_sample)
        )))


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="")
    # argument_parser.add_argument("action", choices={"model", "analysis"})
    subparsers = argument_parser.add_subparsers(help="Type of dumping method")
    corpus_parser = subparsers.add_parser("corpus",
                                          help="Option for dumping the top N topics for each document in a corpus. It "
                                               "requires one argument which is the path to the vectorized corpus.")
    corpus_parser.add_argument("vectorized_corpus_paths", help="Path to the vectorized corpus", nargs="+")
    model_parser = subparsers.add_parser("model",
                                         help="Option for dumping the top N terms and their probabilities for the given"
                                              "model")
    argument_parser.add_argument("input_model_path", help="Path to the model file to be dumped")
    argument_parser.add_argument("output_dir_path", help="Path to a directory where the dumped text file will be saved")
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
    argument_parser.add_argument("-m", "--max-file-objects", type=int, default=-1,
                                 help="It controls the maximum amount of objects writen into a JSON output file. Any "
                                      "negative value or 0 means that the script will save all of the output in a "
                                      "single file. However this is discouraged for large datasets since certain file "
                                      "systems have limitations on the maximum size of a file")
    argument_parser.add_argument("-n", "--top-n", help="This number restricts the amount of records written to the "
                                                       "dumped files. ", default=10, type=int)
    args_namespace = argument_parser.parse_args()
    functions.cli_print_license()

    try:
        vectorized_paths = args_namespace.vectorized_corpus_paths
    except AttributeError as ae:
        vectorized_paths = None
    dump_corpus = vectorized_paths is not None
    input_model_path = args_namespace.input_model_path
    output_dir_path = args_namespace.output_dir_path
    output_name = args_namespace.output_name
    if not output_name:
        current_timestamp = datetime.datetime.today()
        output_name = "{}_{:04}{:02}{:02}{:02}{:02}{:02}{:03}".format(
            "corpus_topics" if dump_corpus else "topic_terms",
            current_timestamp.year,
            current_timestamp.month,
            current_timestamp.day,
            current_timestamp.hour,
            current_timestamp.minute,
            current_timestamp.second,
            int(current_timestamp.microsecond / 1000)
        )
    logger = functions.construct_logger(__name__, output_dir_path, output_name, args_namespace.log_level)
    file_overwrite_confirmation_function = functions.confirm_batch_file_write
    max_file_objects = args_namespace.max_file_objects
    top_n = args_namespace.top_n
    if vectorized_paths:
        dump_corpus_topics(input_model_path, output_dir_path, *vectorized_paths, output_name=output_name, logger=logger,
                           top_n=top_n, file_overwrite_confirmation_function=file_overwrite_confirmation_function,
                           max_file_objects=max_file_objects)
    else:
        dump_model(input_model_path, output_dir_path, output_name=output_name, logger=logger, top_n=top_n,
                   file_overwrite_confirmation_function=file_overwrite_confirmation_function,
                   max_file_objects=max_file_objects)
