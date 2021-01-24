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
This module supports the main functionality of training and producing an LDA topic model

The module accepts a vectorized corpus and a dictionary, in the form of files that are produced by the respective
vectorize module. It then creates an LDA model, supplied with the given dictionary and trained over the given vectorized
corpus.
"""
import argparse
import datetime
import os

from gensim.corpora import Dictionary
from gensim.models import LdaModel, LdaMulticore

from src.core.file import readers
from src.util import docfetch
from src.util import functions

reporting_batch = 10000
output_extensions = ("lda", "lda.id2word", "lda.expElogbeta.npy", "lda.state")


def train_model(number_of_topics, dictionary_path, output_dir_path, *input_file_paths, output_name=None,
                logger=None, workers=1, chunk_size=2000, passes=1, iterations=50,
                file_overwrite_confirmation_function=lambda path: True):
    assert number_of_topics > 1, "Number of topics must be at least 2"
    assert type(number_of_topics) is int, "Number of topics must be an integer"
    assert workers > 0, "At least 1 worker must be used"
    assert type(workers) is int, "Number of workers must be an integer"
    assert chunk_size > 0, "Size of chunks must be at least 1"
    assert type(chunk_size) is int, "Size of chunks must be an integer"
    assert passes > 0, "Number of passes must be at least 1"
    assert type(passes) is int, "Number of passes must be an integer"
    assert iterations > 0, "Number of iterations must be at least 1"
    assert type(iterations) is int, "Number of iterations must be an integer"

    reader = readers.JSONReader(*input_file_paths)

    dictionary_path = os.path.abspath(os.path.expanduser(dictionary_path))
    dictionary = Dictionary.load(dictionary_path)

    output_dir_path = os.path.abspath(os.path.expanduser(output_dir_path))
    os.makedirs(output_dir_path, exist_ok=True)

    if not output_name:
        current_timestamp = datetime.datetime.today()
        output_name = "lda_model_{:04}{:02}{:02}{:02}{:02}{:02}{:03}".format(
            current_timestamp.year,
            current_timestamp.month,
            current_timestamp.day,
            current_timestamp.hour,
            current_timestamp.minute,
            current_timestamp.second,
            int(current_timestamp.microsecond / 1000)
        )

    output_file_name_templates = [output_name + "." + extension for extension in output_extensions]

    logger.debug("Trained model' name pattern: {}.".format(output_name))

    logger.debug("Output files will be saved in: {}.".format(output_dir_path))

    # Call the confirmation function for overwriting files, if similar files already exist
    logger.debug("Checking whether files with the same naming patterns already exist in \"{}\"".format(
        output_dir_path
    ))
    logger.debug("In case files exist, a confirmation for overwriting the files is required.")
    for extension in output_extensions:
        if not file_overwrite_confirmation_function(output_dir_path, (output_name, extension)):
            return

    logger.debug(
        "Executing training of an LDA model of {} topics. Reading from files \"{}\" and dictionary at \"{}\"".format(
            number_of_topics, "\", \"".join(reader.file_paths), dictionary_path
        )
    )
    logger.debug("Model will be updated every {} documents".format(chunk_size))
    # logger.debug("LDA model will be saved in \"{}\"".format(output_file_path))

    if workers == 1:
        model = LdaModel(num_topics=number_of_topics, id2word=dictionary, passes=passes, iterations=iterations)
    else:
        model = LdaMulticore(num_topics=number_of_topics, id2word=dictionary, workers=workers, passes=passes,
                             iterations=iterations)

    batch_buffer = list()
    vectors_seen = 0
    for document_vector in reader:
        batch_buffer.append(document_vector)
        vectors_seen += 1
        if vectors_seen % reporting_batch == 0:
            logger.info("Read {} vectors".format(vectors_seen))
        if len(batch_buffer) >= chunk_size:
            model.update([vector["content"] for vector in batch_buffer])
            batch_buffer = list()
    if len(batch_buffer) >= 0:
        logger.info("Read {} vectors".format(vectors_seen))
        model.update([vector["content"] for vector in batch_buffer])
        batch_buffer = list()

    model.save(os.path.join(output_dir_path, output_name + "." + output_extensions[0]))

    logger.info("Completed training of LDA model of {} topics and saved it in \"{}\"".format(
        number_of_topics, output_dir_path
    ))

    logger.info("Trained model was saved in:\n\t\"{}\"".format("\"\n\t\"".join(
        (os.path.join(output_dir_path, file_name) for file_name in output_file_name_templates)
    )))


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="\n".join(docfetch.sanitize_docstring(__doc__)))
    argument_parser.add_argument("number_of_topics", type=int, help="Number of topics to infer")
    argument_parser.add_argument("input_file_paths", nargs="+", help="One or more path(s) to the corpus files or "
                                                                     "configuration files for each reader")
    argument_parser.add_argument("dictionary_path", help="Path to the inferred dictionary to be used for training")
    argument_parser.add_argument("output_dir_path", help="Path to a directory where the preprocessed dataset will be "
                                                         "saved")
    argument_parser.add_argument("-o", "--output-name", help="Name to be associated with the produced LDA files. If "
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
    argument_parser.add_argument("-c", "--chunk-size", type=int, help="This refers to the size of chunks of vectorized "
                                                                      "documents that will be used to update the model "
                                                                      "each time. A lower number of documents results "
                                                                      "in a smaller memory footprint. Default: 2000",
                                 default=2000)
    argument_parser.add_argument("-p", "--passes", type=int, help="Times any batch of vectorized corpus will be fed to "
                                                                  "the corpus. A higher number helps the model to "
                                                                  "potentially fit well over the vectorized corpus, "
                                                                  "but greatly affects the time needed for training. "
                                                                  "Default: 1", default=1)
    argument_parser.add_argument("-i", "--iterations", type=int, help="Maximum amount of iterations of E-step of the "
                                                                      "variational Bayes estimator used to fit the "
                                                                      "model. A higher number helps the model to "
                                                                      "potentially fit well over the vectorized "
                                                                      "corpus, but greatly affects the time needed for "
                                                                      "training. Default: 50", default=50)
    args_namespace = argument_parser.parse_args()
    functions.cli_print_license()

    number_of_topics = args_namespace.number_of_topics
    input_file_paths = args_namespace.input_file_paths
    dictionary_path = args_namespace.dictionary_path
    output_dir_path = args_namespace.output_dir_path
    output_name = args_namespace.output_name
    if not output_name:
        current_timestamp = datetime.datetime.today()
        output_name = "lda_model_{:04}{:02}{:02}{:02}{:02}{:02}{:03}".format(
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
    workers = args_namespace.workers
    chunk_size = args_namespace.chunk_size
    passes = args_namespace.passes
    iterations = args_namespace.iterations

    train_model(number_of_topics, dictionary_path, output_dir_path, *input_file_paths, output_name=output_name,
                logger=logger,
                workers=workers, chunk_size=chunk_size, passes=passes, iterations=iterations,
                file_overwrite_confirmation_function=file_overwrite_confirmation_function)
