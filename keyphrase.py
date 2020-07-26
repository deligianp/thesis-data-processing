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
This module aims to produce keyphrases for the topics of a model.

To tackle the problem of interpretation and identification of a topic, this module provides a preliminary way of
identifying the keyphrases that are most commonly used throughout documents with a specific primary topic. The module
identifies the most prominent topic of each document as its primary topic and creates and inverted index based on the
primary topics, pointing to groups of documents. For each document in the group the top 10 keyphrases are recorded
using the PositionRank[1] method, as implemented by the pke package. Afterwards, the keyphrases for each document in
the group are aggregated using Borda count, in order to create a total ordering list of the most common keyphrases in
the group. Finally, the final list is pruned and only the top-10 keyphrases are kept.

Using the top-10 keyphrases list inferred in the step above, each topic is assigned as keyphrase the first available
keyphrase from its corresponding list. The availability of a keyphrase states that the keyphrase has not been assigned
to another topic.

To achieve the procedure above the module requires access to a text dump file as produced by the "dump.py", containing
the top topics for a corpus. This corpus is also required in this script in order to find the most commonly used
keyphrases. The final keyphrase assignments are saved in JSON-formatted file with a .keyphrases extension, mapping each
topic to an inferred keyphrase.

The module, as it serves only as a utility functionality it makes several assumptions:
    1. The model requires a file that distinguishes the top topics for each document in a corpus. This file must derive
    from the execution of the "dump.py" script, for the target model whose topics are to get the keyphrases and
    distinguish the topics for the document corpus which will be used in this script.
    2. The document corpus in this script is available in its initial format, meaning that it hasn't been subject to any
    alteration as preprocessing or vectorization.
    3. The module strongly assumes that the target model is well-trained model, meaning that document groups, indeed can
    describe distinguishable semantic aspects. If not, documents which discuss different real topics could be grouped
    together thus resulting in vague or imprecise keyphrase assignments.

[1] Corina Florescu and Cornelia Caragea. PositionRank: An unsupervised approach to keyphrase extraction from scholarly
documents. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics
(Volume1: Long Papers), pages 1105–1115, Vancouver, Canada, July 2017. Association for Computational Linguistics.
"""

import argparse
import datetime
import json
import logging
import os

import pke

from src.core.file import readers
from src.util import docfetch
from src.util import functions
from src.util import parallel

READERS_DICT = readers.available_readers


def _find_document_keyphrases(documents_dict, **kwargs):
    topic = documents_dict["topic"]
    documents = documents_dict["documents"]

    logger = kwargs.get("worker_logger", None)
    if not logger:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    keyphrases_vote_table = dict()
    for document in documents:
        text = document[1].replace(chr(0), '')
        if text == "nan" or text == "":
            continue
        pos = {'NOUN', 'PROPN', 'ADJ'}
        grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"

        extractor = pke.unsupervised.PositionRank()
        extractor.load_document(input=text, language='en', normalization=None)

        extractor.candidate_selection(grammar=grammar, maximum_word_number=3)
        extractor.candidate_weighting(window=2, pos=pos)
        k = 10
        # extracted_keyphrases = extractor.get_n_best(n=k)
        extracted_keyphrases = [keyphrase_result[0] for keyphrase_result in extractor.get_n_best(n=k)]
        for i in range(len(extracted_keyphrases)):
            extracted_keyphrase = extracted_keyphrases[i]
            borda_vote = k - i
            if extracted_keyphrase in keyphrases_vote_table:
                keyphrases_vote_table[extracted_keyphrase][0] += borda_vote
                keyphrases_vote_table[extracted_keyphrase][1] += 1
            else:
                keyphrases_vote_table[extracted_keyphrase] = [borda_vote, 1]
    keyphrases_list = [[key] + keyphrases_vote_table[key] for key in keyphrases_vote_table]
    topic_keyphrases = sorted(keyphrases_list, key=lambda x: x[1], reverse=True)[:10]
    logger.debug(
        "Keyphrases for topic {}: {}".format(topic, ", ".join(
            [k[0] for k in topic_keyphrases]
        )))
    return [{"topic": topic, "topic_keyphrases": topic_keyphrases}]


def infer_keyphrases(reader_ref, corpus_topics_file_path, output_dir_path, *input_file_paths,
                     output_name=None, logger=None, workers=1, top_n=100,
                     file_overwrite_confirmation_function=lambda path: True):
    # If given logger is None then use a logger with a null handler
    if not logger:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    logger.debug("Checking that number of workers and number of top documents are valid natural numbers.")
    logger.debug("Checking if number of workers is at least 1.")
    assert workers > 0, "Number of workers must be at least 1"
    logger.debug("Number of workers is at least 1.")
    logger.debug("Checking if number of workers is an integer.")
    assert type(workers) is int, "Number of workers must be an integer"
    logger.debug("Number of workers is an integer.")
    logger.debug("Checking if number of top documents is at least 1.")
    assert top_n > 0, "Number of top documents must be at least 1"
    logger.debug("Number of top documents is at least 1.")
    logger.debug("Checking if number of top documents is an integer.")
    assert type(top_n) is int, "Number of top documents must be an integer"
    logger.debug("Number of top documents is an integer.")

    # If reader_ref is not a readers.BaseReader subclass, treat it as the lowercase name of a reader class,
    # without the "Reader" ending
    logger.debug("Attempting to infer reader from reader_ref argument.")
    if not issubclass(type(reader_ref), readers.BaseReader):
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

    logger.debug("Attempting to get full absolute path of defined corpus topics file's path.")
    corpus_topics_file_path = os.path.abspath(os.path.expanduser(corpus_topics_file_path))

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
        output_name = "keyphrases_{}{}{}{}{}{}{}".format(
            current_timestamp.year,
            current_timestamp.month,
            current_timestamp.day,
            current_timestamp.hour,
            current_timestamp.minute,
            current_timestamp.second,
            int(current_timestamp.microsecond / 1000)
        )
        logger.debug("Assigned name for output files: {}.".format(output_name))

    output_file_name = output_name + ".keyphrases"
    logger.debug("Constructed keyphrases output file name: {}.".format(output_file_name))
    output_file_path = os.path.join(output_dir_path, output_file_name)
    logger.debug("Output file will be saved in: {}.".format(output_file_path))
    temporary_output_file_name = output_name + ".kp"
    logger.debug("Constructed temporary workers output file name: {}.".format(temporary_output_file_name))
    temporary_output_file_path = os.path.join(output_dir_path, temporary_output_file_name)
    logger.debug("Temporary workers' file will be saved in: {}.".format(temporary_output_file_path))

    logger.debug("Ensuring that no file, under the path \"{}\" already exists.".format(output_file_path))
    logger.debug("In case the file exists, a confirmation for overwritting the file is required.")
    if not file_overwrite_confirmation_function(output_file_path):
        logger.debug("The file exists and the confirmation was denied.")
        return

    logger.info("Loading corpus topics from file \"{}\".".format(corpus_topics_file_path))
    with open(corpus_topics_file_path, "r", encoding="utf-8") as f_handle:
        corpus_topics_dict = json.load(f_handle)

    topic_indexed_dictionary = dict()
    document_indexed_dictionary = dict()
    logger.debug("Constructing topic->list_of_documents map.")
    for document_topics in corpus_topics_dict:
        document_identifier = document_topics["document_identifier"]
        document_primary_topic = document_topics["top_topics"][0]["topic_index"]
        document_primary_topic_probability = document_topics["top_topics"][0]["probability"]
        if document_primary_topic not in topic_indexed_dictionary:
            topic_indexed_dictionary[document_primary_topic] = list()
        topic_indexed_dictionary[document_primary_topic].append(
            (document_identifier, document_primary_topic_probability))

        document_indexed_dictionary[document_identifier] = document_primary_topic

    logger.debug("Sorting and prunning topic indexed document lists.")
    logger.debug("Keeping only the top top_n most representative documents for each topic.")
    for topic_index in topic_indexed_dictionary:
        topic_indexed_dictionary[topic_index] = sorted(topic_indexed_dictionary[topic_index], key=lambda x: x[1],
                                                       reverse=True)[:top_n]

    logger.debug("Constructing a topic indexed dictionary containing the required documents for keyphrase analysis.")
    pending_documents = dict()
    for topic_index in topic_indexed_dictionary:
        pending_documents[topic_index] = set(doc[0] for doc in topic_indexed_dictionary[topic_index])

    logger.debug("Constructing document indexed dictionary containing the primary topics for each document.")
    loaded_documents = [list() for _ in range(len(pending_documents))]
    # for topic_index in topic_indexed_dictionary:
    #     loaded_documents = [[]] * len(pending_documents)

    logger.debug("Creating multiprocessor.")
    multiprocessor = parallel.Multiprocessor(_find_document_keyphrases, workers, output_dir_path,
                                             temporary_output_file_name,
                                             logger=logger, buffer_size=2 * workers)
    logger.debug("Initiating workers. Workers on stand-by.")
    multiprocessor.start()
    all_topics_keyphrases = dict()
    logger.info("Reading through corpus until the top_n documents of a topic have been collected.")
    logger.debug("Will then feed those documents to a worker which will perform the analysis.")
    topics_served = 0
    for record in reader_obj:
        # multiprocessor.feed(record)
        document_id = record[0]
        top_document = True
        try:
            read_document_primary_topic = document_indexed_dictionary[document_id]
            if document_id in pending_documents[read_document_primary_topic]:
                pending_documents[read_document_primary_topic].remove(document_id)
            else:
                top_document = False
        except KeyError:
            top_document = False
        if top_document:
            loaded_documents[read_document_primary_topic].append(record)
            if len(pending_documents[read_document_primary_topic]) == 0:
                logger.info(
                    "All documents for topic {} have been collected. Handing over the documents to a worker.".format(
                        read_document_primary_topic)
                )
                multiprocessor.feed(
                    {"topic": read_document_primary_topic, "documents": loaded_documents[read_document_primary_topic]}
                )
                logger.debug("Documents for topic {}:".format(read_document_primary_topic))
                for document in loaded_documents[read_document_primary_topic]:
                    logger.debug("\t{}".format(document[0]))
                topics_served += 1
                logger.info("Topics served: {}".format(topics_served))
                loaded_documents[read_document_primary_topic] = []
    multiprocessor.close()
    logger.info("Keyphrases analysis finished.")
    # REMOVE IT JUST CHECK
    for document_chunk in loaded_documents:
        if len(document_chunk) != 0:
            print("Topic {} still hasn't been served")

    logger.debug("Retrieving keyphrases analysis results")
    bf_handle = readers.Bz2BagReader(temporary_output_file_path, logger=logger)
    for topic_keyphrases in bf_handle:
        all_topics_keyphrases[topic_keyphrases["topic"]] = topic_keyphrases["topic_keyphrases"]
    #     if doc
    #         documents_loaded += 1
    #     if documents_loaded % reporting_batch == 0:
    #         logger.info("Loaded {} documents".format(documents_loaded))
    # logger.info("Loaded {} documents".format(documents_loaded))
    # logger.info("Joining workers' result to designated output file(s)")
    # number_of_items_processed = multiprocessor.close()
    assigned_keyphrases = set()
    assigned_topic_keyphrases = list()
    logger.info("Assigning keyphrases to topics.")
    for topic in all_topics_keyphrases:
        topic_keyphrase = ""
        topic_keyphrase_list = list()
        for i in range(len(all_topics_keyphrases[topic])):
            topic_keyphrase_list.append(all_topics_keyphrases[topic][i][0])
            constructed_keyphrase = ", ".join(topic_keyphrase_list)
            if constructed_keyphrase not in assigned_keyphrases:
                topic_keyphrase = constructed_keyphrase
                assigned_topic_keyphrases.append({"topic": topic, "keyphrase": constructed_keyphrase})
                assigned_keyphrases.add(constructed_keyphrase)
                break

    logger.info("Dumping assigned keyphrases to file \"{}\".".format(output_file_path))
    with open(output_file_path, "w", encoding="utf-8") as f_handle:
        json.dump(assigned_topic_keyphrases, f_handle, indent=4)

    logger.debug("Deleting temporary file \"{}\".".format(temporary_output_file_path))
    os.remove(temporary_output_file_path)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="\n".join(docfetch.sanitize_docstring(__doc__)))
    argument_parser.add_argument("reader", choices=tuple(READERS_DICT.keys()),
                                 help="An available dataset reader that should handle the dataset of each "
                                      "case. For more information regarding the available readers you can "
                                      "use the argument \"python info.py list-readers\". If the target corpus has been "
                                      "passed from one or more filtering steps, \"bz2bag\" reader should be used.")
    argument_parser.add_argument("input_file_paths", nargs="+", help="One or more path(s) to the corpus files or "
                                                                     "configuration files for each reader")
    argument_parser.add_argument("corpus_topics_dump_file_path", help="Path to a json file containing the corpus "
                                                                      "as inferred by the model for which the "
                                                                      "keyphrases will be produced.")
    argument_parser.add_argument("output_dir_path", help="Path to a directory where the preprocessed dataset will be "
                                                         "saved")
    argument_parser.add_argument("-n", "--top-n", type=int,
                                 help="Defines the amount of top documents per topic that will be analyzed in order to "
                                      "infer the keyphrases for each topic.", default=100)
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

    args_namespace = argument_parser.parse_args()
    functions.cli_print_license()

    reader_ref = args_namespace.reader
    output_dir_path = args_namespace.output_dir_path
    input_file_paths = args_namespace.input_file_paths
    corpus_topics_dump_file_path = args_namespace.corpus_topics_dump_file_path
    output_name = args_namespace.output_name
    top_n = args_namespace.top_n
    if not output_name:
        current_timestamp = datetime.datetime.today()
        output_name = "keyphrases_{}{}{}{}{}{}{}".format(
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
        infer_keyphrases(reader_ref, corpus_topics_dump_file_path, output_dir_path, *input_file_paths,
                         output_name=output_name, logger=logger, workers=workers, top_n=top_n,
                         file_overwrite_confirmation_function=file_overwrite_confirmation_function)
    except Exception as ex:
        logger.critical(str(ex))
        exit(0)
