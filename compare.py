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
""""""

import argparse
import datetime
import logging
import logging.handlers
import os

import numpy as np
from gensim import models

from src.core.evaluate import metrics
from src.util import functions


# This function is designed to run as the root function of each new process
def _execute_metric_calculation(arguments_dictionary, **kwargs):
    logger = kwargs["worker_logger"]
    if not logger:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    metric_function = arguments_dictionary["target_metric"]
    topic_terms_0_path = arguments_dictionary["m0_array_path"]
    topic_terms_1_path = arguments_dictionary["m1_array_path"]
    temporary_dictionary_0_path = arguments_dictionary["m0_dictionary_path"]
    temporary_dictionary_1_path = arguments_dictionary["m1_dictionary_path"]
    topn = arguments_dictionary["topn"]
    output_dir_path = arguments_dictionary["output_dir_path"]
    metric_result = metric_function(topic_terms_0_path, temporary_dictionary_0_path, topic_terms_1_path,
                                    temporary_dictionary_1_path, topn=topn, logger=logger)

    worker_id = kwargs["worker_id"]
    result_file_name = os.path.join(output_dir_path, f".{worker_id}-result")
    np.save(result_file_name, metric_result)

    return [
        {
            "path": result_file_name
        }
    ]


def _divide_to_jobs(array, workers):
    total_tasks = len(array)
    ideal_job_size = int(total_tasks / workers)
    over_tasked_workers = total_tasks % workers
    previous_job_index = 0
    jobs = []
    for w_index in range(workers - 1):
        jobs.append(previous_job_index + (ideal_job_size + 1 if over_tasked_workers > 0 else ideal_job_size))
        previous_job_index = jobs[-1]
        over_tasked_workers -= 1
    if len(jobs) > 0:
        distributed_arrays = np.split(array, jobs, 0)
        return distributed_arrays
    else:
        return [array]


def compare_models(metric_ref, model0, model1, output_dir_path, output_name=None, logger=None, workers=1, topn=50,
                   model0_bounds=None, model1_bounds=None, file_overwrite_confirmation_function=lambda path: True):
    if not logger:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    logger.debug("Checking that number of workers is a valid natural number.")
    logger.debug("Checking if number of workers is at least 1.")
    assert workers > 0, "At least 1 worker must be used"
    logger.debug("Number of workers is at least 1.")
    logger.debug("Checking if number of workers is an integer.")
    assert type(workers) is int, "Number of workers must be an integer"
    logger.debug("Number of workers is an integer.")
    logger.debug("Checking if number of top terms is at least 1.")
    assert topn > 0, "Number of top terms must be at least 1"
    logger.debug("Number of top terms is at least 1.")
    logger.debug("Checking if number of top terms is an integer.")
    assert type(workers) is int, "Number of top terms must be an integer"
    logger.debug("Number of top terms is an integer.")

    if model0_bounds:
        logger.debug("Checking that given bounds for model 0 are valid.")
        logger.debug("Checking if given bounds for model 0 is a tuple of size 2.")
        assert len(model0_bounds) > 0, "Bounds tuple must be of size 2: (STARTING_TOPIC_INDEX, TOPICS_TO_COMPARE)"
        logger.debug("Given bounds for model 0 is a tuple of size 2.")
        logger.debug("Checking if given bounds for model 0 are integers.")
        assert type(model0_bounds[0]) is int and type(model0_bounds[1]) is int, "Bounds tuple must contain integers"
        logger.debug("Given bounds for model 0 are integers.")

    if model1_bounds:
        logger.debug("Checking that given bounds for model 1 are valid.")
        logger.debug("Checking if given bounds for model 1 is a tuple of size 2.")
        assert len(model1_bounds) > 0, "Bounds tuple must be of size 2: (STARTING_TOPIC_INDEX, TOPICS_TO_COMPARE)"
        logger.debug("Given bounds for model 1 is a tuple of size .2")
        logger.debug("Checking if given bounds for model 1 are integers.")
        assert type(model1_bounds[0]) is int and type(model1_bounds[1]) is int, "Bounds tuple must contain integers"
        logger.debug("Given bounds for model 1 are integers.")

    logger.debug("Attempting to infer metric function from metric_ref argument.")
    if not hasattr(metric_ref, "__call__"):
        logger.debug("Argument metric_ref was not a callable. Attempting to infer metric by treating metric_ref as a "
                     "metric name.")
        metric = metrics.registered_metrics[metric_ref]
        metric_name = metric_ref
        logger.debug("Argument metric_ref successfully identified as a metric name.")
    else:
        logger.debug("Argument metric_ref is a function. Treating it like a metric.")
        metric = metric_ref
        metric_name = "compare"

    logger.debug("Checking if argument model0 is a model.LdaModel subclass")
    if not issubclass(type(model0), models.LdaModel):
        logger.debug(
            "Argument model0 is not a model.LdaModel subclass. Treating it as a path to a model.LdaModel file.")
        model0 = models.LdaModel.load(os.path.abspath(os.path.expanduser(model0)))

    logger.debug("Checking if argument model1 is a model.LdaModel subclass")
    if not issubclass(type(model1), models.LdaModel):
        logger.debug(
            "Argument model1 is not a model.LdaModel subclass. Treating it as a path to a model.LdaModel file.")
        model1 = models.LdaModel.load(os.path.abspath(os.path.expanduser(model1)))

    # Get output directory path
    logger.debug("Attempting to get full absolute path of defined output directory's path.")
    output_dir_path = os.path.abspath(os.path.expanduser(output_dir_path))

    logger.debug("Creating output directory if it does not exist.")
    os.makedirs(output_dir_path, mode=0o744, exist_ok=True)

    logger.debug("Checking whether a name for the output files is defined.")
    if not output_name:
        current_timestamp = datetime.datetime.today()
        logger.debug("Name for the output files was not defined.")
        logger.debug("Generating a name based on the current timestamp: {}.".format(
            datetime.datetime.strftime(current_timestamp, "%d-%m-%Y %H:%M:%S")
        ))
        output_name = "{}_{:04}{:02}{:02}{:02}{:02}{:02}{:03}".format(
            metric_name,
            current_timestamp.year,
            current_timestamp.month,
            current_timestamp.day,
            current_timestamp.hour,
            current_timestamp.minute,
            current_timestamp.second,
            int(current_timestamp.microsecond / 1000)
        )
        logger.debug("Assigned name for output files: {}.".format(output_name))

    output_file_name = output_name + ".comparison"
    logger.debug("Constructed comparison matrix output file name: {}.".format(output_file_name))
    output_file_path = os.path.join(output_dir_path, output_file_name)
    logger.debug("Comparison matrix output file will be saved in: {}.".format(output_file_path))

    temporary_workers_result_output_file = output_name + ".comp"
    logger.debug(
        "Constructed temporary workers' comparison output file name: {}.".format(temporary_workers_result_output_file)
    )
    temporary_workers_result_output_path = os.path.join(output_dir_path, temporary_workers_result_output_file)
    logger.debug(
        "Temporary workers' comparison output file will be saved in: {}.".format(temporary_workers_result_output_path)
    )

    logger.debug("Ensuring that no files, under the paths \"{}\" and already exist.".format(
        "\", \"".join((output_file_path, temporary_workers_result_output_path))
    ))
    logger.debug("In case any of the files exist, a confirmation for overwriting the file is required.")
    if not file_overwrite_confirmation_function(output_file_path) or not file_overwrite_confirmation_function(
            temporary_workers_result_output_path):
        return

    # Reduce model 1 topics
    logger.debug("Fetching models' topic-tokens matrices")
    model0_topics = model0.get_topics()
    model1_topics = model1.get_topics()
    model0_dictionary = model0.id2word
    model1_dictionary = model1.id2word

    logger.debug("Fetching models' dictionaries")
    temporary_dictionary_0_name = ".0.{}.dictionary".format(output_name)
    temporary_dictionary_0_path = os.path.join(output_dir_path, temporary_dictionary_0_name)
    model0_dictionary.save(temporary_dictionary_0_path)
    temporary_dictionary_1_name = ".1.{}.dictionary".format(output_name)
    temporary_dictionary_1_path = os.path.join(output_dir_path, temporary_dictionary_1_name)
    model1_dictionary.save(temporary_dictionary_1_path)

    if model0_bounds:
        logger.debug("Reducing first model's topics. Initial shape: {}".format(model0_topics.shape))
        model0_topics = model0_topics[model0_bounds[0]:sum(model0_bounds)]
        logger.debug("Reduced first model's shape: {}".format(model0_topics.shape))
    if model1_bounds:
        logger.debug("Reducing second model's topics. Initial shape: {}".format(model1_topics.shape))
        model1_topics = model1_topics[model1_bounds[0]:sum(model1_bounds)]
        logger.debug("Reduced second model's shape: {}".format(model1_topics.shape))

    logger.debug("Dereferencing loaded models")
    del model0
    del model1

    m0_path = os.path.join(output_dir_path, "m0_matrix.npy")
    np.save(m0_path, model0_topics)
    m1_path = os.path.join(output_dir_path, "m1_matrix.npy")
    np.save(m1_path, model1_topics)

    m = metric(m0_path, temporary_dictionary_0_path, m1_path, temporary_dictionary_1_path, topn=top_n, logger=logger)

    np.save(output_file_path, m)
    # # Create jobs if multiple workers were given
    # logger.debug("Dividing calculation task to {} workers".format(workers))
    # distributed_jobs = _divide_to_jobs(model0_topics, workers)
    # logger.debug("Number of tasks for each worker: {}".format([len(worker_job) for worker_job in distributed_jobs]))
    #
    # logger.debug("Generating temporary files paths to store first model's split matrix")
    # subarray_file_names = [".0_{}_{}.npy".format(i, output_name) for i in
    #                        range(len(distributed_jobs))]
    # subarray_file_paths = [os.path.join(output_dir_path, subarray_file_name) for subarray_file_name in
    #                        subarray_file_names]
    # # subarray_paths = [os.path.join(output_directory_path, ".{}_{}.npy".format(i, os.path.basename(output_file_path)))
    # #                   for i in range(len(distributed_jobs))]
    # logger.debug("Temporary files for first model's split matrix: " + ", ".join(subarray_file_paths))
    # logger.debug("Generating temporary file path for second model's matrix")
    # m1_array_file_name = ".1_0_{}.npy".format(output_name)
    # m1_array_file_path = os.path.join(output_dir_path, m1_array_file_name)
    # logger.debug("Temporary file for second model's matrix: " + m1_array_file_path)
    #
    # logger.debug("Preparing distributed tasks for calculation")
    # for i in range(len(distributed_jobs)):
    #     np.save(subarray_file_paths[i], distributed_jobs[i])
    # np.save(m1_array_file_path, model1_topics)
    #
    # logger.debug("Saving dictionary for first model")
    # model0_dictionary.save(temporary_dictionary_0_path)
    # logger.debug("Saving dictionary for second model")
    # model1_dictionary.save(temporary_dictionary_1_path)
    #
    # logger.debug("Dereferencing split matrices of first model")
    # for i in range(len(distributed_jobs), 0, -1):
    #     del distributed_jobs[i - 1]
    # logger.debug("Dereferencing matrix of first model")
    # del model0_topics
    # logger.debug("Dereferencing matrix of secondary model")
    # del model1_topics
    #
    # logger.debug("Dereferencing first model's dictionary")
    # del model0_dictionary
    # logger.debug("Dereferencing second model's dictionary")
    # del model1_dictionary
    #
    # logger.debug("Creating subprocess arguments")
    # execution_arguments = [
    #     {
    #         "target_metric": metric,
    #         "m0_array_path": subarray_path,
    #         "m1_array_path": m1_array_file_path,
    #         "m0_dictionary_path": temporary_dictionary_0_path,
    #         "m1_dictionary_path": temporary_dictionary_1_path,
    #         "topn": top_n,
    #         "output_dir_path": output_dir_path
    #     } for subarray_path in subarray_file_paths
    # ]
    #
    # logger.debug("Creating multiprocessor.")
    # multiprocessor = parallel.Multiprocessor(_execute_metric_calculation, workers, output_dir_path,
    #                                          output_name, "comp", logger=logger, buffer_size=10000)
    # logger.debug("Initiating workers. Workers on stand-by.")
    # multiprocessor.start()
    # for args_dict in execution_arguments:
    #     multiprocessor.feed(args_dict)
    # number_of_items_processed = multiprocessor.close()
    #
    # file_paths_reader = readers.JSONReader(*multiprocessor.joined_files_writers[0].created_files)
    # paths = list()
    # for path_obj in file_paths_reader:
    #     paths.append(path_obj["path"])
    # file_paths_reader.close()
    #
    # np_subbarrays = list()
    # for path in paths:
    #     np_subbarrays(np.load(path))
    # logger.debug("Joining results")
    # total_result = None
    # for result in results:
    #     if total_result is not None:
    #         total_result = np.concatenate((total_result, result), axis=0)
    #     else:
    #         total_result = result
    #
    # logger.debug("Retrieved a total matrix of shape: {}".format(total_result.shape))
    #
    # logger.debug("Deleting temporary files")
    # for temp_file_path in subarray_file_paths:
    #     os.remove(temp_file_path)
    # os.remove(m1_array_file_path)
    # os.remove(temporary_dictionary_0_path)
    # os.remove(temporary_dictionary_1_path)
    # os.remove(temporary_workers_result_output_path)
    #
    # logger.info("Result of calculation was saved in \"{}\"".format(output_file_path))
    # with bz2.BZ2File(output_file_path, "wb") as fhandle:
    #     pickler = pickle.Pickler(fhandle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickler.dump(total_result)


if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser(description="This script is used to compare two LDA models based on their"
                                                          " different topic-terms distribution. The result is a binary "
                                                          "file named as the concatenation of the two LDA model names "
                                                          "and contains a NumPy MxN with the distances of the M topics "
                                                          "of the first model from the N topics of the second model")
    argument_parser.add_argument("metric", choices=tuple(metrics.registered_metrics),
                                 help="The metric which will be used to compare the two models. For more information "
                                      "regarding the available metrics you can run \"python info.py list-metrics\".")
    argument_parser.add_argument("model0_path", help="Path to the first LDA model")
    argument_parser.add_argument("model1_path", help="Path to the second LDA model")
    argument_parser.add_argument("output_dir_path", help="Path to a directory where the comparison result will be "
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

    argument_parser.add_argument("-N", help="Value of the top N terms to be considered for each topic. Used only by a "
                                            "subset of metrics that take into account only the top-N terms of each "
                                            "topic (see above). Default: 1.", type=int, default=1)
    argument_parser.add_argument("-m0", "--model0-bounds", nargs=2, type=int,
                                 help="Correct syntax: -m0 [STARTING_TOPIC_INDEX] [TOPICS_TO_COMPARE]. It allows to "
                                      "define a subset of topics of model 0 which will be compared. The first argument "
                                      "defines the lowest topic index of the subset while the second argument defines "
                                      "the amount of topics that are part of the subset and begin from the topic index"
                                      "specified.")
    argument_parser.add_argument("-m1", "--model1-bounds", nargs=2, type=int,
                                 help="Correct syntax: -m1 [STARTING_TOPIC_INDEX] [TOPICS_TO_COMPARE]. It allows to "
                                      "define a subset of topics of model 1 which will be compared. The first argument "
                                      "defines the lowest topic index of the subset while the second argument defines "
                                      "the amount of topics that are part of the subset and begin from the topic index"
                                      "specified.")
    args_namespace = argument_parser.parse_args()
    functions.cli_print_license()
    model0 = args_namespace.model0_path
    model1 = args_namespace.model1_path
    m0_bounds = args_namespace.model0_bounds
    m1_bounds = args_namespace.model1_bounds
    target_metric = args_namespace.metric
    output_dir_path = args_namespace.output_dir_path
    output_name = args_namespace.output_name
    if not output_name:
        current_timestamp = datetime.datetime.today()
        output_name = "{}_{:04}{:02}{:02}{:02}{:02}{:02}{:03}".format(
            target_metric,
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
    top_n = args_namespace.N

    try:
        compare_models(target_metric, model0, model1, output_dir_path, output_name=output_name, logger=logger,
                       workers=workers, topn=top_n, model0_bounds=m0_bounds, model1_bounds=m1_bounds,
                       file_overwrite_confirmation_function=functions.confirm_file_write)
    except Exception as ex:
        logger.critical(str(ex))
