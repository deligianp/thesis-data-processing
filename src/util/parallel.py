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
import logging
import logging.handlers
import multiprocessing
import os
import time

from src.core.file import writers, readers


def _process_function(target_function, worker_id, input_queue, output_queue, logging_queue, output_path,
                      number_of_output_files, buffer_size=10000, worker_kwargs=None):
    worker_time = str(time.time())
    worker_output_writers = [
        writers.JSONWriter("{}-{}-{}".format(worker_id, i, worker_time),
                           output_directory=output_path,
                           output_files_extension="temp",
                           buffer_size=buffer_size) for i in range(number_of_output_files)
    ]
    worker_logger = logging.getLogger("worker-{}".format(worker_id))
    worker_logger.setLevel(logging.DEBUG)
    if logging_queue is None:
        worker_logger.addHandler(logging.NullHandler())
    else:
        worker_logger.addHandler(logging.handlers.QueueHandler(logging_queue))
    sentinel_captured = False
    if not worker_kwargs:
        worker_kwargs = dict()
    worker_kwargs["worker_id"] = worker_id
    worker_kwargs["worker_logger"] = worker_logger
    while not sentinel_captured:
        input_value = input_queue.get()
        if input_value is not None:
            try:
                output_value = target_function(input_value, **worker_kwargs)
            except Exception as ex:
                worker_logger.error(str(ex))
                continue
            if output_value is not None:
                for i in range(min(len(worker_output_writers), len(output_value))):
                    result = output_value[i]
                    worker_output_writers[i].write_object(result)
        else:
            sentinel_captured = True
    for writer in worker_output_writers:
        writer.close()
    output_queue.put([writer.created_files for writer in worker_output_writers])


class Multiprocessor:

    def __init__(self, process_function, workers, output_path, output_file_name,
                 *output_file_extensions, kwargs_dictionary=None, logger=None, buffer_size=-1,
                 max_objects_per_output_file=None):
        if kwargs_dictionary is None:
            kwargs_dictionary = dict()
        self.output_file_name = output_file_name
        self.output_file_extensions = output_file_extensions
        self.process_function = process_function
        self.workers = workers
        self.output_path = os.path.abspath(os.path.expanduser(output_path))
        self.max_objects_per_output_file = max_objects_per_output_file
        self.kwargs_dictionary = kwargs_dictionary if kwargs_dictionary is not None else dict()

        self.logger = logger
        self.joined_files_writers = None

        self.buffer_size = buffer_size

        self._state = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def feed(self, input_value):
        if self._state is not None:
            if input_value is not None:
                self._state["input_queue"].put(input_value)
                # self._state["rotating_index"] = (self._state["rotating_index"] + 1) % self._state["num_of_workers"]
            else:
                raise ValueError("Input value cannot be None")
        else:
            raise multiprocessing.ProcessError("Multiprocessor has not yet forked")

    def start(self):
        if self._state is None:
            self._state = dict()
            self._state["num_of_workers"] = self.workers
            self._state["manager"] = multiprocessing.Manager()  # for the manager
            if self.logger is not None:
                self._state["logging_queue"] = self._state["manager"].Queue(-1)
                self._state["logging_queue_listener"] = logging.handlers.QueueListener(self._state["logging_queue"],
                                                                                       *self.logger.handlers,
                                                                                       respect_handler_level=True)
                self._state["logging_queue_listener"].start()
                logging_queue_references = [self._state["logging_queue"]] * self._state["num_of_workers"]
            else:
                logging_queue_references = [None] * self._state["num_of_workers"]
            self._state["input_queue"] = self._state["manager"].Queue(10000)
            self._state["output_queue"] = self._state["manager"].Queue()
            self._state["worker_ids"] = tuple(range(self._state["num_of_workers"]))
            self._state["processes"] = list()
            for worker_id in range(self._state["num_of_workers"]):
                p = multiprocessing.Process(
                    target=_process_function,
                    name="worker-{}".format(worker_id),
                    args=(
                        self.process_function,
                        worker_id,
                        self._state["input_queue"],
                        self._state["output_queue"],
                        self._state["logging_queue"],
                        self.output_path,
                        len(self.output_file_extensions),
                    ),
                    kwargs={
                        "buffer_size": self.buffer_size,
                        "worker_kwargs": self.kwargs_dictionary
                    }
                )
                p.start()
                self._state["processes"].append(p)
            self._state["rotating_index"] = 0
        else:
            raise multiprocessing.ProcessError("Multiprocessor object is already forked")

    def close(self):
        accumulated_records = list()
        if self._state is not None:
            self._state["manager"] = None
            for _ in range(self._state["num_of_workers"]):
                self._state["input_queue"].put(None)
            for process in self._state["processes"]:
                process.join()
            if "logging_queue_listener" in self._state:
                self._state["logging_queue_listener"].enqueue_sentinel()
                self._state["logging_queue_listener"].stop()
            for process in self._state["processes"]:
                process.terminate()
            self.joined_files_writers = list()
            while not self._state["output_queue"].empty():
                output_files = self._state["output_queue"].get()
                files_to_be_written = min(len(self.output_file_extensions), len(output_files))
                if files_to_be_written > len(self.joined_files_writers):
                    for i in range(files_to_be_written - len(self.joined_files_writers)):
                        self.joined_files_writers.append(
                            writers.JSONWriter(
                                self.output_file_name, output_directory=self.output_path,
                                max_file_objects_amount=self.max_objects_per_output_file,
                                output_files_extension=self.output_file_extensions[len(self.joined_files_writers)],
                                buffer_size=self.buffer_size
                            )
                        )
                for i in range(files_to_be_written):
                    if len(output_files[i]) > 0:
                        json_reader = readers.JSONReader(*output_files[i])
                        file_accumulated_records = 0
                        for object_dictionary in json_reader:
                            try:
                                self.joined_files_writers[i].write_object(object_dictionary)
                            except Exception as ex:
                                self.logger.warn("Error when attempting to write {}".format(str(object_dictionary)))
                                self.logger.warn("Error: {}".format(str(ex)))
                            file_accumulated_records += 1
                        json_reader.close()
                        for output_file in output_files[i]:
                            os.remove(output_file)
                        try:
                            accumulated_records[i] += file_accumulated_records
                        except IndexError:
                            accumulated_records += [0] * (i - len(accumulated_records) + 1)
                            accumulated_records[i] += file_accumulated_records

            for writer in self.joined_files_writers:
                writer.close()
        self._state = None
        return accumulated_records
