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
import bz2
import copy
import logging
import logging.handlers
import multiprocessing
import os
import pickle
import time


class Multiprocessor:

    def __init__(self, process_function, workers, output_path, *output_file_names, kwargs_dictionary=None, logger=None,
                 buffer_size=-1):
        if kwargs_dictionary is None:
            kwargs_dictionary = dict()
        self.output_file_names = output_file_names
        self.process_function = process_function
        self.workers = workers
        self.output_path = os.path.abspath(os.path.expanduser(output_path))
        self.kwargs_dictionary = kwargs_dictionary if kwargs_dictionary is not None else dict()

        self.logger = logger

        self.buffer_size = buffer_size

        self._state = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _worker_function(self, input_queue, output_queue, worker_id, logging_queue):
        worker_output_files = [os.path.join(
            self.output_path, ".{}-{}-{}.temp".format(worker_id, i, str(time.time()).replace(".", ""))
        ) for i in range(len(self.output_file_names))]
        worker_buffers = [list() for i in range(len(worker_output_files))]
        worker_buffers_size = [self.buffer_size] * len(worker_output_files)
        worker_logger = logging.getLogger("worker-{}".format(worker_id))
        worker_logger.setLevel(logging.DEBUG)
        if logging_queue is None:
            worker_logger.addHandler(logging.NullHandler())
        else:
            worker_logger.addHandler(logging.handlers.QueueHandler(logging_queue))
        sentinel_captured = False
        [open(worker_output_file, "wb").close() for worker_output_file in worker_output_files]
        bf_handles = [None] * len(worker_output_files)
        worker_kwargs = copy.deepcopy(self.kwargs_dictionary)
        worker_kwargs["worker_id"] = worker_id
        worker_kwargs["worker_logger"] = worker_logger
        while not sentinel_captured:
            input_value = input_queue.get()
            if input_value is not None:
                # arrived = False
                try:
                    # if input_value[0] == 12533608:
                    #     worker_logger.debug("pre_pre_temp_save_values: {}".format(input_value[1]))
                    #     arrived = True
                    output_value = self.process_function(input_value, **worker_kwargs)
                    # if output_value[0][0] == 12533608:
                    #     worker_logger.debug("came out: {}".format(", ".join(output_value[0][1])))
                except Exception as ex:
                    worker_logger.error(str(ex))
                    continue
                if output_value is not None:
                    # if output_value[0][0] == 12533608:
                    #     worker_logger.debug("pre_temp_save_values: {}".format(", ".join(output_value[0][1])))
                    for i in range(min(len(worker_output_files), len(output_value))):
                        result = output_value[i]
                        worker_buffers[i].append(result)
                        worker_buffers_size[i] -= 1
                        if worker_buffers_size[i] < 1:
                            bf_handles[i] = open(worker_output_files[i], "ab")
                            pickler = pickle.Pickler(bf_handles[i], protocol=pickle.HIGHEST_PROTOCOL)
                            while len(worker_buffers[i]) > 0:
                                value = worker_buffers[i].pop(0)
                                # if value[0] == 12533608:
                                #     worker_logger.debug("on_temp_save_values: {}".format(", ".join(value[1])))
                                pickler.dump(value)
                            worker_logger.debug(
                                "Dumped {} results on {}".format(self.buffer_size, worker_output_files[i]))
                            worker_buffers_size[i] = self.buffer_size
                            if worker_buffers_size[i] > 0:
                                bf_handles[i].close()
                                bf_handles[i] = None
            else:
                sentinel_captured = True
        for i in range(len(worker_output_files)):
            if len(worker_buffers[i]) > 0:
                if bf_handles[i] is None:
                    bf_handles[i] = open(worker_output_files[i], "ab")
                pickler = pickle.Pickler(bf_handles[i], protocol=pickle.HIGHEST_PROTOCOL)
                remaining = len(worker_buffers[i])
                while len(worker_buffers[i]) > 0:
                    value = worker_buffers[i].pop(0)
                    # if value[0] == 12533608:
                    #     worker_logger.debug("on_temp_save_values: {}".format(", ".join(value[1])))
                    pickler.dump(value)
                worker_logger.debug("Dumped {} results on {}".format(remaining, worker_output_files[i]))
            if bf_handles[i] is not None:
                bf_handles[i].close()
        output_queue.put(worker_output_files)

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
            # self._state["pool"] = multiprocessing.Pool(self._state["num_of_workers"])  # n-1 processes for the workers
            # self._state["pool"] = multiprocessing.Pool(self._state["num_of_workers"])  # n-1 processes for the workers
            if self.logger is not None:
                self._state["logging_queue"] = self._state["manager"].Queue(-1)
                self._state["logging_queue_listener"] = logging.handlers.QueueListener(self._state["logging_queue"],
                                                                                       *self.logger.handlers,
                                                                                       respect_handler_level=True)
                self._state["logging_queue_listener"].start()
                logging_queue_references = [self._state["logging_queue"]] * self._state["num_of_workers"]
            else:
                logging_queue_references = [None] * self._state["num_of_workers"]
            # self._state["input_queues"] = tuple(self._state["manager"].Queue()
            #                                     for _ in range(self._state["num_of_workers"])
            #                                     )
            self._state["input_queue"] = self._state["manager"].Queue(10000)
            self._state["output_queue"] = self._state["manager"].Queue()
            self._state["worker_ids"] = tuple(range(self._state["num_of_workers"]))
            self._state["processes"] = list()
            for worker_id in range(self._state["num_of_workers"]):
                p = multiprocessing.Process(
                    target=self._worker_function,
                    name="worker-{}".format(worker_id),
                    args=(
                        self._state["input_queue"],
                        self._state["output_queue"],
                        worker_id,
                        self._state["logging_queue"])
                )
                p.start()
                self._state["processes"].append(p)
            # self._state["async_result"] = self._state["pool"].starmap(self._worker_function, worker_args)
            self._state["rotating_index"] = 0
        else:
            raise multiprocessing.ProcessError("Multiprocessor object is already forked")

    def close(self):
        accumulated_records = list()
        if self._state is not None:
            self._state["manager"] = None
            # for queue in self._state["input_queues"]:
            for _ in range(self._state["num_of_workers"]):
                self._state["input_queue"].put(None)
            # self._state["pool"].close()
            # self._state["pool"].join()
            for process in self._state["processes"]:
                process.join()
            if "logging_queue_listener" in self._state:
                self._state["logging_queue_listener"].enqueue_sentinel()
                self._state["logging_queue_listener"].stop()
            [bz2.BZ2File(os.path.join(self.output_path, self.output_file_names[i]), "wb").close()
             for i in range(len(self.output_file_names))]
            while not self._state["output_queue"].empty():
                output_files = self._state["output_queue"].get()
                for i in range(min(len(self.output_file_names), len(output_files))):
                    with bz2.BZ2File(os.path.join(self.output_path, self.output_file_names[i]), "ab") as bf_out_handle:
                        pickler = pickle.Pickler(bf_out_handle, protocol=pickle.HIGHEST_PROTOCOL)
                        with open(output_files[i], "rb") as bf_in_handle:
                            file_accumulated_records = 0
                            unpickler = pickle.Unpickler(bf_in_handle)
                            reached_eof = False
                            while not reached_eof:
                                try:
                                    output_value = unpickler.load()
                                except EOFError:
                                    reached_eof = True
                                    continue
                                pickler.dump(output_value)
                                file_accumulated_records += 1
                            os.remove(output_files[i])
                            try:
                                accumulated_records[i] += file_accumulated_records
                            except IndexError:
                                accumulated_records += [0] * (i - len(accumulated_records) + 1)
                                accumulated_records[i] += file_accumulated_records
        self._state = None
        return accumulated_records
