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
This module hosts the writers that are available and used by the trainer

Writers are used to save the intermediate states of data, during a topic modeling task
"""
import json
import os
from abc import ABCMeta, abstractmethod


class BaseWriter(metaclass=ABCMeta):
    """
    Base interface for defining writers.
    """

    @abstractmethod
    def write_object(self, obj):
        """
        Abstract method that should implement the storing process of an object.

        :param obj: The object to be written
        :type obj: object
        """
        pass

    @abstractmethod
    def close(self):
        """
        Abstract method that should implement the finalization of the object-writing process .
        """
        pass


class JSONWriter(BaseWriter):
    """
    A writer class that accepts Python dictionaries and stores them in sets of JSON files.

    Produced JSON files contain a root JSON array, in which, all different objects are stored. JSON writer can be
    configured to either store all objects in a single result file or a set of result files. For the first case, each
    file's name ends with ".[EXTENSION]" where [EXTENSION] refers to the extension specified during the initialization
    of the writer. For the second case, each file's name ends with ".part[N].[EXTENSION]", where [N] is an index that
    describes that identifies the part in the set of result files.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and exc_val is None and exc_tb is None:
            self.close()
        else:
            print(exc_type)
            print(exc_val)
            # If an error occured inside the context manager, then erase produced files
            for file in self.created_files:
                try:
                    os.remove(file)
                except Exception as ex:
                    raise ex

    def __init__(self, output_files_prefix, output_directory=None, max_file_objects_amount=None,
                 output_files_extension=None, buffer_size=1000, indent=None, encoding=None):
        """
        Constructs a JSON writer

        :param output_files_prefix: common name for produced output files. This translates to the part of the of the
            produced files' name before the "part[NUMBER].[EXTENSION]" ending
        :type output_files_prefix: str
        :param output_directory: path pointing to the EXISTING output directory where result files will be stored in.
            By default, the current working directory is used
        :type output_directory: str, optional
        :param max_file_objects_amount: the maximum number of objects a JSON file must contain. This should be an
            integer, greater than 0. If an integer, 0 or lower, is given, or None, then all objects will be stored in a
            single file. By default, all objects will be stored in a single file
        :type max_file_objects_amount: Union[int, None], optional
        :param output_files_extension: extension to be used for the produced result files. If not given, "json" is used
            as the extension
        :type output_files_extension: str, optional
        :param buffer_size: the amount of objects to be written in a single file write process. This defaults to 1000
        :type buffer_size: int, Optional
        :param indent: the indentation to be used in resulting output files, as described in the arguments of
            `Python's json package's "dump" function <https://docs.python.org/3/library/json.html#json.dump>`_. By
            default the writer will use the most condense representation of each object, while maintaining the
            indentation of the root array at all times, to a single TAB
        :type indent: Union[str, int, None], optional
        :param encoding: encoding of the produced output files. For the available encodings refer to the
            `Python's standard encodigns <https://docs.python.org/3/library/codecs.html#standard-encodings>`_. If not
            defined, "utf-8" is used
        :type encoding: str
        """
        # Call super
        super(JSONWriter, self).__init__()

        #
        self.created_files = []
        self.current_file_path = None
        self.current_file_handle = None
        self.current_file_index = 1
        self.current_file_objects = 0
        self.buffer = []

        self.output_files_prefix = output_files_prefix

        self.output_files_extension = output_files_extension or "json"
        self.max_file_objects_amount = max_file_objects_amount or -1
        self.output_directory = output_directory or "."

        self.indent = indent
        self.encoding = encoding or "utf-8"

        assert buffer_size > 0 and type(buffer_size) is int, "Buffer size must be a positive integer"
        self.buffer_size = buffer_size

    def _flush_buffer(self):

        # Write data to file
        if not self.current_file_path:

            # If there is a limit in the number of objects that a file can contain, then it is possible that the output
            # result may split into different files. Therefore, the output file name will contain the part number
            if self.max_file_objects_amount > 0:
                current_file_path = os.path.join(
                    self.output_directory, self.output_files_prefix + ".part{}.{}".format(
                        self.current_file_index, self.output_files_extension
                    )
                )

            # If max_file_objects_amount is not greater than zero, then the writer will write all produced output in a
            # single file. Thus no part number is needed in the produced file name
            else:
                current_file_path = os.path.join(
                    self.output_directory, self.output_files_prefix + ".{}".format(self.output_files_extension)
                )

            # Create the file and overwrite any other file that exists with the same name
            current_file_handle = open(current_file_path, "w", encoding="utf-8")

            # Register current writer status
            self.created_files.append(current_file_path)
            self.current_file_path = current_file_path
            self.current_file_handle = current_file_handle

            # Write in the beginning of the file the notation for defining a JSON array
            self.current_file_handle.write("[\n")
            self.current_file_objects = 0

        if self.max_file_objects_amount > 0:
            num_of_objects = min(self.buffer_size, self.max_file_objects_amount - self.current_file_objects)
        else:
            num_of_objects = self.buffer_size
        output_json = ",\n".join(self.buffer[:num_of_objects])

        if self.current_file_objects != 0:
            output_json = ",\n" + output_json
        self.current_file_handle.write(output_json)
        self.current_file_objects += num_of_objects
        self.buffer = self.buffer[num_of_objects:]

        if self.current_file_objects >= self.max_file_objects_amount > 0:
            self.current_file_handle.write("\n]")
            self.current_file_handle.close()
            self.current_file_handle = None
            self.current_file_path = None
            self.current_file_index += 1

    def write_object(self, object_dictionary):
        json_string = json.dumps(object_dictionary, indent=self.indent)
        json_file_string = "\t" + json_string.replace("\n", "\n\t")
        self.buffer.append(json_file_string)

        if len(self.buffer) >= self.buffer_size:
            self._flush_buffer()

    def close(self):
        while len(self.buffer) > 0:
            self._flush_buffer()
        if self.current_file_handle:
            self.current_file_handle.write("\n]")
            self.current_file_handle.close()
            self.current_file_handle = None
            self.current_file_path = None
            self.current_file_index += 1
