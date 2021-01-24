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
Module that contains the system's readers as well as the BaseReader that custom readers, inside application.readers
should extend

The module is loaded by all of the data reading scripts and provides all available readers in the scope of this system.
All reader class definitions are stored inside
"""
import importlib.util
import inspect
import logging
import os
import sys
from abc import ABCMeta, abstractmethod

import ijson.backends.yajl2_c as ijson


class BaseReader(metaclass=ABCMeta):

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def read_batch(self, batch_size):
        pass


class JSONReader(BaseReader):

    def read(self):
        if self.file_obj is None:
            iter(self)
        val = next(self)
        return val

    def read_batch(self, batch_size):
        return tuple(next(self) for _ in range(batch_size))

    def __init__(self, *file_paths, logger=None):
        self.file_paths = tuple(os.path.abspath(os.path.expanduser(file_path)) for file_path in file_paths if
                                os.path.exists(os.path.abspath(os.path.expanduser(file_path))))
        assert len(self.file_paths) > 0, "No valid files could be located"

        self.file_obj = None

        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler())

    def __iter__(self):
        if self.file_obj is not None:
            self.file_obj.close()
        self.file_index = 0
        self.object_index = 0
        self.file_obj = open(self.file_paths[self.file_index], "rb")
        self.json_parser = ijson.items(self.file_obj, "item")
        return self

    def __next__(self):
        read_further = True
        while read_further:
            try:
                object_value = next(self.json_parser)
            except StopIteration:
                self.file_index += 1
                self.file_obj.close()
                self.file_obj = None
                if self.file_index == len(self.file_paths):
                    raise StopIteration
                else:
                    self.file_obj = open(self.file_paths[self.file_index], "rb")
                    self.json_parser = ijson.items(self.file_obj, "item")
                    continue
            except OSError:
                self.logger.error(
                    "An error occurred while attempting to read from the file {}.\nEnsure that the file can be read by "
                    "the provided reader: JSONReader".format(self.file_paths[self.file_index])
                )
                self.file_index += 1
                continue

            self.object_index += 1
            return object_value

    def close(self):
        pass


available_readers = {
    (cl[0].lower()[:-len("Reader")] if cl[0].endswith("Reader") else cl[0].lower()): cl[1]
    for cl in inspect.getmembers(sys.modules[__name__], inspect.isclass) if
    (issubclass(cl[1], BaseReader) and cl[0] != "BaseReader" and cl[0].endswith("Reader"))
}
custom_readers = importlib.util.find_spec("application.readers")
if custom_readers is not None:
    loader = custom_readers.loader
    custom_module = loader.load_module()
    for ccl in inspect.getmembers(custom_module, inspect.isclass):
        if issubclass(ccl[1], BaseReader) and ccl[0] != "BaseReader":
            if ccl[0].endswith("Reader"):
                description_string = ccl[0][:-len("Reader")].lower()
            else:
                description_string = ccl[0].lower()

            if description_string in available_readers:
                description_string = "app_" + description_string
                if description_string in available_readers:
                    continue
            available_readers[description_string] = ccl[1]
