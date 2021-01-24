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

from src.util import docfetch

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(
        description="This script serves as an informative assistant that can be used "
                    "in order to obtain information regarding other parts of this "
                    "application")
    argument_parser.add_argument("command", choices=("list-readers", "list-filters", "list-preprocessors"))

    arguments_namespace = argument_parser.parse_args()

    command = arguments_namespace.command

    if command == "list-readers":
        from src.core.file import readers as f_readers

        available_readers = f_readers.available_readers

        for preprocessor_key in available_readers:
            preprocessor_class = available_readers[preprocessor_key]
            documentation = preprocessor_class.__doc__ if preprocessor_class.__doc__ is not None and len(
                preprocessor_class.__doc__.strip()) > 0 else ""
            pretty_documentation = "* " + preprocessor_key + " - " + preprocessor_class.__name__ + "\n"
            if documentation != "":
                pretty_documentation += "\n    " + "\n    ".join(
                    chunk.replace("\n", "\n    ") for chunk in docfetch.sanitize_docstring(documentation, width=68)
                ) + "\n"
            print(pretty_documentation)

    elif command == "list-filters":
        from src.core.train import filters

        available_preprocessors = filters.available_filters
        for preprocessor_key in available_preprocessors:
            preprocessor_class = available_preprocessors[preprocessor_key]
            documentation = preprocessor_class.__doc__ if preprocessor_class.__doc__ is not None and len(
                preprocessor_class.__doc__.strip()) > 0 else ""
            pretty_documentation = "* " + preprocessor_key + " - " + preprocessor_class.__name__ + "\n"
            if documentation != "":
                pretty_documentation += "\n    " + "\n    ".join(
                    chunk.replace("\n", "\n    ") for chunk in docfetch.sanitize_docstring(documentation, width=68)
                ) + "\n"
            print(pretty_documentation)

    elif command == "list-preprocessors":
        from src.core.train import preprocessors

        available_preprocessors = preprocessors.available_preprocessors
        for preprocessor_key in available_preprocessors:
            preprocessor_class = available_preprocessors[preprocessor_key]
            documentation = preprocessor_class.__doc__ if preprocessor_class.__doc__ is not None and len(
                preprocessor_class.__doc__.strip()) > 0 else ""
            pretty_documentation = "* " + preprocessor_key + " - " + preprocessor_class.__name__ + "\n"
            if documentation != "":
                pretty_documentation += "\n    " + "\n    ".join(
                    chunk.replace("\n", "\n    ") for chunk in docfetch.sanitize_docstring(documentation, width=68)
                ) + "\n"
            print(pretty_documentation)
