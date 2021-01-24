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
import os

import nltk

CLI_LICENSE = "Copyright (C) 2020  Panagiotis Deligiannis\nThis program comes with ABSOLUTELY NO WARRANTY.\nThis is " \
              "free software, and you are welcome to redistribute it\nunder certain conditions."


def confirm_file_write(path):
    if os.path.exists(path):
        confirmation_given = False
        while not confirmation_given:
            dialog = input(
                "File \"{}\" already exists! Overwrite (Y/n)?".format(path))
            if dialog == "n":
                return False
            elif dialog == "Y":
                return True
    return True


def confirm_batch_file_write(directory, name_pattern):
    # base_filename = os.path.basename(path)
    # parent_directory = os.path.dirname(os.path.abspath(os.path.expanduser(path)))
    # if os.path.exists(parent_directory) and len(
    #         [file_name for file_name in os.listdir(parent_directory) if file_name.startswith(base_filename)]) > 0:
    #     confirmation_given = False
    name_prefix = name_pattern[0]
    name_extension = name_pattern[1]
    directory_files = os.listdir(directory)
    existing_pattern_files = []
    # pattern = r"^" + re.escape(name_prefix) + re.escape(".part") + r"\d+" + re.escape("." + name_extension) + r"$"
    for file in directory_files:
        # match = re.match(
        #     pattern,
        #     file
        # )
        if file.startswith(name_prefix) and file.endswith(name_extension):
            existing_pattern_files.append(file)
    if len(existing_pattern_files) > 0:
        confirmation_given = False
        while not confirmation_given:
            print(
                "Other files exist inside \"{}\" that start with \"{}\" and have an extension \"{}\". If the "
                "operation continues, these files might get overwritten.".format(
                    directory, name_prefix, name_extension
                )
            )
            example_files = existing_pattern_files[:3]
            if len(existing_pattern_files) > 3:
                example_files += ["..."]
            print("Such files are:\n\t{}".format("\n\t".join(
                list(example_files)
            )))
            dialog = input("Overwrite (Y/n)?")
            if dialog == "n":
                return False
            elif dialog == "Y":
                return True
    return True


def construct_logger(name, output_directory, execution_name, log_level):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_files_path = os.path.join(os.path.abspath(os.path.expanduser(output_directory)), "logs")
    os.makedirs(log_files_path, mode=0o744, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(log_files_path, "log_{}.log".format(
        execution_name
    )), mode="w", encoding="utf-8")
    file_handler.setLevel(logging.ERROR)
    file_formatter = logging.Formatter("%(module)s: %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if log_level > 0:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)
        if log_level > 1:
            debug_file_handler = logging.FileHandler(
                os.path.join(log_files_path, "debug_log_{}.log".format(
                    execution_name
                )), mode="w", encoding="utf-8")
            debug_file_handler.setLevel(logging.DEBUG)
            debug_file_formatter = logging.Formatter("%(levelname)s-%(module)s: %(message)s")
            debug_file_handler.setFormatter(debug_file_formatter)
            logger.addHandler(debug_file_handler)
    return logger


def cli_print_license():
    print(CLI_LICENSE)


def nltk_verify_resource(resources_reference, resource_name, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    try:
        nltk.data.find(resources_reference)
    except LookupError as le:
        logger.warning(
            "Could not find \"{}\". Attempting to fetch resource from online repository...".format(resource_name))
        fetch_successful = nltk.download(resource_name)
        if not fetch_successful:
            raise RuntimeError("Resource \"{}\" could not be found or retrieved.".format(resource_name))
