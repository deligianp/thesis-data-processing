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
A basic custom module that normalizes a docstring for CLI printing.

The module offers a single function for normalizing a given docstring in order to restrict its width to fit to a CLI.
In addition it attempts to maintain its morphological structure and indentation.

To achieve the above, the module makes certain assumptions.
    1. The docstring should begin either:
        * after the opening quotes
        * in any subsequent line, indented by one position to the right, in respect to the documented code entity.
    2. The docstring uses a nicely-indented form. This means that the amount of the indentation should decrease only
    when it has been prior increased. As a result, if docstring is indented more than any subsequent line, the
    indentation, will be in respect with the leftmost line.
    3. Line changes inside docstring, are used to break the docstring to separate lines. The resulting docstring
    segments are being connected automatically with a space. Thus there is no reason for docstring authors to leave a
    space before changing to a new line.
"""
import re
import textwrap


def _reduce_upper_vertical_space(lines_list):
    """
    This function is used internally by the module.

    Given a list of lines, the function removes any empty lines before the first line with non-whitespace content.
    :param lines_list: The list of lines.
    :return: Returns a reduced list of lines where any empty upper lines have been removed.
    """
    non_empty_line_number = 0
    ne_line_found = False
    i = 0
    while not ne_line_found and i < len(lines_list):
        line = lines_list[i]
        if len(line.strip()) > 0:
            ne_line_found = True
            non_empty_line_number = i
            continue
        i += 1
    return lines_list[non_empty_line_number:]


def _reduce_lower_vertical_space(lines_list):
    """
    This function is used internally by the module.

    Given a list of lines, the function removes any empty lines after the last line with non-whitespace content.
    :param lines_list: The list of lines.
    :return: Returns a reduced list of lines where any empty lower lines have been removed.
    """
    lines_list.reverse()
    reduced_lines_list = _reduce_upper_vertical_space(lines_list)
    lines_list.reverse()
    reduced_lines_list.reverse()
    return reduced_lines_list


def _reduce_vertical_space(lines_list):
    """
    This function is used internally by the module.

    Given a list of lines, the function removes any empty lines above and beneath the first and last line with
    non-whitespace content, respectively.
    :param lines_list: The list of lines.
    :return: Returns a reduced list of lines where any empty lines above and beneath the first and last line with
    non-whitespace content, respectively, have been removed.
    """
    left_reduced = _reduce_upper_vertical_space(lines_list)
    return _reduce_lower_vertical_space(left_reduced)


def _normalize_identation(lines_list):
    """
    This function is used internally by the module.

    It normalizes the indentation for all lines by subtracting the common indentation amount.

    Records the amount of indentation for each line and subtracts from each the lowest common amount of indentation.
    This translates to shifting all lines to the left until the left most line has no indentation.
    :param lines_list: The list of lines.
    :return: The given list of lines with normalized indentation.
    """
    identation_lengths = list()
    if len(lines_list) > 1:
        for line in lines_list[1:]:
            if len(line.strip()) > 0:
                identation_lengths.append(len(line) - len(line.lstrip()))
        selected_length = min(identation_lengths)
        first_line_identation = len(lines_list[0]) - len(lines_list[0].lstrip())
        normalized_lines = [lines_list[0].rstrip()[min(first_line_identation, selected_length):]]
        normalized_lines += [line[selected_length:] for line in lines_list[1:]]
        return normalized_lines
    else:
        return lines_list[0].strip()


def _text_chunks(lines_list):
    """
    This function is used internally by the module.

    It infers the text chunks that their contents should be wrapped to new lines when they reach the end of CLI.

    To be able to create textual representations that can be wrapped when a line in CLI has been filled, the initial
    lines of a docstring must be concatenated together. However, this generates a new problem; there are times where
    the split in the docstring represent an actual line break rather that the author's wrapping of the docstring.

    This function aims to identify the aforementioned conditions and transform a given list of docstring lines to a
    list of chunks along with their indentation, where each chunk can be wrapped when a new line is needed. A new chunk
    is generated when:
        * the indentation changes
            e.g. The docstring
                \"\"\"
                line 0
                line 1
                    line 2
                \"\"\"
            yields the chunks [["line 0 line 1",""], ["line 2","    "]]
        * an empty line is met
            e.g. The docstring
                \"\"\"
                line 0


                line 1
                \"\"\"
            yields the chunks [["line 0",""], ["",""], ["",""], ["line 1",""]]
        * a listing item is defined
            e.g. The docstring
                \"\"\"
                line 0
                    * this is a very long line that also
                    has text in the following line too
                    * line 2
                line 3
                \"\"\"
            yields the chunks [["line 0",""],
            ["* this is a very long line that also has text in the following line too","    "],
            ["* line 2","    "]
            ["line 3",""]]

    For all of the examples above, the second part of each chunk, represents the chunk indentation.
    :param lines_list: The list of lines.
    :return: The corresponding list of chunks.
    """
    chunks = list()
    current_chunk_text = ""
    current_indentation = ""
    for line in lines_list:
        if len(line.strip()) > 0:
            match = re.match(r"^(\s+)*(((\d)+\. )|([-*+] ))*(.+)$", line)
            line_identation = match.group(1)
            listing_declaration = match.group(2)
            rest_of_line = match.group(6)
            line_identation = "" if not line_identation else line_identation
            if line_identation != current_indentation:
                # case where new idented block was found
                chunks.append([current_chunk_text, current_indentation])
                if listing_declaration:
                    current_chunk_text = listing_declaration + rest_of_line
                else:
                    current_chunk_text = rest_of_line
                current_indentation = line_identation
            else:
                if listing_declaration is not None:
                    chunks.append([current_chunk_text, current_indentation])
                    current_chunk_text = listing_declaration + rest_of_line
                    current_indentation = line_identation
                else:
                    current_chunk_text += rest_of_line
        else:
            chunks.append([current_chunk_text, current_indentation])
            chunks.append(["", ""])
            current_indentation = ""
            current_chunk_text = ""
    if len(current_chunk_text) > 0:
        chunks.append([current_chunk_text, current_indentation])
    return chunks


def _replace_tabs(lines_list, tabsize):
    """
    This function is used internally by the module.

    It replaces all tab instances, \"\t\", with multiple spaces, found inside a list of lines.

    :param lines_list: The list of lines.
    :param tabsize: The amount of spaces to be used for a single tab character. Default: 4.
    :return: Returns a list of lines where tab instances, \"\t\", have been replaced with multiple spaces.
    """
    return [line.replace("\t", " " * tabsize) for line in lines_list]


def sanitize_docstring(docstring, tabsize=4, width=72, normalize_indentation=True):
    """
    The function for sanitizing a docstring to a fixed width and with no tabs for indentation

    :param docstring: a docstring
    :param tabsize: amount of spaces that each tab, \"\t\", should be substituted with
    :param width: target number of characters for each line
    :param normalize_indentation: if True, the common indentation whitespace from all lines is removed so that text is
    shifted to the left
    :return: Returns a list of chunks that can be joined with a new line to obtain the original representation of the
    docstring
    """
    lines = docstring.split("\n")
    reduced_lines = _reduce_vertical_space(lines)
    no_tabs_lines = _replace_tabs(reduced_lines, tabsize)
    if normalize_indentation:
        normalized_lines = _normalize_identation(no_tabs_lines)
        chunks = _text_chunks(normalized_lines)
    else:
        chunks = _text_chunks(no_tabs_lines)
    normalized_chunks = list()
    for chunk in chunks:
        normalized_chunks.append("\n".join(
            textwrap.wrap(chunk[0], width=width, initial_indent=chunk[1], subsequent_indent=chunk[1])
        ))
    return normalized_chunks
