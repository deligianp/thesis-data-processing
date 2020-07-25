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

import numpy as np
from gensim.corpora import Dictionary


def jaccard_distance_matrix(topic_terms_0_path, dictionary_0, topic_terms_1_path, dictionary_1, topn=50, logger=None):
    """
    Given the top-N terms of two topics, jaccard distance calculates the proportion of terms that are common in the two
    top-N sets.

    :param str topic_terms_0_path: path to the first model's topic-terms numpy array file
    :param str topic_terms_1_path: path to the second model's topic-terms numpy array file
    :param int topn: the number of most frequent terms in each topic
    :return numpy.ndarray: Jaccard distance matrix, shaped MxN, for M,N the number of topics in the first and second
    models' topic terms matrices respectively
    """
    if not logger:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    logger.debug("Loading first topic terms matrix from given path.")
    m0_topics = np.load(topic_terms_0_path)
    logger.debug("Loaded first topic terms matrix.")
    logger.debug("Loading second topic terms matrix from given path.")
    m1_topics = np.load(topic_terms_1_path)
    logger.debug("Loaded second topic terms matrix.")

    logger.debug("Loading first model dictionary")
    m0_dictionary = Dictionary.load(dictionary_0)
    logger.debug("Loading second model dictionary")
    m1_dictionary = Dictionary.load(dictionary_1)

    # Transform to "ranked token indices"
    logger.debug("First matrix transformation: Keeping the indices of the top {} tokens, for each topic.".format(topn))
    m0_topics = np.flip(np.argsort(m0_topics), 1)[:, :topn]
    logger.debug("Second matrix transformation: Keeping the indices of the top {} tokens, for each topic.".format(topn))
    m1_topics = np.flip(np.argsort(m1_topics), 1)[:, :topn]

    jd_matrix = np.zeros((len(m0_topics), len(m1_topics)))
    logger.debug("Procceeding to calculation...")
    for topic_index_0 in range(len(m0_topics)):
        topn_terms_0 = set(m0_dictionary[term_index] for term_index in m0_topics[topic_index_0])
        for topic_index_1 in range(len(m1_topics)):
            topn_terms_1 = set(m1_dictionary[term_index] for term_index in m1_topics[topic_index_1])
            # DEBUG - start
            # intersection = topn_terms_0.intersection(topn_terms_1)
            # union = topn_terms_0.union(topn_terms_1)
            # DEBUG - end
            jd_matrix[topic_index_0][topic_index_1] = len(topn_terms_0.intersection(topn_terms_1)) / len(
                topn_terms_0.union(topn_terms_1))
    return jd_matrix


# # Given a vector V and a matrix A, calculate the distance of that vector V from each vector in the given matrix A
# def _euclidean_distance_vector_matrix(vector, matrix):
#     # max() function is used to compensate for the calculation error of square root that results in a correlation
#     # coefficient lower that -1 in some cases
#     return np.apply_along_axis(lambda u, v: np.linalg.norm(u - v), 1, matrix, vector)
#
#
# def euclidean_distance_matrix(topic_terms_0_path, topic_terms_1_path):  # TODO
#     """
#     This function treats each topic-term array, of each of the models given, as a vector with values [0-1] with a vector
#     sum of 1. This way the distance between each topic representation can be modeled by the euclidean distance.
#
#     :param str topic_terms_0_path: path to the first model's topic-terms numpy array file
#     :param str topic_terms_1_path: path to the second model's topic-terms numpy array file
#     :return numpy.ndarray: Euclidean distance matrix, shaped MxN, for M,N the number of topics in the first and second
#     models' topic terms matrices respectively
#     """
#
#     MODULE_LOGGER.debug("Loading first topic terms matrix from given path.")
#     m0_topics = np.load(topic_terms_0_path)
#     MODULE_LOGGER.debug("Loaded first topic terms matrix.")
#     MODULE_LOGGER.debug("Loading second topic terms matrix from given path.")
#     m1_topics = np.load(topic_terms_1_path)
#     MODULE_LOGGER.debug("Loaded second topic terms matrix.")
#
#     MODULE_LOGGER.debug("Procceeding to calculation...")
#     return np.apply_along_axis(_euclidean_distance_vector_matrix, 1, m0_topics, m1_topics)


def spearman_rho_correlation_matrix(topic_terms_0_path, dictionary_0, topic_terms_1_path, dictionary_1, topn=50,
                                    logger=None):
    """
    Spearman rho is a correlation coefficient that is used to measure the distance between the ranks of different ranked
     objects.

    This implementation of Spearman rho correlation, takes into account the rank of the top-N terms of each topic of the
     two models and provides a metric of how much the two topic-term distributions rank the terms alike.

    :param str topic_terms_0_path: path to the first model's topic-terms numpy array file
    :param str topic_terms_1_path: path to the second model's topic-terms numpy array file
    :param int topn: the number of most frequent terms in each topic
    :return: numpy.ndarray: Spearman rho correlation matrix, shaped MxN, for M,N the number of topics in in the first
    and second models' topic terms matrices respectively
    """
    logger.debug("Loading first topic terms matrix from given path.")
    m0_topics = np.load(topic_terms_0_path)
    logger.debug("Loaded first topic terms matrix.")
    logger.debug("Loading second topic terms matrix from given path.")
    m1_topics = np.load(topic_terms_1_path)
    logger.debug("Loaded second topic terms matrix.")

    logger.debug("Loading first model dictionary")
    m0_dictionary = Dictionary.load(dictionary_0)
    logger.debug("Loading second model dictionary")
    m1_dictionary = Dictionary.load(dictionary_1)

    logger.debug("First matrix transformation: Keeping the indices of the top {} tokens, for each topic.")
    m0_topics = np.flip(np.argsort(m0_topics), 1)[:, :topn]
    logger.debug("Second matrix transformation: Keeping the indices of the top {} tokens, for each topic.")
    m1_topics = np.flip(np.argsort(m1_topics), 1)[:, :topn]

    m0_num_topics = len(m0_topics)
    m1_num_topics = len(m1_topics)

    correlation_coefficients = np.zeros((m0_num_topics, m1_num_topics))
    logger.debug("Procceeding to calculation...")
    for topic_index_0 in range(m0_num_topics):
        topn_terms_0 = [m0_dictionary[term_index] for term_index in m0_topics[topic_index_0]]
        topn_term_ranks_0 = {topn_terms_0[index]: index + 1 for index in range(topn)}
        for topic_index_1 in range(m1_num_topics):
            topn_terms_1 = [m1_dictionary[term_index] for term_index in m1_topics[topic_index_1]]
            topn_term_ranks_1 = {topn_terms_1[index]: index + 1 for index in range(topn)}
            correlation_coefficients[topic_index_0, topic_index_1] = _top_n_spearman_rho(topn_term_ranks_0,
                                                                                         topn_term_ranks_1)
    return correlation_coefficients


def _top_n_spearman_rho(t0, t1):
    """
    Top-N spearman correlation is a variant of Spearman correlation coefficient that compares the difference of rankings
     between between two subsets, containing the top-N ranking objects of a common universe. The top-N lists are mapping
     the top-N objects of a universe to rank numbers [1-N]. Since the two lists can have different ranking criteria,
    each list might have ranking to exclusive objects. To compensate, top-N Spearman rho, assigns as rank, for each non
    common object as l=N+1.

    More information:
    Ronald Fagin, Ravi Kumar, D. Sivakumar: Comparing top k lists (2003)

    :param dict t0: Mapping of object -> rank
    :param dict t1: Mapping of object -> rank
    :return:
    """
    n = len(t0)
    assert n == len(t1), "Different-sized mappings were given"
    l_parameter = n + 1
    spearman_footrule = 0
    for key in t0:
        spearman_footrule += (t0[key] - t1.pop(key, l_parameter)) ** 2
    for key in t1:
        spearman_footrule += (t1[key] - l_parameter) ** 2
    spearman_rho = 1 - ((6 * spearman_footrule) / (n * (n + 1) * (2 * n + 1)))
    return spearman_rho


registered_metrics = {
    "jaccard": jaccard_distance_matrix,
    "spearman": spearman_rho_correlation_matrix
}
