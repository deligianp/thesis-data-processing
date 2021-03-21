import random
import math


def crossvalidation_split(reader, writers):
    random.shuffle(writers)
    writer_idx = 0
    for instance in reader:
        writers[writer_idx].write(instance)
        writer_idx = (writer_idx + 1) % len(writers)


def proportional_split(reader, writer, proportion, batch_size=None):
    """
    Proportional split is used in order to create a training-test split based on a proportion.

    The proportion corresponds to the amount of training instances that will form the training set, with respect to the
    total size of the dataset.

    The function expects a reader instance and a pair of writers, as well as the training set proportion, as a
    decimal number in the range (0,1).
    :param reader:
    :param writer:
    :param proportion:
    :param batch_size:
    :return:
    """
    if not batch_size:
        for instance in reader:
            writer_idx = 0 if random.random() < proportion else 1
            writer[writer_idx].write_object(instance)
    else:
        dataset_size = len(reader)
        training_size = math.floor(dataset_size * proportion)
        n_training_instances = 0

        reached_eof = False
        while not reached_eof:
            batch = reader.read_batch(batch_size=batch_size)
            reached_eof = len(batch) == 0
            if reached_eof:
                continue
            random.shuffle(batch)
            n_batch_candidate_training_instances = math.floor(len(batch) * proportion)
            if training_size - n_training_instances > n_batch_candidate_training_instances:
                n_batch_training_instances = n_batch_candidate_training_instances
            else:
                n_batch_training_instances = training_size - n_training_instances
            for instance in batch[:n_batch_training_instances]:
                writer[0].write_object(instance)
            for instance in batch[n_batch_training_instances:]:
                writer[1].write_object(instance)
