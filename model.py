import datetime
import logging
import numpy as np
import os
import sys
import time
import math
import autosklearn.classification
import autosklearn.regression


class Model:
    def __init__(self, metadata):
        """
        The initalization procedure for your method given the metadata of the task
        """
        """
        Args:
          metadata: an DecathlonMetadata object. Its definition can be found in
              ingestion/dev_datasets.py
        """
        self.metadata_ = metadata

        # Getting details of the data from meta data
        # Product of output dimensions in case of multi-dimensional outputs...
        self.output_dim = math.prod(self.metadata_.get_output_shape())

        row_count, col_count = self.metadata_.get_tensor_shape()[2:4]
        channel = self.metadata_.get_tensor_shape()[1]
        sequence_size = self.metadata_.get_tensor_shape()[0]

        self.num_train = self.metadata_.size()

        self.input_shape = (sequence_size, channel, row_count, col_count)
        print("\n\nINPUT SHAPE = ", self.input_shape)


        # Auto-Sklearn Implementation Choice
        if self.metadata_.get_task_type() == "continuous":
            self.automl = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=120)
        elif self.metadata_.get_task_type() == "single-label":
            self.automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120,
                                                                              per_run_time_limit=30,)
        elif self.metadata_.get_task_type() == "multi-label":
            self.automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60,per_run_time_limit=30,
                                                                              # Bellow two flags are provided to speed up calculations
                                                                              # Not recommended for a real implementation
                                                                              initial_configurations_via_metalearning=0,
                                                                              smac_scenario_args={"runcount_limit": 1},)
        else:
            raise NotImplementedError

        # Attributes for managing time budget
        # Cumulated number of training steps
        self.birthday = time.time()
        self.total_train_time = 0
        self.total_test_time = 0

        # no of examples at each step/batch
        self.train_batch_size = 64
        self.test_batch_size = 64



    def train(
        self, dataset, val_dataset=None, val_metadata=None, remaining_time_budget=None
    ):
        """
        The training procedure of your method given training data, validation data (which is only directly provided in certain tasks, otherwise you are free to create your own validation strategies), and remaining time budget for training.
        """

        """Train this algorithm on the Pytorch dataset.

        ****************************************************************************
        ****************************************************************************

        Args:
          dataset: a `DecathlonDataset` object. Each of its examples is of the form
                (example, labels)
              where `example` is a dense 4-D Tensor of shape
                (sequence_size, row_count, col_count, num_channels)
              and `labels` is a 1-D or 2-D Tensor

          val_dataset: a 'DecathlonDataset' object. Is not 'None' if a pre-split validation set is provided, in which case you should use it for any validation purposes. Otherwise, you are free to create your own validation split(s) as desired.
          
          val_metadata: a 'DecathlonMetadata' object, corresponding to 'val_dataset'.

          remaining_time_budget: time remaining to execute train(). The method
              should be tuned to fit within this budget.
        """

        logger.info("Begin training...")

        # If PyTorch dataloader for training set doen't already exists, get the train dataloader
        # if not hasattr(self, "trainloader"):
        #     self.trainloader = self.get_dataloader(
        #         dataset,
        #         self.train_batch_size,
        #         "train",
        #     )

        train_start = time.time()

        X = []
        y = []

        for i in len(dataset.dataset):
          xx, yy = dataset.dataset[i]
          X.append(xx)
          y.append(yy)

        X = np.array(X)
        y = np.array(y)

        self.automl.fit(X, y)

        # Training loop
        epochs_to_train = 200  # may adjust as necessary
        # self.trainloop(self.criterion, self.optimizer, epochs=epochs_to_train)
        train_end = time.time()

        # Update for time budget managing
        train_duration = train_end - train_start
        self.total_train_time += train_duration

        logger.info(
            "{} epochs trained. {:.2f} sec used. ".format(
                epochs_to_train, train_duration
            )
            + "Total time used for training: {:.2f} sec. ".format(self.total_train_time)
        )

    def test(self, dataset, remaining_time_budget=None):
        """Test this algorithm on the Pytorch dataloader.

        Args:
          Same as that of `train` method, except that the `labels` will be empty.
        Returns:
          predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
              here `sample_count` is the number of examples in this dataset as test
              set and `output_dim` is the number of labels to be predicted. The
              values should be binary or in the interval [0,1].
        """

        test_begin = time.time()

        logger.info("Begin testing...")

        if not hasattr(self, "testloader"):
            self.testloader = self.get_dataloader(
                dataset,
                self.test_batch_size,
                "test",
            )

        # get predictions automl model
        predictions = self.automl.predict(dataset[0])

        test_end = time.time()
        # Update some variables for time management
        test_duration = test_end - test_begin

        logger.info(
            "[+] Successfully made predictions. {:.2f} sec used. ".format(test_duration)
        )
        return predictions

    ##############################################################################
    #### Above 3 methods (__init__, train, test) should always be implemented ####
    ##############################################################################

def get_logger(verbosity_level):
    """Set logging format to something like:
    2019-04-25 12:52:51,924 INFO model.py: <message>
    """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(filename)s: %(message)s"
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


logger = get_logger("INFO")