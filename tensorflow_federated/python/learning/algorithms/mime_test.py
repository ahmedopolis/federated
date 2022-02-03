# Copyright 2022, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
from unittest import mock

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_update_aggregator
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.algorithms import aggregation
from tensorflow_federated.python.learning.algorithms import mime
from tensorflow_federated.python.learning.framework import dataset_reduce
from tensorflow_federated.python.learning.optimizers import adagrad
from tensorflow_federated.python.learning.optimizers import adam
from tensorflow_federated.python.learning.optimizers import rmsprop
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.optimizers import yogi
from tensorflow_federated.python.learning.templates import client_works
from tensorflow_federated.python.learning.templates import distributors


class MimeLiteClientWorkComputationTest(test_case.TestCase,
                                        parameterized.TestCase):

  @parameterized.named_parameters(
      ('uniform', client_weight_lib.ClientWeighting.UNIFORM),
      ('num_examples', client_weight_lib.ClientWeighting.NUM_EXAMPLES))
  def test_type_properties(self, weighting):
    model_fn = model_examples.LinearRegression
    optimizer = sgdm.build_sgdm(learning_rate=0.1, momentum=0.9)
    client_work_process = mime._build_mime_lite_client_work(
        model_fn, optimizer, weighting)
    self.assertIsInstance(client_work_process, client_works.ClientWorkProcess)

    mw_type = model_utils.ModelWeights(
        trainable=computation_types.to_type([(tf.float32, (2, 1)), tf.float32]),
        non_trainable=computation_types.to_type([tf.float32]))
    expected_param_model_weights_type = computation_types.at_clients(mw_type)
    expected_param_data_type = computation_types.at_clients(
        computation_types.SequenceType(
            computation_types.to_type(model_fn().input_spec)))
    expected_result_type = computation_types.at_clients(
        client_works.ClientResult(
            update=mw_type.trainable,
            update_weight=computation_types.TensorType(tf.float32)))
    expected_optimizer_state_type = type_conversions.type_from_tensors(
        optimizer.initialize(
            type_conversions.type_to_tf_tensor_specs(mw_type.trainable)))
    expected_aggregator_type = computation_types.to_type(
        collections.OrderedDict(value_sum_process=(), weight_sum_process=()))
    expected_state_type = computation_types.at_server(
        (expected_optimizer_state_type, expected_aggregator_type))
    expected_measurements_type = computation_types.at_server(
        collections.OrderedDict(
            train=collections.OrderedDict(
                loss=tf.float32, num_examples=tf.int32),
            stat=collections.OrderedDict(num_examples=tf.int64)))

    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type)
    expected_initialize_type.check_equivalent_to(
        client_work_process.initialize.type_signature)

    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            weights=expected_param_model_weights_type,
            client_data=expected_param_data_type),
        result=measured_process.MeasuredProcessOutput(
            expected_state_type, expected_result_type,
            expected_measurements_type))
    expected_next_type.check_equivalent_to(
        client_work_process.next.type_signature)

  def test_created_model_raises(self):
    with self.assertRaises(TypeError):
      mime._build_mime_lite_client_work(
          model_examples.LinearRegression(),
          sgdm.build_sgdm(1.0),
          client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES)

  def test_keras_optimizer_raises(self):
    with self.assertRaises(TypeError):
      mime._build_mime_lite_client_work(
          model_examples.LinearRegression,
          lambda: tf.keras.optimizers.SGD(1.0),
          client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES)

  def test_unweighted_full_gradient_aggregator_raises(self):
    with self.assertRaises(TypeError):
      mime._build_mime_lite_client_work(
          model_examples.LinearRegression(),
          sgdm.build_sgdm(1.0),
          client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
          full_gradient_aggregator=mean.UnweightedMeanFactory())


class MimeLiteClientWorkExecutionTest(test_case.TestCase,
                                      parameterized.TestCase):

  def create_dataset(self):
    # Create a dataset with 4 examples:
    dataset = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            y=[[0.0], [0.0], [1.0], [1.0]]))
    # Repeat the dataset 2 times with batches of 3 examples,
    # producing 3 minibatches (the last one with only 2 examples).
    # Note that `batch` is required for this dataset to be useable,
    # as it adds the batch dimension which is expected by the model.
    return dataset.repeat(2).batch(3)

  def create_model(self):
    return model_examples.LinearRegression(feature_dim=2)

  def initial_weights(self):
    return model_utils.ModelWeights(
        trainable=[tf.zeros((2, 1)), tf.constant(0.0)],
        non_trainable=[0.0],
    )

  @parameterized.named_parameters(('non-simulation', False),
                                  ('simulation', True))
  @mock.patch.object(
      dataset_reduce,
      '_dataset_reduce_fn',
      wraps=dataset_reduce._dataset_reduce_fn)
  def test_client_tf_dataset_reduce_fn(self, simulation, mock_method):
    process = mime._build_mime_lite_client_work(
        model_fn=self.create_model,
        optimizer=sgdm.build_sgdm(learning_rate=0.1, momentum=0.9),
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        use_experimental_simulation_loop=simulation)
    client_data = [self.create_dataset()]
    client_model_weights = [self.initial_weights()]
    process.next(process.initialize(), client_model_weights, client_data)
    if simulation:
      mock_method.assert_not_called()
    else:
      mock_method.assert_called()

  @parameterized.named_parameters(
      ('adagrad', adagrad.build_adagrad(0.1)), ('adam', adam.build_adam(0.1)),
      ('rmsprop', rmsprop.build_rmsprop(0.1)), ('sgd', sgdm.build_sgdm(0.1)),
      ('sgdm', sgdm.build_sgdm(0.1, momentum=0.9)),
      ('yogi', yogi.build_yogi(0.1)))
  def test_execution_with_optimizer(self, optimizer):
    process = mime._build_mime_lite_client_work(
        self.create_model,
        optimizer,
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES)
    client_data = [self.create_dataset()]
    client_model_weights = [self.initial_weights()]
    state = process.initialize()
    output = process.next(state, client_model_weights, client_data)
    self.assertEqual(8, output.measurements['stat']['num_examples'])

  def test_custom_metrics_aggregator(self):

    def sum_then_finalize_then_times_two(metric_finalizers,
                                         local_unfinalized_metrics_type):

      @computations.federated_computation(
          computation_types.at_clients(local_unfinalized_metrics_type))
      def aggregation_computation(client_local_unfinalized_metrics):
        unfinalized_metrics_sum = intrinsics.federated_sum(
            client_local_unfinalized_metrics)

        @computations.tf_computation(local_unfinalized_metrics_type)
        def finalizer_computation(unfinalized_metrics):
          finalized_metrics = collections.OrderedDict()
          for metric_name, metric_finalizer in metric_finalizers.items():
            finalized_metrics[metric_name] = metric_finalizer(
                unfinalized_metrics[metric_name]) * 2
          return finalized_metrics

        return intrinsics.federated_map(finalizer_computation,
                                        unfinalized_metrics_sum)

      return aggregation_computation

    process = mime._build_mime_lite_client_work(
        model_fn=self.create_model,
        optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        metrics_aggregator=sum_then_finalize_then_times_two)
    client_model_weights = [self.initial_weights()]
    client_data = [self.create_dataset()]
    output = process.next(process.initialize(), client_model_weights,
                          client_data)
    self.assertEqual(output.measurements['stat']['num_examples'], 8)
    # Train metrics should be multiplied by two by the custom aggregator.
    self.assertEqual(output.measurements['train']['num_examples'], 16)


class FedAvgTest(test_case.TestCase, parameterized.TestCase):
  """Tests construction of the FedAvg training process."""

  def test_construction_calls_model_fn(self):
    # Assert that the the process building does not call `model_fn` too many
    # times. `model_fn` can potentially be expensive (loading weights,
    # processing, etc).
    mock_model_fn = mock.Mock(side_effect=model_examples.LinearRegression)
    mime.build_weighted_mime_lite(
        model_fn=mock_model_fn,
        optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9))
    self.assertEqual(mock_model_fn.call_count, 3)

  @parameterized.named_parameters(
      ('non-simulation_tff_optimizer', False),
      ('simulation_tff_optimizer', True),
  )
  @mock.patch.object(
      dataset_reduce,
      '_dataset_reduce_fn',
      wraps=dataset_reduce._dataset_reduce_fn)
  def test_client_tf_dataset_reduce_fn(self, simulation, mock_method):
    mime.build_weighted_mime_lite(
        model_fn=model_examples.LinearRegression,
        optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
        use_experimental_simulation_loop=simulation)
    if simulation:
      mock_method.assert_not_called()
    else:
      mock_method.assert_called()

  @mock.patch.object(mime, 'build_weighted_mime_lite')
  def test_build_weighted_mime_lite_called_by_unweighted_mime_lite(
      self, mock_mime_lite):
    mime.build_unweighted_mime_lite(
        model_fn=model_examples.LinearRegression,
        optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9))
    self.assertEqual(mock_mime_lite.call_count, 1)

  @mock.patch.object(mime, 'build_weighted_mime_lite')
  @mock.patch.object(aggregation, 'as_weighted_aggregator')
  def test_aggregation_wrapper_called_by_unweighted(self, _, mock_as_weighted):
    mime.build_unweighted_mime_lite(
        model_fn=model_examples.LinearRegression,
        optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9))
    self.assertEqual(mock_as_weighted.call_count, 1)

  def test_raises_on_non_callable_model_fn(self):
    with self.assertRaises(TypeError):
      mime.build_weighted_mime_lite(
          model_fn=model_examples.LinearRegression(),
          optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9))

  def test_raises_on_invalid_client_weighting(self):
    with self.assertRaises(TypeError):
      mime.build_weighted_mime_lite(
          model_fn=model_examples.LinearRegression,
          optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
          client_weighting='uniform')

  def test_raises_on_invalid_distributor(self):
    model_weights_type = type_conversions.type_from_tensors(
        model_utils.ModelWeights.from_model(model_examples.LinearRegression()))
    distributor = distributors.build_broadcast_process(model_weights_type)
    invalid_distributor = iterative_process.IterativeProcess(
        distributor.initialize, distributor.next)
    with self.assertRaises(TypeError):
      mime.build_weighted_mime_lite(
          model_fn=model_examples.LinearRegression,
          optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
          model_distributor=invalid_distributor)

  def test_weighted_raises_on_unweighted_aggregator(self):
    aggregator = model_update_aggregator.robust_aggregator(weighted=False)
    with self.assertRaisesRegex(TypeError, 'WeightedAggregationFactory'):
      mime.build_weighted_mime_lite(
          model_fn=model_examples.LinearRegression,
          optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
          model_aggregator=aggregator)
    with self.assertRaisesRegex(TypeError, 'WeightedAggregationFactory'):
      mime.build_weighted_mime_lite(
          model_fn=model_examples.LinearRegression,
          optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
          full_gradient_aggregator=aggregator)

  def test_unweighted_fed_avg_raises_on_weighted_aggregator(self):
    aggregator = model_update_aggregator.robust_aggregator(weighted=True)
    with self.assertRaisesRegex(TypeError, 'UnweightedAggregationFactory'):
      mime.build_unweighted_mime_lite(
          model_fn=model_examples.LinearRegression,
          optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
          model_aggregator=aggregator)
    with self.assertRaisesRegex(TypeError, 'UnweightedAggregationFactory'):
      mime.build_unweighted_mime_lite(
          model_fn=model_examples.LinearRegression,
          optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
          full_gradient_aggregator=aggregator)


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  test_case.main()
