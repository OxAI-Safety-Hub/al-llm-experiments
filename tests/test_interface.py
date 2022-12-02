import sys
import io
import itertools

from sklearn.model_selection import ParameterGrid

from al_llm.parameters import Parameters
from al_llm.experiment import Experiment
from al_llm.constants import WANDB_PROJECTS
from al_llm.utils import UnlabelledSamples


class CycleStringIO(io.TextIOBase):
    """Simple I/O stream which repeatidly cycles through an iterator"""

    def __init__(self, iterable):
        super().__init__()
        self.cycler = itertools.cycle(iterable)

    def readline(self, size=-1):
        return next(self.cycler)


def test_pool_simulator_interface():

    # Make a dummy experiment using PoolSimulatorInterface
    parameters = Parameters(
        sample_generator_base_model="pool", full_loop=True, is_running_pytests=True
    )
    args = Experiment.make_experiment(parameters, WANDB_PROJECTS["sandbox"], "test")
    experiment = Experiment(**args)

    # Run it, to make sure there are no errors
    experiment.run()

    # Check that the prompt method returns the correct labels
    dataset_train = args["dataset_container"].dataset_remainder
    initial_dataset_slice = dataset_train.with_format(None)
    samples = UnlabelledSamples(initial_dataset_slice["text"][:10])
    samples.suggested_labels = initial_dataset_slice["labels"][:10]
    prompt_output = args["interface"].prompt(samples)
    assert prompt_output.labels == samples.suggested_labels


class TestCLIInterface:
    """Test the CLI interface with different amgibuity and skip settings"""

    def test_dummy_experiment(self):

        # The number of samples to use in each test
        num_samples = 20

        # Loop over different combinations of ambiguity and skip settings
        extra_parameters_grid = {
            "ambiguity_mode": ["none", "only_mark"],
            "allow_skipping": [False, True],
        }
        for extra_parameters in ParameterGrid(extra_parameters_grid):

            # Create a dummy experiment using these parametes
            parameters = Parameters(
                full_loop=True,
                is_running_pytests=True,
                num_samples=num_samples,
                **extra_parameters
            )
            args = Experiment.make_experiment(
                parameters, WANDB_PROJECTS["sandbox"], "test"
            )
            experiment = Experiment(**args)

            # Determine the prompt inputs
            num_categories = len(experiment.dataset_container.categories)
            num_options = num_categories
            if extra_parameters_grid["ambiguity_mode"] == "only_mark":
                num_options *= 2
            if extra_parameters_grid["allow_skipping"]:
                num_options += 1
            prompt_inputs = [str(i) for i in range(num_options)]

            # Determine the expected resulting labels, ambiguities and skip
            # mask
            expected_output = [(i, 0, 0) for i in range(num_categories)]
            if extra_parameters_grid["ambiguity_mode"] == "only_mark":
                expected_output.extend([(i, 1, 0) for i in range(num_categories)])
            if extra_parameters_grid["allow_skipping"]:
                expected_output.append((0, 0, 1))

            # Run the prompt feeding the prompt inputs to STDIN, cycling
            # through them for as long as necessary
            sys.stdin = CycleStringIO(prompt_inputs)
            samples = experiment.sample_generator.generate()
            prompt_output = experiment.interface.prompt(samples)

            # Make sure that the prompt output aligns with what we expect
            iterator = zip(
                expected_output,
                prompt_output.labels,
                prompt_output.ambiguities,
                prompt_output.skip_mask,
            )
            category_keys = list(experiment.dataset_container.categories.keys())
            for expected, label, amgibuity, skip in iterator:
                print(expected, label, amgibuity, skip)
                assert label == category_keys[expected[0]]
                assert amgibuity == expected[1]
                assert skip == expected[2]

    def setup_method(self):
        self.orig_stdin = sys.stdin

    def teardown_method(self):
        sys.stdin = self.orig_stdin
