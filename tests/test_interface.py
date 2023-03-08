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
        dataset_name="dummy",
        acquisition_function="dummy",
        sample_generator_base_model="pool",
        full_loop=True,
        is_running_pytests=True,
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
                dataset_name="dummy",
                acquisition_function="dummy",
                full_loop=True,
                is_running_pytests=True,
                num_samples=num_samples,
                num_iterations=1,
                dev_mode=True,
                **extra_parameters
            )

            args = Experiment.make_experiment(
                parameters, WANDB_PROJECTS["sandbox"], "test"
            )

            experiment = Experiment(**args)

            # Determine the prompt inputs
            num_categories = len(experiment.dataset_container.categories)

            # Calculate how many options the user is presented
            num_user_options = num_categories

            if extra_parameters["ambiguity_mode"] == "only_mark":
                # Each option has an unambiguous and ambiguous version
                num_user_options *= 2

            if extra_parameters["allow_skipping"]:
                # There is an extra option for "skipping"
                num_user_options += 1

            # List all of the ways the user could respond to a prompt
            prompt_user_inputs = [str(i) for i in range(num_user_options)]

            # Run the prompt feeding the prompt inputs to STDIN, cycling
            # through them for as long as necessary
            sys.stdin = CycleStringIO(prompt_user_inputs)
            samples = experiment.sample_generator.generate()
            prompt_output = experiment.interface.prompt(samples)

            # Make sure that the returned lists have the correct length
            assert num_samples == len(prompt_output.labels)
            assert num_samples == len(prompt_output.ambiguities)
            assert num_samples == len(prompt_output.skip_mask)

            category_labels = list(experiment.dataset_container.categories.keys())

            for i in range(num_samples):
                # Get the input number which is fed to the prompt
                prompt_input_number = i % len(prompt_user_inputs)

                actual_label = prompt_output.labels[i]
                actual_ambiguity = prompt_output.ambiguities[i]
                actual_skip = prompt_output.skip_mask[i]

                if 0 <= prompt_input_number < num_categories:
                    # The first num_categories options are just plain categories
                    assert actual_label == category_labels[prompt_input_number]
                    assert actual_ambiguity == 0
                    assert actual_skip == 0
                elif (
                    extra_parameters["ambiguity_mode"] == "only_mark"
                    and num_categories <= prompt_input_number < num_categories * 2
                ):
                    # The label returned will be category + num_categories
                    assert (
                        actual_label
                        == category_labels[prompt_input_number - num_categories]
                    )
                    assert actual_ambiguity == 1
                    assert actual_skip == 0
                elif (
                    extra_parameters["allow_skipping"]
                    and prompt_input_number == len(prompt_user_inputs) - 1
                ):
                    # If skipping is on, the last option is for skip
                    assert actual_label == category_labels[0]
                    assert actual_ambiguity == 0
                    assert actual_skip == 1
                else:
                    raise ValueError("This case shouldn't be reached")

    def setup_method(self):
        self.orig_stdin = sys.stdin

    def teardown_method(self):
        sys.stdin = self.orig_stdin
