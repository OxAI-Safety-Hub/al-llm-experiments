import sys
import io

from al_llm.experiment import Experiment
from al_llm.parameters import Parameters
from al_llm.constants import WANDB_PROJECTS


class ZeroStringIO(io.TextIOBase):
    """Simple I/O stream which returns infinite lines of '0'"""

    def readline(size=-1):
        return "0"


class TestFullLoopDummyExperiment:
    """Run the full dummy experiment, repeatedly feeding '0' as the input"""

    def test_dummy_experiment(self):
        parameters = Parameters(full_loop=True, is_running_pytests=True)
        dummy_args = Experiment.make_experiment(
            parameters, WANDB_PROJECTS["sandbox"], "test"
        )
        experiment = Experiment(**dummy_args)
        sys.stdin = ZeroStringIO()
        experiment.run()

    def setup_method(self):
        self.orig_stdin = sys.stdin

    def teardown_method(self):
        sys.stdin = self.orig_stdin
