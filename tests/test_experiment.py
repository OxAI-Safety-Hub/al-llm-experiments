import sys
import io

from al_llm.experiment import Experiment
from al_llm.parameters import Parameters


class ZeroStringIO(io.TextIOBase):
    """Simple I/O stream which returns infinite lines of '0'"""

    def readline(size=-1):
        return "0"


class TestFullLoopDummyExperiment:
    """Run the full dummy experiment, repeatedly feeding '0' as the input"""

    def test_dummy_experiment(self):
        parameters = Parameters()
        dummy_args = Experiment.make_experiment(
            parameters, "test", is_running_pytests=True
        )
        experiment = Experiment(**dummy_args)
        sys.stdin = ZeroStringIO()
        experiment.run_full()

    def setup_method(self):
        self.orig_stdin = sys.stdin

    def teardown_method(self):
        sys.stdin = self.orig_stdin
