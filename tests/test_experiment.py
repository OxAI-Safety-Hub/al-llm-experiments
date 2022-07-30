import sys
import io

from al_llm.experiment import Experiment


class ZeroStringIO(io.TextIOBase):
    """Simple I/O stream which returns infinite lines of '0'"""

    def readline(size=-1, /):
        return "0"


class TestDummyExperiment:
    """Run the dummy experiment, repeatedly feeding '0' as the input"""

    def test_dummy_experiment(self):
        dummy_args = Experiment.make_dummy_experiment()
        experiment = Experiment(**dummy_args)
        sys.stdin = ZeroStringIO()
        experiment.run()

    def setup_method(self):
        self.orig_stdin = sys.stdin

    def teardown_method(self):
        sys.stdin = self.orig_stdin
