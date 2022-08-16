from al_llm import Experiment

dummy_args = Experiment.make_dummy_experiment("dummy")
experiment = Experiment(**dummy_args)
experiment.run_full()
