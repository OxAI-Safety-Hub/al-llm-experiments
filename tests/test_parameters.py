from al_llm.parameters import Parameters


def test_parameter_constructor():
    # Create a parameters with mostly defaults.
    _ = Parameters(dataset_name="dummy", acquisition_function="dummy")
