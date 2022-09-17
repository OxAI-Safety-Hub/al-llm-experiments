from al_llm.classifier import HuggingFaceClassifier, UncertaintyMixin


def test_subclassing():

    # All Hugging Face classifiers must provide an uncertainty measure to
    # ensure that TokenByTokenSampleGenerator works
    assert issubclass(HuggingFaceClassifier, UncertaintyMixin)
