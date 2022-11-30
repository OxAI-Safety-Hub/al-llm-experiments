import pytest

from al_llm.utils import UnlabelledSamples


def test_unlabelled_samples():

    # Create a basic test set of samples
    samples = UnlabelledSamples([str(i) for i in range(10)])

    # Make sure we get an AttributeError when trying to access the suggested
    # labels
    with pytest.raises(AttributeError):
        samples.suggested_labels

    # Make sure we get back `UnlabelledSamples` objects when indexing
    assert isinstance(samples[:4], UnlabelledSamples)
    assert isinstance(samples[1:6:2], UnlabelledSamples)

    # Make sure we can index by lists
    assert isinstance(samples[[1, 3, 5]], UnlabelledSamples)
    assert samples[[1, 3, 5]] == ["1", "3", "5"]

    # Try to add suggest labels of the wrong length
    with pytest.raises(ValueError):
        samples.suggested_labels = [1, 2, 4]
    with pytest.raises(ValueError):
        samples.suggested_labels = list(range(20))

    # Add suggested labels of the correct length
    samples.suggested_labels = [i % 3 for i in range(10)]

    # Try to get them again
    assert samples.suggested_labels is not None

    # Make sure that indexing simultaneously indexes the suggested labels. In
    # other words, indexing should be commutative with getting the suggested
    # labels
    index = slice(3, 8)
    assert samples[index].suggested_labels == samples.suggested_labels[index]
    index = [3, 6, 7]
    assert samples[index].suggested_labels == [
        samples.suggested_labels[i] for i in index
    ]
