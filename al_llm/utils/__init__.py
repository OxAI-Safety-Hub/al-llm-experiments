from typing import Optional

from .fake_data import FakeSentenceGenerator, FakeLabelGenerator


class UnlabelledSamples(list):
    """List-like object which holds unlabelled samples

    Can optionally hold a list of the suggested labels for the samples. This
    is produced for example when doing pool-based sampling.

    Extends list functionality by allowing indexing by a list of indices.

    Attributes
    ----------
    suggested_labels : list, optional
        The list of suggested labels
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._suggested_labels: Optional[list] = None

    @property
    def suggested_labels(self):
        if self._suggested_labels is None:
            raise AttributeError("Suggested labels are not set")
        return self._suggested_labels

    @suggested_labels.setter
    def suggested_labels(self, labels: list):
        if len(labels) != len(self):
            raise ValueError(
                f"Paramater `labels` must have same length as `self`, but "
                f"{len(labels)} != {len(self)}."
            )
        self._suggested_labels = labels

    def __getitem__(self, key):

        # Allow indexing by a list of indices
        if isinstance(key, list):
            new_unlabelled_samples = UnlabelledSamples(
                [super(UnlabelledSamples, self).__getitem__(i) for i in key]
            )
            try:
                new_unlabelled_samples.suggested_labels = [
                    self.suggested_labels[i] for i in key
                ]
            except AttributeError:
                pass
            return new_unlabelled_samples

        # If we're slicing the samples, also slice the labels if available
        elif isinstance(key, slice):
            new_unlabelled_samples = UnlabelledSamples(super().__getitem__(key))
            try:
                new_unlabelled_samples.suggested_labels = self.suggested_labels[key]
            except AttributeError:
                pass
            return new_unlabelled_samples

        # Otherwise following the normal getitem procedure
        return super().__getitem__(key)
