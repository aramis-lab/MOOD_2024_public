from typing import Any, Dict, Iterable, Tuple, Union

from torch.utils.data import ConcatDataset, Dataset, StackDataset

from clinicadl.utils.caps_dataset.data import CapsDataset


class CapsConcatDataset(ConcatDataset):
    """
    Concatenation of CapsDatasets.

    The only condition is that the CapsDatasets
    must have the same mode (e.g. image or patch).

    Parameters
    ----------
    datasets : Iterable[CapsDataset]
        The datasets to concatenate.
    """

    def __init__(self, datasets: Iterable[CapsDataset]) -> None:
        mode = [d.mode for d in self.datasets]
        if all(i == mode[0] for i in mode):
            self.mode = mode[0]
        else:
            raise AttributeError(
                "All the CapsDataset must have the same mode: 'image','patch','roi','slice', etc."
            )
        super().__init__(*datasets)


class CapsPairedDataset(StackDataset):
    """
    A stack of CapsDatasets.

    It is a bijection between datasets, so
    the only requirement is that they have the same lengths.

    Parameters
    ----------
    *args : CapsDataset
        To stack datasets as tuple.
    **kwargs : CapsDataset
        To stack dataset as dict.

    See: https://pytorch.org/docs/stable/data.html#torch.utils.data.StackDataset
    """

    def __init__(
        self, *args: CapsDataset, **kwargs: CapsDataset
    ) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(self.datasets, dict):
            self.mode = {name: d.mode for name, d in self.datasets.items()}
        else:
            self.mode = tuple([d.mode for d in self.datasets])


class CapsUnpairedDataset(Dataset):
    """
    A stack of CapsDataset that can have different lengths.

    It is a injection between datasets.
    There is no requirement on the datasets (e.g. on mode, modality or whatsoever).
    Otherwise, works like CapsPairedDataset.

    Parameters
    ----------
    *args : CapsDataset
        To stack datasets as tuple.
    **kwargs : CapsDataset
        To stack dataset as dict.
    """

    def __init__(
        self, *args: CapsDataset, **kwargs: CapsDataset
    ) -> None:
        if args:
            if kwargs:
                raise ValueError("Supports either args or kwargs, but both were given.")
            self.datasets = args
            self.mode = tuple([d.mode for d in self.datasets])
        elif kwargs:
            self.datasets = kwargs
            self.mode = {name: d.mode for name, d in self.datasets.items()}
        else:
            raise ValueError("At least one dataset should be passed")

    def __len__(self) -> Union[Tuple[int, ...], Dict[Any, int]]:
        """
        Gets the lengths of the datasets.
    
        Returns
        -------
        Union[Tuple[int, ...], Dict[Any, int]]
            The data corresponding to the indices.
            If the datasets were passed as args, data will be returned as a tuple.
            If the datasets were passed as kwargs, data will be returned as a dict.
        """
        if isinstance(self.datasets, dict):
            return {name: len(d) for name, d in self.datasets.items()}
        else:
            return tuple(len(d) for d in self.datasets)

    def __getitem__(
        self, indices: Union[Tuple[int, ...], Dict[Any, int]]
    ) -> Union[Tuple[int, ...], Dict[Any, int]]:
        """
        Gets data from the collections of datasets.

        Parameters
        ----------
        indices : Union[Tuple[int, ...], Dict[Any, int]]
            The indices to get.
            Must be a tuple if the datasets were passed as a tuple.
            Must be a dict if the datasets were passed as a dict.

        Returns
        -------
        Union[Tuple[int, ...], Dict[Any, int]]
            The data corresponding to the indices.
            If the datasets were passed as a tuple, data will be returned as a tuple.
            If the datasets were passed as a dict, data will be returned as a dict.

        Raises
        ------
        ValueError
            If the number of indices doesn't match the number of datasets.
        ValueError
            If an index is greater that the length of the corresponding dataset.
        ValueError
            If indices were passed as a dict, but the names don't match those of the datasets.
        ValueError
            If indices were passed as a dict, whereas the datasets were passed as a tuple, or inversely.
        """
        if len(indices) != len(self.datasets):
            raise ValueError("The number of indexes must match the number of datasets.")

        if isinstance(indices, dict):
            if not isinstance(self.datasets, dict):
                raise ValueError(
                    "You can't pass a dict for indices if datasets were not passed as a dict."
                )
            if indices.keys() != self.datasets.keys():
                raise ValueError(
                    "The keys for the indices must match the keys passed for the datasets."
                )
            if all(indices[name] < len(self.datasets[name]) for name in indices):
                raise ValueError(
                    "The indexes must be inferior than the length of each dataset"
                )
            return {name: dataset[indices[name]] for name, dataset in self.datasets.items()}
        else:
            if isinstance(self.datasets, dict):
                raise ValueError(
                    "You must pass a dict for the indices if the datasets were passed as a dict."
                )
            if all(index < len(dataset) for index, dataset in zip(indices, self.datasets)):
                raise ValueError(
                    "The indexes must be inferior than the length of each dataset"
                )
            return tuple(dataset[idx] for idx, dataset in zip(indices, self.datasets))
