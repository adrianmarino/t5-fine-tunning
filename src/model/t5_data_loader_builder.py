import torch
from torch.utils.data import Dataset, DataLoader


class T5DataLoaderBuilder():
    def __init__(self, tokenizer, hparams, create_fn):
        super().__init__()
        self.__tokenizer  = tokenizer
        self.__create_fn  = create_fn
        self.__hparams    = hparams

    def val(self): return self.__build(
      self.__hparams.eval_path,
      self.__hparams.eval_batch_size
    )

    def train(self): return self.__build(
        self.__hparams.train_path,
        self.__hparams.train_batch_size,
        drop_last   = True,
        shuffle     = True
    )

    def test(self): return self.__build(
        self.__hparams.test_path,
        self.__hparams.test_batch_size,
        shuffle     = True
    )

    def __build(
        self,
        type_path,
        batch_size,
        num_workers = 4,
        drop_last   = False,
        shuffle     = False
    ):
        return DataLoader(
            self.__create_fn(self.__tokenizer, type_path, self.__hparams),
            batch_size  = batch_size,
            num_workers = num_workers,
            shuffle     = shuffle,
            drop_last  = drop_last
        )