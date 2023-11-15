import torch
from torch.optim import Adam
import pytorch_lightning as pl


class T5FineTuner(pl.LightningModule):
    def __init__(self, model, dl_builder):
        super().__init__()
        self.model        = model
        self.__dl_builder = dl_builder

    def forward(
        self,
        input_ids,
        attention_mask         = None,
        decoder_input_ids      = None,
        decoder_attention_mask = None,
        labels                 = None
    ):
        return self.model(
            input_ids,
            attention_mask         = attention_mask,
            decoder_input_ids      = decoder_input_ids,
            decoder_attention_mask = decoder_attention_mask,
            labels                 = labels
        )

    def _get_batch_loss(self, batch, metric):
        labels = batch["target_ids"]
        labels[labels[:, :] == self.model.tokenizer.pad_token_id] = -100

        loss = self(
            input_ids               = batch["source_ids"],
            attention_mask          = batch["source_mask"],
            decoder_attention_mask  = batch["target_mask"],
            labels                  = labels
        ).loss
        self.log(metric, loss, prog_bar=True)
        return loss

    def is_logger(self):
        return self.trainer.global_rank <= 0

    def training_step(self, batch, batch_idx):
        return self._get_batch_loss(batch, 'train_loss')

    def validation_step(self, batch, batch_idx):
        return self._get_batch_loss(batch, 'val_loss')

    def configure_optimizers(self):
        return [Adam(self.model.parameters(), self.model.hyper_hparams.lr)]

    def val_dataloader(self):
        return self.__dl_builder.train()

    def train_dataloader(self):
        return self.__dl_builder.train()
