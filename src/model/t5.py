from transformers import T5ForConditionalGeneration, T5Tokenizer

import pytorch_lightning as pl
import model as ml


class T5(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hyper_hparams = hparams
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.hyper_hparams.model_name_or_path
        )
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.hyper_hparams.tokenizer_name_or_path,
            truncation=self.hyper_hparams.tokenizer_truncation,
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def decode(self, inputs, sanitize=True):
        outputs = []
        for _input in inputs:
            output = self.tokenizer.decode(_input)
            if sanitize:
                output = ml.sanitize(output)
            outputs.append(output)
        return outputs

    def parameters(self):
        return self.model.parameters()

    def predict(self, batch, max_length=2):
        outs = self.model.generate(
            input_ids      = batch['source_ids'],
            attention_mask = batch['source_mask'],
            max_length     = max_length,
        )
        return self.decode(outs)

