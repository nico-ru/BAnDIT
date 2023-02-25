import torch
import random
import pytorch_lightning as pl
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import optim

from src.models.components.decoder_attn_gru import AttnDecoderGRU
from src.models.components.encoder_gru import EncoderGRU


class SequenceRNN(pl.LightningModule):
    def __init__(
        self,
        embedding_size,
        embedding_dimension,
        max_length,
        optimizer: optim.Optimizer,
        tf_rate: float = 0.8,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding_dimension = embedding_dimension
        self.max_length = max_length
        self.optimizer = optimizer
        self.tf_rate = tf_rate

        self.encoder = EncoderGRU(self.embedding_size)
        self.decoder = AttnDecoderGRU(self.embedding_size, max_length=self.max_length)
        self.metric = torch.nn.CosineEmbeddingLoss()
        self.save_hyperparameters()

    def forward(self, input, tf: bool = False):
        assert input.size(0) == 1, "batch has to be of size 1"
        input_tensor = input[0]

        encoder_hidden = self.encoder.initHidden()
        input_length = input_tensor.size(0)
        encoder_outputs = torch.zeros(
            self.max_length, self.encoder.hidden_size, dtype=torch.float
        )
        loss = torch.tensor(0, dtype=torch.float)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden
            )
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.zeros(self.embedding_size)
        decoder_hidden = self.decoder.initHidden()
        decoder_outputs = list()
        probabilities = list()

        count = 0
        for di in range(self.max_length):
            decoder_output, decoder_hidden, _ = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            target = (
                input_tensor[di]
                if di < input_tensor.size(0)
                else torch.ones(self.embedding_size)
            )
            target = target.view(self.embedding_dimension, -1)
            decoder_output = decoder_output.view(self.embedding_dimension, -1)
            target_labels = torch.ones(decoder_output.size(0))
            loss += self.metric(decoder_output, target, target_labels)

            if tf and random.random() < self.tf_rate:
                decoder_input = target
            else:
                decoder_input = decoder_output

            decoder_outputs.append(decoder_output)
            count += 1

        assert count != 0, "count can not be 0"
        loss = loss / count
        return decoder_outputs, loss, encoder_outputs, probabilities

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        _, loss, *_ = self.forward(batch, tf=True)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx) -> None:
        _, loss, *_ = self.forward(batch)
        self.log("test_loss", loss)

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        _, loss, *_ = self.forward(batch)
        self.log("val_loss", loss)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        input_tensor = batch
        decoder_outputs, loss, *_ = self.forward(input_tensor)
        return (decoder_outputs, input_tensor.squeeze().tolist(), loss)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())  # type: ignore
        return optimizer
