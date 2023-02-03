import torch
import random
from torch import optim
import pytorch_lightning as pl

from typing import Any, List, Optional, Tuple
from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.models.components.decoder_attn_rnn import AttnDecoderRNN
from src.models.components.encoder_rnn import EncoderRNN


class MessageEventEncoderLitModule(pl.LightningModule):
    def __init__(
        self,
        vector_size,
        hidden_size,
        max_length,
        START,
        END,
        optimizer: optim.Optimizer,
        tf_rate: float = 0.8,
    ):
        super().__init__()
        self.input_size = self.output_size = vector_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.START = START
        self.END = END
        self.tf_rate = tf_rate

        self.optimizer = optimizer
        self.encoder = EncoderRNN(self.input_size, self.hidden_size)
        self.decoder = AttnDecoderRNN(
            self.hidden_size, self.output_size, max_length=self.max_length
        )
        self.metric = torch.nn.NLLLoss()
        self.save_hyperparameters()

    def encode(self, input) -> torch.Tensor:
        input_tensor = input
        encoder_hidden = self.encoder.initHidden()
        input_length = input_tensor.size(0)
        encoder_outputs = torch.zeros(
            self.max_length, self.encoder.hidden_size, dtype=torch.float
        )
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden
            )
            encoder_outputs[ei] = encoder_output[0, 0]

        return encoder_outputs

    def decode(
        self, input, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        decoder_input = self.START
        decoder_hidden = self.decoder.initHidden()
        decoder_outputs = list()

        loss = torch.tensor(0, dtype=torch.float)
        count = 0
        for di in range(self.max_length):
            decoder_output, decoder_hidden, _ = self.decoder(
                decoder_input, decoder_hidden, input
            )
            _, top_i = decoder_output.topk(1)
            decoder_input = top_i.squeeze()

            if target is not None:
                target_i = (
                    target[di] if di < target.size(0) else torch.tensor([self.END])
                )
                loss += self.metric(decoder_output, target_i)

            decoder_outputs.append(top_i.item())
            count += 1
            if top_i.item() == self.END:
                break

        if target is not None:
            loss = loss / count
        return torch.tensor(decoder_outputs), loss

    def forward(self, input) -> Tuple[List, torch.Tensor, torch.Tensor]:
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

        decoder_input = self.START
        decoder_hidden = self.decoder.initHidden()
        decoder_outputs = list()

        count = 0
        for di in range(self.max_length):
            decoder_output, decoder_hidden, _ = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            _, top_i = decoder_output.topk(1)
            decoder_input = top_i.squeeze().detach()

            target = (
                input_tensor[di]
                if di < input_tensor.size(0)
                else torch.tensor([self.END])
            )
            loss += self.metric(decoder_output, target)
            decoder_outputs.append(top_i.item())
            count += 1
            if top_i.item() == self.END:
                break

        assert count != 0, "count can not be 0"
        loss = loss / count
        return decoder_outputs, loss, encoder_outputs

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        input_tensor = batch
        assert input_tensor.size(0) == 1, "batch has to be of size 1"
        input_tensor = input_tensor[0]
        target_tensor = input_tensor

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

        decoder_input = self.START
        decoder_hidden = self.decoder.initHidden()

        teacher_forcing = random.random() < self.tf_rate
        count = 0
        if teacher_forcing:
            for di in range(input_length):
                decoder_output, decoder_hidden, _ = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                loss += self.metric(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]
                count += 1
        else:
            for di in range(input_length):
                decoder_output, decoder_hidden, _ = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                _, top_i = decoder_output.topk(1)
                decoder_input = top_i.squeeze().detach()

                loss += self.metric(decoder_output, target_tensor[di])
                count += 1
                if decoder_input.item() == self.END:
                    break

        assert count != 0, "count can not be 0"
        loss = loss / count
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx) -> None:
        input_tensor = batch
        _, loss, _ = self.forward(input_tensor)
        self.log("test_loss", loss)

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        _, loss, _ = self.forward(batch)
        self.log("val_loss", loss)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        input_tensor = batch
        decoder_outputs, loss, _ = self.forward(input_tensor)
        return (decoder_outputs, input_tensor.squeeze().tolist(), loss)

    def configure_optimizers(self):
        return self.optimizer(self.parameters())  # type: ignore
