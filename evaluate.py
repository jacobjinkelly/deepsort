"""For performing inference using the model.
"""

import torch

from utils import device, SOS_token, EOS_token, MAX_LENGTH


# noinspection PyCallingNonCallable
def evaluate(encoder, decoder, input_tensor, is_ptr, max_length=MAX_LENGTH):
    """
    Perform inference using the model.
    """
    with torch.no_grad():
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(input_length, encoder.hidden_dim, device=device)

        for i in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
            encoder_outputs[i] += encoder_output[0, 0]

        decoder_input, decoder_hidden = torch.tensor([[SOS_token]], device=device), encoder_hidden

        decoded_output = []
        decoder_attentions = torch.zeros(max_length, input_length)

        for i in range(max_length):
            args = (decoder_input, decoder_hidden, encoder_outputs)
            if is_ptr:
                args += (input_tensor,)
            decoder_output, decoder_hidden, decoder_attention = decoder(*args)
            # decoder_attentions[i] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_output.append('<EOS>')
                break
            else:
                decoded_output.append(topi.item())

            decoder_input = topi.squeeze().detach()

        return decoded_output, decoder_attentions[:len(decoded_output) + 1]
