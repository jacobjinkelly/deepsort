"""For training models.
"""
import time
import random

import torch
import torch.nn as nn

from utils import device, SOS_token, EOS_token, time_since, save_checkpoint, load_checkpoint
from visual import show_plot


# ignore noncallable/unresolvedreferences errors for torch.tensor, torch.unsqueeze respectively (bug in PyTorch)
# noinspection PyCallingNonCallable,PyUnresolvedReferences
def train(input_tensor, target_tensor, encoder, decoder, encoder_optim, decoder_optim, is_ptr, criterion,
          teacher_force):
    """
    One step in the training loop.
    """
    encoder_hidden = encoder.init_hidden()

    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(input_length, encoder.hidden_dim, device=device)

    loss = 0

    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
        encoder_outputs[i] = encoder_output[0, 0]

    decoder_input, decoder_hidden = torch.tensor([[SOS_token]], device=device), encoder_hidden

    use_teacher_force = random.random() < teacher_force

    for i in range(target_length):
        args = (decoder_input, decoder_hidden, encoder_outputs)
        if is_ptr:
            args += (input_tensor,)
        decoder_output, decoder_hidden, _ = decoder(*args)
        if not use_teacher_force:
            topv, topi = decoder_output.topk(1)
            # detach from history as input
            decoder_input = topi.squeeze().detach()
        else:
            decoder_input = target_tensor[i]
        loss += criterion(decoder_output, target_tensor[i])

        if not use_teacher_force and decoder_input.item() == EOS_token:
            break

    loss.backward()

    encoder_optim.step()
    decoder_optim.step()

    return loss.item() / target_length


def train_iters(encoder, decoder, encoder_optim, decoder_optim, is_ptr, training_pairs, n_iters, teacher_force=0.5,
                print_every=500, plot_every=10, save_every=1000):
    """
    The training loop.
    """
    checkpoint = load_checkpoint()
    if checkpoint:
        start_iter = checkpoint["iter"]
        plot_losses = checkpoint["plot_losses"]
        print_loss_total = checkpoint["print_loss_total"]
        plot_loss_total = checkpoint["plot_loss_total"]
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        encoder_optim.load_state_dict(checkpoint["encoder_optim"])
        decoder_optim.load_state_dict(checkpoint["decoder_optim"])
    else:
        start_iter = 1
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

    criterion = nn.NLLLoss()

    epoch, size = 0, len(training_pairs)

    start = time.time()
    for i in range(start_iter, n_iters + 1):
        if (i - 1) % size == 0 and i - 1 != 0:
            epoch += 1
        input_tensor, target_tensor = training_pairs[i - 1 - epoch * size]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optim, decoder_optim, is_ptr, criterion,
                     teacher_force)
        print_loss_total += loss
        plot_loss_total += loss

        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, i / n_iters), i, i / n_iters * 100, print_loss_avg))

        if i % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_loss_total = 0
            plot_losses.append(plot_loss_avg)

        if i % save_every == 0:
            save_checkpoint({
                "iter": i + 1,
                "plot_losses": plot_losses,
                "print_loss_total": print_loss_total,
                "plot_loss_total": plot_loss_total,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "encoder_optim": encoder_optim.state_dict(),
                "decoder_optim": decoder_optim.state_dict(),
            }, i, save_every)

    show_plot(plot_losses)
