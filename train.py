"""For training models.
"""
import random
import time
import math

import numpy as np
import torch
import torch.nn as nn

from utils import device, SOS_token, EOS_token, time_since, save_checkpoint, load_checkpoint
from visual import show_plot


# ignore noncallable/unresolvedreferences errors for torch.tensor, torch.unsqueeze respectively (bug in PyTorch)
# noinspection PyCallingNonCallable,PyUnresolvedReferences
def train(pairs, encoder, decoder, encoder_optim, decoder_optim, is_ptr, criterion, teacher_force_ratio, grad_clip):
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

    teacher_force = random.random() < teacher_force_ratio

    for i in range(target_length):
        args = (decoder_input, decoder_hidden, encoder_outputs)
        if is_ptr:
            args += (input_tensor,)
        decoder_output, decoder_hidden, _ = decoder(*args)
        if not teacher_force:
            topv, topi = decoder_output.topk(1)
            # detach from history as input
            decoder_input = topi.squeeze().detach()
        else:
            decoder_input = target_tensor[i]
        loss += criterion(decoder_output, target_tensor[i])

        if not teacher_force and decoder_input.item() == EOS_token:
            break

    loss.backward()

    # clip gradients (to avoid exploding gradients)
    nn.utils.clip_grad_norm(encoder.parameters(), grad_clip), nn.utils.clip_grad_norm(decoder.parameters(), grad_clip)

    encoder_optim.step()
    decoder_optim.step()

    return loss.item() / target_length


# surpress warning of math.floor() returning a float. In Python 3 returns it returns an int.
# noinspection PyTypeChecker
def train_iters(encoder, decoder, optim, optim_params, weight_init, grad_clip, is_ptr, training_pairs, n_epochs,
                batch_size, teacher_force_ratio=0.5, print_every=500, plot_every=10, save_every=1000):
    """
    The training loop.
    """
    encoder_optim = optim(encoder.parameters(), **optim_params)
    decoder_optim = optim(decoder.parameters(), **optim_params)

    checkpoint = load_checkpoint()
    if checkpoint:
        start_epoch = checkpoint["epoch"]
        first_batch = checkpoint["batch"]
        plot_losses = checkpoint["plot_losses"]
        print_loss_total = checkpoint["print_loss_total"]
        plot_loss_total = checkpoint["plot_loss_total"]
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        encoder_optim.load_state_dict(checkpoint["encoder_optim"])
        decoder_optim.load_state_dict(checkpoint["decoder_optim"])
    else:
        start_epoch = 0
        first_batch = 0
        encoder.apply(weight_init)  # initialize weights
        decoder.apply(weight_init)  # initialize weights
        plot_losses = []
        print_loss_total = 0        # Reset every print_every
        plot_loss_total = 0         # Reset every plot_every

    criterion = nn.NLLLoss()

    size, n_iters = len(training_pairs), n_epochs * len(training_pairs)

    start = time.time()
    for epoch in range(start_epoch, n_epochs):
        np.random.shuffle(training_pairs)
        start_batch = first_batch if epoch == start_epoch else 0
        for batch in range(start_batch, math.ceil(size / batch_size)):
            l, u = batch * batch_size, (batch + 1) * batch_size + 1
            u = size + 1 if u > size + 1 else u

            pairs = training_pairs[l:u]
            loss = train(pairs, encoder, decoder, encoder_optim, decoder_optim, is_ptr, criterion, teacher_force_ratio,
                         grad_clip)
            print_loss_total += loss
            plot_loss_total += loss

            current_iter = math.ceil(size / batch_size) * epoch + batch

            if current_iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (time_since(start, current_iter / n_iters),
                                             current_iter, current_iter / n_iters * 100, print_loss_avg))

            if current_iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_loss_total = 0
                plot_losses.append(plot_loss_avg)

            if current_iter % save_every == 0:
                save_checkpoint({
                    "epoch": epoch,
                    "batch": batch,
                    "plot_losses": plot_losses,
                    "print_loss_total": print_loss_total,
                    "plot_loss_total": plot_loss_total,
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "encoder_optim": encoder_optim.state_dict(),
                    "decoder_optim": decoder_optim.state_dict(),
                })

    show_plot(plot_losses)
