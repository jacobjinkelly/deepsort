"""
Training a model using the EWC method.
"""
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from data import tensors_from_pair
from evaluate import evaluate
from utils import SOS_token, device, EOS_token, time_since, save_checkpoint, load_checkpoint


def train(encoder, decoder, optim, optim_params, importance, weight_init, grad_clip, is_ptr, tasks, n_epochs,
          teacher_force_ratio, print_every, plot_every, save_every):
    """
    Train on an assortment of tasks.
    """
    encoder_optim = optim(encoder.parameters(), **optim_params)
    decoder_optim = optim(decoder.parameters(), **optim_params)

    checkpoint = load_checkpoint("ewc_ptr" if is_ptr else "ewc_vanilla")
    if checkpoint:
        start_task = checkpoint["task"]
        first_epoch = checkpoint["epoch"]
        first_iter = checkpoint["iter"]
        plot_losses = checkpoint["plot_losses"]
        print_loss_total = checkpoint["print_loss_total"]
        plot_loss_total = checkpoint["plot_loss_total"]
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        encoder_optim.load_state_dict(checkpoint["encoder_optim"])
        decoder_optim.load_state_dict(checkpoint["decoder_optim"])
    else:
        start_task = 0
        first_epoch = 0
        first_iter = 0
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        encoder.apply(weight_init)  # initialize weights
        decoder.apply(weight_init)  # initialize weights

    current_iter = sum([len(tasks[i]) for i in range(start_task)]) * n_epochs + len(tasks[start_task]) * first_epoch + \
        first_iter
    start = time.time()
    for task in range(start_task, len(tasks)):
        training_pairs = deepcopy(tasks[task])
        size, n_iters = len(training_pairs), n_epochs * len(training_pairs)
        start_epoch = first_epoch if task == start_task else 0
        for epoch in range(start_epoch, n_epochs):
            np.random.shuffle(training_pairs)
            start_iter = first_iter if epoch == start_epoch else 0
            for i in range(start_iter, size):
                loss = train_step(training_pairs[i], tasks[:task], encoder, decoder, encoder_optim, decoder_optim,
                                  is_ptr, F.cross_entropy, importance, teacher_force_ratio, grad_clip)
                print_loss_total += loss
                plot_loss_total += loss
                current_iter += 1

                if current_iter % print_every == 0:
                    print_loss_avg, print_loss_total = print_loss_total / print_every, 0
                    print('%s (task: %d epoch: %d iter: %d %d%%) %.4f' % (time_since(start, current_iter / n_iters),
                                                                          task, epoch, i + 1,
                                                                          current_iter / n_iters * 100,
                                                                          print_loss_avg))

                if current_iter % plot_every == 0:
                    plot_loss_avg, plot_loss_total = plot_loss_total / plot_every, 0
                    plot_losses.append(plot_loss_avg)

                if current_iter % save_every == 0:
                    if i + 1 < size:
                        save_task = task
                        save_epoch = epoch
                        save_iter = i + 1
                    else:
                        save_iter = 0
                        if epoch + 1 < n_epochs:
                            save_task = task
                            save_epoch = epoch + 1
                        else:
                            save_task = task + 1
                            save_epoch = 0
                    save_checkpoint({
                        "task": save_task,
                        "epoch": save_epoch,
                        "iter": save_iter,
                        "plot_losses": plot_losses,
                        "print_loss_total": print_loss_total,
                        "plot_loss_total": plot_loss_total,
                        "encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                        "encoder_optim": encoder_optim.state_dict(),
                        "decoder_optim": decoder_optim.state_dict(),
                    }, "ewc_ptr" if is_ptr else "ewc_vanilla")


# ignore warning that torch.tensor is not callable (bug in PyTorch)
# noinspection PyCallingNonCallable
def train_step(training_pair, tasks, encoder, decoder, encoder_optim, decoder_optim, is_ptr, criterion, importance,
               teacher_force_ratio, grad_clip):
    """
    One step in the training loop.
    """
    encoder_hidden = encoder.init_hidden()
    encoder_optim.zero_grad(), decoder_optim.zero_grad()

    loss = 0
    input_tensor, target_tensor = training_pair
    input_length, target_length = input_tensor.size(0), target_tensor.size(0)

    encoder_outputs = torch.zeros(input_length, encoder.hidden_dim, device=device)

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

        loss += criterion(decoder_output, target_tensor[i]) + importance * get_loss(tasks, encoder, decoder, is_ptr)

        if not teacher_force and decoder_input.item() == EOS_token:
            break

    loss.backward()

    # clip gradients (to avoid exploding gradients)
    nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip), nn.utils.clip_grad_norm(decoder.parameters(), grad_clip)

    encoder_optim.step()
    decoder_optim.step()

    return loss.item() / target_length


# noinspection PyCallingNonCallable,PyUnresolvedReferences
def get_loss(tasks, encoder, decoder, is_ptr):
    """
    Returns loss function for previous tasks
    """
    precision_matrices, means = {}, {}
    encoder_params = {n: p for n, p in encoder.named_parameters() if p.requires_grad}
    decoder_params = {n: p for n, p in decoder.named_parameters() if p.requires_grad}
    params = {**encoder_params, **decoder_params}
    for n, p in deepcopy(params).items():
        means[n] = Variable(p.data)
        p.data.zero_()
        precision_matrices[n] = Variable(p.data)

    criterion = nn.NLLLoss()
    for training_pairs in tasks:
        for training_pair in training_pairs:
            encoder_hidden = encoder.init_hidden()
            encoder.zero_grad(), decoder.zero_grad()

            input_tensor, target_tensor = training_pair
            input_length, target_length = input_tensor.size(0), target_tensor.size(0)

            encoder_outputs = torch.zeros(input_length, encoder.hidden_dim, device=device)

            for i in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
                encoder_outputs[i] = encoder_output[0, 0]

            decoder_input, decoder_hidden = torch.tensor([[SOS_token]], device=device), encoder_hidden

            decoded_output = []
            for i in range(target_length):
                args = (decoder_input, decoder_hidden, encoder_outputs)
                if is_ptr:
                    args += (input_tensor,)
                decoder_output, decoder_hidden, _ = decoder(*args)
                topv, topi = decoder_output.topk(1)
                decoded_output.append(topi.item())
                if topi.item() == EOS_token:
                    break
                # detach from history as input
                decoder_input = topi.squeeze().detach()
            decoded_output = torch.Tensor(decoded_output)
            loss = criterion(torch.unsqueeze(decoded_output, 1), target_tensor)
            loss.backward()

            for n, p in encoder.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / (len(tasks) * len(training_pairs))
            for n, p in decoder.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / (len(tasks) * len(training_pairs))

    precision_matrices = {n: p for n, p in precision_matrices.items()}

    loss = 0

    for n, p in encoder.named_parameters():
        _loss = precision_matrices[n] * (p - means[n]) ** 2
        loss += _loss.sum()
    for n, p in decoder.named_parameters():
        _loss = precision_matrices[n] * (p - means[n]) ** 2
        loss += _loss.sum()

    return loss
