"""For training models.
"""
import torch
import torch.nn as nn
from torch import optim
from utils import device, MAX_LENGTH, SOS_token, EOS_token, time_since,
                                                save_checkpoint, load_checkpoint
import time
import random
from visual import show_plot

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
                    decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size,
                                                                device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio \
                                                                    else False

    if use_teacher_forcing:
        # i.e. feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di] # teacher forcing
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            # detach from history as input
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_iters(encoder, decoder, encoder_optim, decoder_optim,
                training_pairs, n_iters, print_every=1000, plot_every=100,
                save_every=1000, learning_rate=0.01):

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
        print_loss_total = 0 # Reset every print_every
        plot_loss_total = 0 # Reset every plot_every

    criterion = nn.NLLLoss()

    start = time.time()
    for iter in range(start_iter, n_iters + 1):
        input_tensor, target_tensor = training_pairs[iter - 1]

        loss = train(input_tensor, target_tensor, encoder, decoder,
            encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters),
                                    iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_loss_total = 0
            plot_losses.append(plot_loss_avg)

        if iter % save_every == 0:
            save_checkpoint({
                "iter": iter + 1,
                "plot_losses": plot_losses,
                "print_loss_total": print_loss_total,
                "plot_loss_total": plot_loss_total,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "encoder_optimize": encoder_optimizer.state_dict(),
                "decoder_optimizer": decoder_optimizer.state_dict(),
            }, iter, save_every)

    show_plot(plot_losses)
