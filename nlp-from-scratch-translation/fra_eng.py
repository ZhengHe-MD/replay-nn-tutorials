"""
This file adds batch-training support to the original tutorial.

the model translates French to English
"""

import random
import re

import unicodedata
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from data import load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_LENGTH = 10

SOS = '<SOS>'
EOS = '<EOS>'
PAD = '<PAD>'
UNK = '<UNK>'


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {SOS: 0, EOS: 1, PAD: 2, UNK: 3}
        self.word2count = {}
        self.index2word = {0: SOS, 1: EOS, 2: PAD, 3: UNK}
        self.n_words = 4  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def sentence_indexes(self, sentence):
        return [self.word2index[word] for word in sentence.split(' ')]


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    # to keep '.!?' as a word itself
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]", r" ", s)
    return s


eng_prefixes = (
    'i am', 'i m ',
    'he is', 'he s ',
    'she is', 'she s ',
    'you are', 'you re ',
    'we are', 'we re ',
    'they are', 'they re '
)


def should_keep(pair):
    return len(pair[0].split(' ')) < MAX_LENGTH and \
           len(pair[1].split(' ')) < MAX_LENGTH and \
           pair[1].startswith(eng_prefixes)


def read_lang_pair(source, target):
    print("reading lines...")
    pairs = load_data(source, target)
    pairs = [(normalize_string(s), normalize_string(t)) for (s, t) in pairs]
    source_lang = Lang(source)
    target_lang = Lang(target)
    return source_lang, target_lang, pairs


def filter_pairs(pairs):
    return [pair for pair in pairs if should_keep(pair)]


class DataProvider:
    def __init__(self, source_lang_name: str, target_lang_name: str):
        self.source_lang_name = source_lang_name
        self.target_lang_name = target_lang_name
        self.source_lang, self.target_lang, self.pairs = self.load()
        self.n_pairs = len(self.pairs)

    def load(self):
        source_lang, target_lang, pairs = read_lang_pair(self.source_lang_name, self.target_lang_name)
        print(f'read {len(pairs)} sentence pairs')
        pairs = filter_pairs(pairs)
        print(f'trimmed to {len(pairs)} sentence pairs')
        print('counting words...')
        for pair in pairs:
            source_lang.add_sentence(pair[0])
            target_lang.add_sentence(pair[1])
        print("counted words:")
        print(source_lang.name, source_lang.n_words)
        print(target_lang.name, target_lang.n_words)
        return source_lang, target_lang, pairs

    def get_tensor_pairs_in_batch(self, batch_size=8, page=0):
        start = (batch_size * page) % self.n_pairs
        source_tensors, target_tensors = [], []
        for i in range(batch_size):
            pair_index = (start + i) % self.n_pairs
            source_tensor, target_tensor = self.get_tensor_pairs(self.pairs[pair_index])
            source_tensors.append(source_tensor)
            target_tensors.append(target_tensor)
        return (
            pad_sequence(source_tensors, batch_first=False, padding_value=self.source_lang.word2index[PAD]),
            pad_sequence(target_tensors, batch_first=False, padding_value=self.target_lang.word2index[PAD])
        )

    def get_tensor_pairs(self, pair):
        source_sentence, target_sentence = pair
        source_tensor = self.get_source_tensor(source_sentence)
        target_tensor = self.get_target_tensor(target_sentence)
        return source_tensor, target_tensor

    def get_source_tensor(self, sentence):
        if self.source_lang is None:
            raise Exception('please load the data first')
        source_tensor = self.source_lang.sentence_indexes(sentence)
        source_tensor.append(self.source_lang.word2index[EOS])
        return torch.LongTensor(source_tensor)

    def get_target_tensor(self, sentence):
        if self.target_lang is None:
            raise Exception('please load the data first')
        target_tensor = self.target_lang.sentence_indexes(sentence)
        target_tensor.append(self.target_lang.word2index[EOS])
        return torch.LongTensor(target_tensor)


class SimpleEncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, batch_size):
        super(SimpleEncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, -1, self.hidden_size)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


class SimpleDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size):
        super(SimpleDecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, -1, self.hidden_size)
        output = self.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.fc(output[0]))
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, -1, self.hidden_size)
        embedded = self.dropout(embedded)

        # shape=(8, 10)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # attn_weights=(8, 1, 10), encoder_outputs=(8, 10, 256) => (8, 1, 256)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.swapaxes(0, 1))
        output = torch.cat((embedded[0], attn_applied.swapaxes(0, 1)[0]), 1)
        output = self.attn_combine(output)
        output = F.relu(output)
        output, hidden = self.gru(output.unsqueeze(0), hidden)
        output = F.log_softmax(self.fc(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


HIDDEN_SIZE = 256
BATCH_SIZE = 8
LR = 0.001
TEACHER_FORCING_RATIO = 0.5
N_ITER = 75000
PRINT_EVERY = 1000


class Trainer:
    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider
        self.batch_size = BATCH_SIZE

        self.encoder = SimpleEncoderRNN(input_size=data_provider.source_lang.n_words,
                                        embed_size=HIDDEN_SIZE,
                                        hidden_size=HIDDEN_SIZE,
                                        batch_size=self.batch_size).to(device)
        self.decoder = AttnDecoderRNN(hidden_size=HIDDEN_SIZE,
                                      output_size=data_provider.target_lang.n_words,
                                      batch_size=self.batch_size).to(device)
        # self.decoder = SimpleDecoderRNN(hidden_size=HIDDEN_SIZE,
        #                                 output_size=data_provider.target_lang.n_words).to(device)
        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=LR)
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=LR)
        self.criterion = nn.NLLLoss()

        self.teacher_forcing_ratio = TEACHER_FORCING_RATIO
        self.n_iter = N_ITER
        self.print_every = PRINT_EVERY

    def exec(self, checkpoint=None):
        if checkpoint is not None:
            try:
                self.load(checkpoint)
            except FileNotFoundError:
                print(f"checkpoint {checkpoint} not found, train from scratch")

        total_loss = 0
        for i in range(1, self.n_iter + 1):
            source_list, target_list = self.data_provider.get_tensor_pairs_in_batch(self.batch_size, i - 1)
            source_list = source_list.to(device)
            target_list = target_list.to(device)

            loss = self.train(source_list, target_list)
            total_loss += loss

            if i % self.print_every == 0:
                random_pair = random.choice(self.data_provider.pairs)
                random_source_sentence, random_target_sentence = random_pair
                output_target_sentence, attentions = self.evaluate(random_source_sentence)

                print(f'loss: {total_loss / self.print_every} | random source: {random_source_sentence} | '
                      f'random target: {random_target_sentence} | output target: {output_target_sentence}')
                total_loss = 0

        if checkpoint is not None:
            self.save(checkpoint)

    def train(self, source_list, target_list):

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        source_length = source_list.size(0)
        target_length = target_list.size(0)

        encoder_outputs = torch.zeros(MAX_LENGTH, self.batch_size, self.encoder.hidden_size, device=device)

        loss = 0

        encoder_hidden = self.encoder.init_hidden(self.encoder.batch_size).to(device)
        for ei in range(source_length):
            encoder_output, encoder_hidden = self.encoder(
                source_list[ei].unsqueeze(dim=0), encoder_hidden)
            encoder_outputs[ei] = encoder_output[0]

        decoder_input = torch.full(
            (self.batch_size, 1),
            fill_value=self.data_provider.target_lang.word2index[SOS],
            device=device)
        decoder_hidden = encoder_hidden
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(
                    decoder_output, target_list[di])
                decoder_input = target_list[di]
        else:
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                # https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426/3
                decoder_input = topi.squeeze().detach()

                loss += self.criterion(decoder_output, target_list[di])
                # if decoder_input.item() == self.data_provider.target_lang.word2index[EOS]:
                #     break
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / target_length

    def evaluate(self, sentence):
        with torch.no_grad():
            source_tensor = self.data_provider.get_source_tensor(sentence).to(device)
            source_list = source_tensor.view(-1, 1)
            source_length = source_list.size(0)

            encoder_outputs = torch.zeros(MAX_LENGTH, 1, self.encoder.hidden_size, device=device)
            encoder_hidden = self.encoder.init_hidden(1).to(device)
            for ei in range(source_length):
                encoder_output, encoder_hidden = self.encoder(source_list[ei].unsqueeze(dim=0), encoder_hidden)
                encoder_outputs[ei] = encoder_output[0]

            decoder_input = torch.tensor([[
                self.data_provider.target_lang.word2index[SOS]
            ]], device=device)

            decoder_hidden = encoder_hidden
            decoded_words = []
            decoder_attentions = torch.zeros(MAX_LENGTH, 1, MAX_LENGTH)

            for di in range(MAX_LENGTH):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention

                topv, topi = decoder_output[0].topk(1)
                if topi.item() == self.data_provider.target_lang.word2index[EOS]:
                    decoded_words.append(EOS)
                    break
                else:
                    decoded_words.append(self.data_provider.target_lang.index2word[topi.item()])
                decoder_input = torch.tensor([[topi.squeeze().detach()]], device=device)
            return decoded_words, decoder_attentions[:di + 1]

    def evaluate_randomly(self, n=10):
        for i in range(n):
            pair = random.choice(self.data_provider.pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words, attentions = self.evaluate(pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')

    def elaborate_randomly(self):
        pair = random.choice(self.data_provider.pairs)
        output_words, attentions = self.evaluate(pair[0])
        output_sentence = ''
        if len(output_words) > 0:
            output_sentence = ' '.join(output_words[:-1])
        print(f'source sentence: {pair[0]}')
        print(f'target sentence: {pair[1]}')
        print(f'predict sentence: {output_sentence}')
        show_attention(pair[0], output_words, attentions.squeeze(1))

    def save(self, filename='translation.pt'):
        state_dict = {
            'encoder_model': self.encoder.state_dict(),
            'decoder_model': self.decoder.state_dict(),
            'encoder_optimizer': self.encoder_optimizer.state_dict(),
            'decoder_optimizer': self.decoder_optimizer.state_dict(),
            'criterion': self.criterion.state_dict()
        }
        torch.save(state_dict, filename)

    def load(self, filename='translation.pt'):
        state_dict = torch.load(filename)
        self.encoder.load_state_dict(state_dict['encoder_model'])
        self.encoder_optimizer.load_state_dict(state_dict['encoder_optimizer'])
        self.decoder.load_state_dict(state_dict['decoder_model'])
        self.decoder_optimizer.load_state_dict(state_dict['decoder_optimizer'])
        self.criterion.load_state_dict(state_dict['criterion'])


def show_attention(input_sentence, output_words, attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()