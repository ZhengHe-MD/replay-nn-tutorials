"""
the model translates Chinese to English
"""
import random

import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.rnn import pad_sequence

from data import load_data
from nltk.tokenize import word_tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SOS = '<SOS>'
EOS = '<EOS>'
PAD = '<PAD>'
UNK = '<UNK>'

HIDDEN_SIZE = 256
BATCH_SIZE = 8
LR = 0.01
TEACHER_FORCING_RATIO = 0.5
N_ITER = 75000
PRINT_EVERY = 1000
MAX_DECODE_LENGTH = 20


def tokenize_cmn(x: str) -> list[str]:
    return [ch for ch in x.strip()]


def tokenize_eng(x: str) -> list[str]:
    return word_tokenize(x)


lang2tokenizer = {
    'cmn': tokenize_cmn,
    'eng': tokenize_eng
}


class Lang:
    def __init__(self, name):
        self.name = name
        self.tokenize = lang2tokenizer[name]
        self.word2index = {SOS: 0, EOS: 1, PAD: 2, UNK: 3}
        self.word2count = {}
        self.index2word = {0: SOS, 1: EOS, 2: PAD, 3: UNK}
        self.n_word = 4

    def add_sentence(self, sentence):
        for word in self.tokenize(sentence):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_word
            self.word2count[word] = 1
            self.index2word[self.n_word] = word
            self.n_word += 1
        else:
            self.word2count[word] += 1

    def sentence_indexes(self, sentence):
        return [self.word2index[word] for word in self.tokenize(sentence)]

    def tokenize(self, sentence):
        return self.tokenize(sentence)


class DataProvider:
    def __init__(self, source_lang_code='cmn', target_lang_code='eng'):
        self.source_lang = Lang(source_lang_code)
        self.target_lang = Lang(target_lang_code)
        self.pairs = []
        self.n_pair = 0
        self.load(source_lang_code, target_lang_code)

    def load(self, source_lang_code, target_lang_code):
        pairs = []
        for pair in load_data(source_lang_code, target_lang_code):
            if len(self.source_lang.tokenize(pair[0])) >= MAX_DECODE_LENGTH \
                    or len(self.target_lang.tokenize(pair[1])) >= MAX_DECODE_LENGTH:
                continue
            pairs.append(pair)
        self.pairs = pairs
        self.n_pair = len(self.pairs)
        print(f'read {len(self.pairs)} sentence pairs')
        print('build vocabulary...')
        for pair in self.pairs:
            self.source_lang.add_sentence(pair[0])
            self.target_lang.add_sentence(pair[1])
        print('word counts:')
        print(self.source_lang.name, self.source_lang.n_word)
        print(self.target_lang.name, self.target_lang.n_word)

    def get_source_tensor(self, sentence) -> torch.Tensor:
        source_tensor = self.source_lang.sentence_indexes(sentence)
        source_tensor.append(self.source_lang.word2index[EOS])
        return torch.LongTensor(source_tensor)

    def get_target_tensor(self, sentence) -> torch.Tensor:
        target_tensor = self.target_lang.sentence_indexes(sentence)
        target_tensor.append(self.target_lang.word2index[EOS])
        return torch.LongTensor(target_tensor)

    def get_tensor_pairs(self, pair) -> (torch.Tensor, torch.Tensor):
        source_sentence, target_sentence = pair
        source_tensor = self.get_source_tensor(source_sentence)
        target_tensor = self.get_target_tensor(target_sentence)
        return source_tensor, target_tensor

    def get_tensor_pairs_in_batch(self, batch_size=8, page=0) -> (torch.Tensor, torch.Tensor):
        start = (batch_size * page) % self.n_pair
        source_tensors, target_tensors = [], []
        for i in range(batch_size):
            pair_index = (start + i) % self.n_pair
            source_tensor, target_tensor = self.get_tensor_pairs(self.pairs[pair_index])
            source_tensors.append(source_tensor)
            target_tensors.append(target_tensor)
        return (
            pad_sequence(source_tensors, batch_first=False, padding_value=self.source_lang.word2index[PAD]),
            pad_sequence(target_tensors, batch_first=False, padding_value=self.target_lang.word2index[PAD])
        )


class SimpleEncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SimpleEncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size)

    def forward(self, x, hidden):
        # NOTE: Let the framework infer batch_size because batch_size can be different between training and inference.
        embedded = self.embedding(x).view(1, -1, self.hidden_size)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


class SimpleDecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(SimpleDecoderRNN, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        output = self.embedding(x).view(1, -1, self.hidden_size)
        output = self.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.fc(output[0]))
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


class AttnDecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x).view(1, -1, self.hidden_size)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # lhs: (batch_size, 1, max_length)
        # rhs: (batch_size, max_length, hidden_size)
        # => : (batch_size, 1, hidden_size)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.swapaxes(0, 1))
        # => : (batch_size, hidden_size * 2)
        output = torch.cat((embedded[0], attn_applied.squeeze(1)), dim=1)
        # => : (batch_size, hidden_size)
        output = F.relu(self.attn_combine(output))
        # => : (1, batch_size, hidden_size)
        output, hidden = self.gru(output.unsqueeze(0), hidden)
        # => : (batch_size, hidden_size)
        output = F.log_softmax(self.fc(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


class Trainer:
    def __init__(self, provider: DataProvider):
        self.provider = provider
        self.encoder = SimpleEncoderRNN(vocab_size=provider.source_lang.n_word,
                                        embed_size=HIDDEN_SIZE,
                                        hidden_size=HIDDEN_SIZE).to(device)
        self.decoder = AttnDecoderRNN(vocab_size=provider.target_lang.n_word,
                                      hidden_size=HIDDEN_SIZE,
                                      max_length=MAX_DECODE_LENGTH).to(device)
        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=LR)
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=LR)
        self.criterion = nn.NLLLoss()

    def exec(self, checkpoint=None):
        if checkpoint is not None:
            self.load(checkpoint)

        total_loss = 0
        for i in range(1, N_ITER + 1):
            source_list, target_list = self.provider.get_tensor_pairs_in_batch(BATCH_SIZE, i - 1)
            source_list, target_list = source_list.to(device), target_list.to(device)

            loss = self.train(source_list, target_list)
            total_loss += loss

            if i % PRINT_EVERY == 0:
                random_pair = random.choice(self.provider.pairs)
                random_source_sentence, random_target_sentence = random_pair
                output_target_sentence, attentions = self.evaluate(random_source_sentence)

                print(f'loss: {total_loss / PRINT_EVERY} | random source: {random_source_sentence} | '
                      f'random target: {random_target_sentence} | output target: {output_target_sentence}')
                total_loss = 0

        if checkpoint is not None:
            self.save(checkpoint)

    def train(self, source_list, target_list):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        source_length, target_length = source_list.size(0), target_list.size(0)

        loss: torch.Tensor = torch.tensor(0, dtype=torch.float, device=device)

        # encode
        encoder_outputs = torch.zeros(MAX_DECODE_LENGTH, BATCH_SIZE, HIDDEN_SIZE, device=device)
        encoder_hidden = self.encoder.init_hidden(BATCH_SIZE).to(device)
        for ei in range(source_length):
            encoder_output, encoder_hidden = self.encoder(
                source_list[ei].unsqueeze(0), encoder_hidden)
            encoder_outputs[ei] = encoder_output[0]
        # decode
        decoder_input = torch.full(size=(BATCH_SIZE, 1),
                                   fill_value=self.provider.target_lang.word2index[SOS],
                                   device=device)
        decoder_hidden = encoder_hidden
        use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += self.criterion(decoder_output, target_list[di])
            if use_teacher_forcing:
                decoder_input = target_list[di]
            else:
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return loss.item() / target_length

    def evaluate(self, sentence):
        with torch.no_grad():
            source_tensor = self.provider.get_source_tensor(sentence).to(device)
            source_list = source_tensor.view(-1, 1)
            source_length = source_list.size(0)
            # encode
            encoder_outputs = torch.zeros(MAX_DECODE_LENGTH, 1, HIDDEN_SIZE, device=device)
            encoder_hidden = self.encoder.init_hidden(1).to(device)
            for ei in range(source_length):
                encoder_output, encoder_hidden = self.encoder(source_list[ei].unsqueeze(0), encoder_hidden)
                encoder_outputs[ei] = encoder_output[0]
            # decode
            decoder_input = torch.tensor([[self.provider.target_lang.word2index[SOS]]], device=device)
            decoder_hidden = encoder_hidden
            decoded_words = []
            decoder_attentions = torch.zeros(MAX_DECODE_LENGTH, 1, MAX_DECODE_LENGTH)
            for di in range(MAX_DECODE_LENGTH):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention

                topv, topi = decoder_output[0].topk(1)
                if topi.item() == self.provider.target_lang.word2index[EOS]:
                    decoded_words.append(EOS)
                    break
                else:
                    decoded_words.append(self.provider.target_lang.index2word[topi.item()])
                decoder_input = torch.tensor([[topi.squeeze().detach()]], device=device)
            return decoded_words, decoder_attentions[:di + 1]

    def save(self, filename):
        state_dict = {
            'encoder_model': self.encoder.state_dict(),
            'decoder_model': self.decoder.state_dict(),
            'encoder_optimizer': self.encoder_optimizer.state_dict(),
            'decoder_optimizer': self.decoder_optimizer.state_dict(),
            'criterion': self.criterion.state_dict()
        }
        torch.save(state_dict, filename)

    def load(self, filename):
        state_dict = torch.load(filename)
        self.encoder.load_state_dict(state_dict['encoder_model'])
        self.encoder_optimizer.load_state_dict(state_dict['encoder_optimizer'])
        self.decoder.load_state_dict(state_dict['decoder_model'])
        self.decoder_optimizer.load_state_dict(state_dict['decoder_optimizer'])
        self.criterion.load_state_dict(state_dict['criterion'])
