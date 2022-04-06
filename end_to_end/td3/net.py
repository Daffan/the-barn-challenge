import numpy as np
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    this is the encoding module
    """
    def __init__(self,
                 input_dim,
                 num_layers=2,
                 hidden_size=512,
                 history_length=1,
                 concat_action=False,
                 dropout=0.0):
        """
        state_dim: the state dimension
        stacked_frames: #timesteps considered in history
        hidden_size: hidden layer size
        num_layers: how many layers

        the input state should be of size [batch, stacked_frames, state_dim]
        the output should be of size [batch, hidden_size]
        """
        super().__init__()
        self.hidden_size = self.feature_dim = hidden_size
    
    def forward(self, states, actions=None):
        return None


class MLPEncoder(Encoder):
    def __init__(self,
                 input_dim,
                 num_layers=2,
                 hidden_size=512,
                 history_length=1,
                 concat_action=False,
                 dropout=0.0):
        super().__init__(input_dim=input_dim,
                         num_layers=num_layers,
                         hidden_size=hidden_size,
                         history_length=history_length,
                         concat_action=concat_action,
                         dropout=dropout)

        layers = []
        for i in range(num_layers):
            input_dim = hidden_size if i > 0 else input_dim * history_length
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)  # mlp will flatten time sequence
        return self.net(x)


class CNNEncoder(Encoder):
    def __init__(self,
                 input_dim,
                 num_layers=2,
                 hidden_size=512,
                 history_length=1,
                 concat_action=False,
                 dropout=0.0):
        super().__init__(input_dim=input_dim,
                         num_layers=num_layers,
                         hidden_size=hidden_size,
                         history_length=history_length,
                         concat_action=concat_action,
                         dropout=dropout)

        layers = []
        if num_layers > 1:
            for i in range(num_layers-1):
                input_channel = hidden_size if i > 0 else input_dim
                layers.append(nn.Conv1d(in_channels=input_channel,
                                        out_channels=hidden_size,
                                        kernel_size=3, padding=1))
                layers.append(nn.ReLU())

            layers.extend([
                nn.Conv1d(in_channels=hidden_size,
                          out_channels=hidden_size,
                          kernel_size=history_length, padding=0),
                nn.ReLU()])
        else:
            layers.extend([
                nn.Conv1d(in_channels=input_dim,
                          out_channels=hidden_size,
                          kernel_size=history_length, padding=0),
                nn.ReLU()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x [batch, seq_len, state_dim]
        x = x.permute(0, 2, 1) # [batch, state_dim, seq_len]
        x = self.net(x) # [batch, hidden_dim, 1]
        return x.squeeze(-1)


class RNNEncoder(Encoder):
    def __init__(self,
                 input_dim,
                 num_layers=2,
                 hidden_size=512,
                 history_length=1,
                 concat_action=False,
                 dropout=0.0):
        super().__init__(input_dim=input_dim,
                         num_layers=num_layers,
                         hidden_size=hidden_size,
                         history_length=history_length,
                         concat_action=concat_action,
                         dropout=dropout)

        self.num_layers = num_layers
        self.net = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)

    def forward(self, x):
        """
        always start with h0 = 0
        """
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device)
        output, hn = self.net(x, h0)
        y = output[:,-1]
        return y


class TransformerEncoder(Encoder):
    #
    #This model uses GPT to model (state_1(action_1), state_2(action_2), ...)
    #
    def __init__(self,
                 input_dim,
                 num_layers=2,
                 hidden_size=512,
                 history_length=1,
                 concat_action=False,
                 dropout=0.0):
        super().__init__(input_dim=input_dim,
                         num_layers=num_layers,
                         hidden_size=hidden_size,
                         history_length=history_length,
                         concat_action=concat_action,
                         dropout=dropout)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=8
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(history_length, hidden_size)
        self.embed_state = torch.nn.Linear(input_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

    def forward(self, states, actions=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(states.device)
        state_embeddings = self.embed_state(states)
        timesteps = torch.tensor([list(range(seq_length))] * batch_size, dtype=torch.long).to(states.device)
        time_embeddings = self.embed_timestep(timesteps)
        inputs = state_embeddings + time_embeddings
        inputs = self.embed_ln(inputs)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
        )
        x = transformer_outputs['last_hidden_state'] # [batch, history_length, hidden_size]
        x = x[:,-1]
        return x

#### below are previous code

class MLP(nn.Module):
    def __init__(self, input_dim, num_layers=2, hidden_layer_size=512):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = hidden_layer_size

        layers = []
        for i in range(num_layers):
            input_dim = hidden_layer_size if i > 0 else self.input_dim
            layers.append(nn.Linear(input_dim, hidden_layer_size))
            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class CNN(nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()
        self.feature_dim = 512
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1568, 512),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.fc(x)
