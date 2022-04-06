import numpy as np
import torch
import transformers
from torch import nn

from td3.trajectory_gpt2 import GPT2Model


class Thunk(nn.Module):
    """
    this is the encoding module
    """
    def __init__(self,
                 state_dim,
                 stacked_frames,
                 hidden_size,
                 num_layers,
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
        self.feature_dim = hidden_size
    
    def forward(self, states, actions=None):
        return None


class MLPThunk(Thunk):
    def __init__(self,
                 state_dim,
                 stacked_frames,
                 hidden_size,
                 num_layers,
                 concat_action=False,
                 dropout=0.1):
        super().__init__()

        layers = []
        for i in range(num_layers):
            input_dim = hidden_size if i > 0 else state_dim*stacked_frames
            layers.append(nn.Linear(input_dim, hidden_layer_size))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.shape[0], -1))


class CNN(Thunk):

    def __init__(self,
                 state_dim,
                 stacked_frames,
                 hidden_size,
                 num_layers,
                 concat_action=False,
                 dropout=0.1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=stacked_frames, out_channels=16, kernel_size=(8, 8), stride=(4, 4)),
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


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_layer_size=512, num_layers=2, history_length=1):
        super().__init__()
        self.hidden_size = 512
        self.num_layers = num_layers

        self.net = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_layer_size,
                          num_layers=num_layers,
                          batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device)
        output, hn = self.net(x, h0)
        y = output[:,-1]
        return y


class Transformer(nn.Module):

    """
    This model uses GPT to model (state_1(action_1), state_2(action_2), ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        return state_preds, action_preds, return_preds
