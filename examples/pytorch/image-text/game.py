import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions
import logging
from tqdm.auto import tqdm
from transformers import FuyuForCausalLM, FuyuProcessor
from peft import LoraConfig

# Set the root logger level to ERROR
logging.basicConfig(level=logging.ERROR)

# Iterate over all existing loggers and set their levels to ERROR
for logger in logging.root.manager.loggerDict.values():
    if isinstance(logger, logging.Logger):  # Check if it is a Logger instance
        logger.setLevel(logging.ERROR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the FrozenLake environment
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)

# Define the custom Fuyu model for the task
class CustomFuyu(nn.Module):
    """
    Custom Fuyu model for the FrozenLake environment.
    """
    def __init__(self, config, action_space_size=1):
        super().__init__()
        self.fuyu = FuyuForCausalLM.from_pretrained("adept/fuyu-8b",
                                                    load_in_4bit=True,
                                                    output_hidden_states=True,
                                                    bnb_4bit_use_double_quant=True,
                                                    bnb_4bit_quant_type="nf4",
                                                    bnb_4bit_compute_dtype=torch.bfloat16)
        
        self.fuyu.add_adapter(config, adapter_name="lora")
        
        # Define a simple feed-forward network with ReLU activations
        self.linear1 = nn.Linear(self.fuyu.config.hidden_size, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, action_space_size)

    def forward(self, max_token, prompt_length, **kwargs):
        # Generate tokens and return the full output including hidden states
        generation_output = self.fuyu.generate(
            **kwargs, 
            max_new_tokens=max_token, 
            return_dict_in_generate=True
        )

        # Extract the sequences and the hidden states from the output
        output = generation_output['sequences']
        hidden_states = generation_output['hidden_states']

        # Get the last hidden state
        last_hidden_state = hidden_states[-1][-1].squeeze(0)
        last_hidden_state = last_hidden_state.to(dtype=torch.float32)

        # Pass the last hidden state through the feed-forward network
        x = F.relu(self.linear1(last_hidden_state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)  # Output layer does not need an activation
        return x

# Hyperparameters
learning_rate = 0.001
discount_factor = 0.8
num_episodes = 100
entropy_coeff = 0.01

# Initialize the Fuyu models and optimizer
lora_config = LoraConfig(target_modules=["query_key_value"], init_lora_weights=False)
max_token = 25
model = CustomFuyu(lora_config, env.action_space.n).to(device)
processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for episode in tqdm(range(num_episodes), desc="Training Episodes"):
    state = env.reset()[0]
    state_one_hot = np.zeros(env.observation_space.n)
    state_one_hot[state] = 1  # One-hot encode the state

    done = False
    episode_log_probs = []
    episode_rewards = []

    while not done:
        state_one_hot_string = ''.join(map(str, state_one_hot.astype(int)))
        text_prompt = ("Game: Frozen Lake\n\n"
               "Objective: Navigate on a 4x4 grid from Start (S) to Goal (G) without falling into Holes (H).\n\n"
               "Grid Space:\n"
               "[['S', 'F', 'F', 'F'],\n"
               " ['F', 'H', 'F', 'H'],\n"
               " ['F', 'F', 'F', 'H'],\n"
               " ['H', 'F', 'F', 'G']]\n\n"
               "'S' represents the game-starting position (character), 'H' represents a hole, 'F' represents the frozen lake surface (walkable path), and 'G' represents the goal (chest or box)\n\n"
               "Action Space:\n"
               "0: Move LEFT\n"
               "1: Move DOWN\n"
               "2: Move RIGHT\n"
               "3: Move UP\n\n"
               "Directions:\n"
               "Please note, movements are relative to the grid's orientation, not the player's facing direction.\n\n"
               "Instructions:\n"
               "You will be provided with textual descriptions of the game's current state. Your response should be the numeric code for the next action you choose to take.\n\n"
               "Example:\n"
               "For a move to the right, respond with '2'.\n\n"
               "Now let's play the game\n"
                      "Current state vector:")
                      
        full_prompt = text_prompt + " " + state_one_hot_string
        
        print("Full prompt", full_prompt)

        inputs = processor(text=full_prompt, return_tensors="pt").to(device)
        prompt_length = inputs.input_ids.size(1)

        output = model(max_token=max_token, prompt_length=prompt_length, **inputs)
        layer_norm = nn.LayerNorm(output.shape[-1]).to(device)
        output = layer_norm(output)
        prob = F.softmax(output, dim=1)

        action_dist = distributions.Categorical(prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        print("action item", action.item())
        next_state, reward, done, _, _ = env.step(action.item())
        print("next state", next_state, "reward", reward, "done", done)
        
        next_state_one_hot = np.zeros(env.observation_space.n)
        next_state_one_hot[next_state] = 1

        episode_log_probs.append(log_prob)
        episode_rewards.append(reward)

        state_one_hot = next_state_one_hot

    # Compute returns
    G = 0
    returns = []
    for reward in reversed(episode_rewards):
        G = reward + discount_factor * G
        returns.insert(0, G)  # Insert the return at the beginning of the list

    # Reverse the returns list so that it corresponds to the order of the episode_log_probs
    returns = returns[::-1]

    # Compute policy loss
    policy_loss = -torch.stack(episode_log_probs) * torch.tensor(returns, device=device)
    policy_loss = policy_loss.sum()
    entropy = -action_dist.entropy().mean()

    loss = policy_loss + entropy * entropy_coeff

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 10 == 0:
        print(f"Episode: {episode + 1}, Total Reward: {sum(episode_rewards)}")

env.close()
