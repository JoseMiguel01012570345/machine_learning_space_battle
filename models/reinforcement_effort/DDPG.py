import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers

import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt

# Specify the `render_mode` parameter to show the attempts of the agent in a pop up window.
metric = 0
env = gym.make()

num_states = len(env.observation_space.flatten())
print("Size of State Space ->  {}".format(num_states))
num_actions = len(env.action_space)
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.upper_bound 
lower_bound = env.lower_bound

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, 1))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') observation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    # @tf.function
    
    def update(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            
            target_actions = target_actor(next_state_batch, training=True)
                        
            critic_value = critic_model([state_batch, action_batch], training=True)
            
            tar_critic = target_critic(
                [next_state_batch, target_actions ], training=True
            )
            y = reward_batch + gamma * tar_critic
            
            critic_loss = tf.reduce_mean(tf.square(y - critic_value))
            
            os.system('cls')
            print('----------------------------------------------------')
            print("critic_loss:",critic_loss.numpy())
            
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            
            actions = actor_model(state_batch, training=True)
                        
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.reduce_mean(critic_value)
            print("actor_loss:", actor_loss.numpy() )
            print('----------------------------------------------------')
            
        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype="float32")
        next_state_batch = tf.convert_to_tensor(
            self.next_state_buffer[batch_indices]
        )

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target(target, original, tau):
    target_weights = target.get_weights()
    original_weights = original.get_weights()

    for i in range(len(target_weights)):
        target_weights[i] = original_weights[i] * tau + target_weights[i] * (1 - tau)

    target.set_weights(target_weights)

def save_models():
    
    folder_path = f'./map_types/metric{metric}' 
    
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    # Assuming actor_model and critic_model are already defined and trained
    # Save the actor model
    actor_model.save( folder_path +"/" +'actor_model.h5' )

    # Save the critic model
    critic_model.save( folder_path + "/" +'critic_model.h5' )
    

def get_actor():
    # Initialize weights between 1 and 2
    last_init = keras.initializers.RandomUniform(minval=1.0, maxval=2.0)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(2, activation="tanh")(inputs)
    outputs = layers.Dense(1, activation="relu", kernel_initializer=last_init)(out)

    outputs = outputs * upper_bound
    model = keras.Model(inputs, outputs)
    return model

def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(4, activation="relu")(state_input)
    state_out = layers.Dense(4, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(1,))
    action_out = layers.Dense(1, activation="relu")(action_input)

    # Both are passed through separate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(2, activation="tanh")(concat)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = keras.Model([state_input, action_input], outputs)

    return model

def policy(state):
    
    act =actor_model(state)
    # print(act.numpy())
    sampled_actions = tf.squeeze(act)

    sampled_actions = sampled_actions.numpy()

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    
    return [np.squeeze(legal_action)]

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = keras.optimizers.Adam(critic_lr)
actor_optimizer = keras.optimizers.Adam(actor_lr)

total_episodes = 300
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(50000, 1)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

# Takes about 4 min to train
for ep in range(total_episodes):
    prev_state, _ = env.reset()
    episodic_reward = 0

    while True:
        tf_prev_state = tf.expand_dims(
            tf.convert_to_tensor(prev_state), 0
        )

        action = policy(tf_prev_state)
        # Receive state and reward from environment.
        state, reward, done, truncated, _ = env.step(action)
        
        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()

        update_target(target_actor, actor_model, tau)
        update_target(target_critic, critic_model, tau)

        # End this episode when `done` or `truncated` is True
        if done or truncated:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)


# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.show()