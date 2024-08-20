import os

# os.system('cls') 
os.environ["KERAS_BACKEND"] = "tensorflow"
import threading

import keras
from keras import layers

import gym
import numpy as np
import tensorflow as tf

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 1000
max_episodes = 100000  # Limit training episodes, will run until solved if smaller than 1
max_rw =10000
avg_reward_list = []
# Use the Atari environment
# Specify the `render_mode` parameter to show the attempts of the agent in a pop up window.
env = gym.make()  # , render_mode="human")

num_actions = env.action_space.shape[0]
observation_space= env.observation_space

def create_q_model(input_data):
    # Network defined by the Deepmind paper
    
    return keras.Sequential(
        [
            # Convolutions on the frames on the screen
            layers.Input( shape= ( input_data.shape[0] , input_data.shape[1] , 1 ) ),
            layers.Conv2D(32, 8, strides=4, activation="relu"),
            layers.Conv2D(64, 4, strides=2, activation="relu"),
            layers.Conv2D(64, 3, strides=1, activation="relu"),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(num_actions, activation="linear"),
        ]
    )

def save_model():
    print('saving model...')
    model_target.save(f'./model_target', overwrite=True)
    pass

# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model(observation_space)
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model(observation_space)

# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.025, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 1000000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 100 actions
update_after_actions = 100
# How often to update the target network
update_target_network = 100
# Using huber loss for stability
loss_function = keras.losses.mae

model_target.compile( optimizer=optimizer , loss = loss_function)
model.compile( optimizer=optimizer , loss = loss_function)

avg_loss = 10000000
loss_list = []
max_memory_loss = 100
epsilon_loss = 1e-3
epochs = 2
best_avg_loss = 100000000
avg_count = 0

for epoch in range(epochs):
    
    for sample in range(env.len_sample):
    
        env.next()
        
        while True:
            observation , progress , _ = env.reset()
            state = np.array(observation)
            episode_reward = 0

            for timestep in range(1, max_steps_per_episode):
                frame_count += 1

                # Use epsilon-greedy for exploration
                if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                    # Take random action
                    action = np.random.choice(num_actions)
                else:
                    # Predict action Q-values
                    # From environment state
                    state_tensor = tf.convert_to_tensor(state)
                    state_tensor = tf.expand_dims(state_tensor, 0)
                    action_probs = model(state_tensor, training=False)
                    # Take best action
                    action = tf.argmax(action_probs[0]).numpy()

                # Decay probability of taking random action
                epsilon -= epsilon_interval / epsilon_greedy_frames
                epsilon = max(epsilon, epsilon_min)

                # Apply the sampled action in our environment
                state_next, reward, done, truncated , _ = env.step(action)
                state_next = np.array(state_next)

                episode_reward += reward

                # Save actions and states in replay buffer
                action_history.append(action)
                state_history.append(state)
                state_next_history.append(state_next)
                done_history.append(done)
                rewards_history.append(reward)
                state = state_next

                # Update every fourth frame and once batch size is over 32
                if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
                    # Get indices of samples for replay buffers
                    indices = np.random.choice(range(len(done_history)), size=batch_size)

                    # Using list comprehension to sample from replay buffer
                    state_sample = np.array([state_history[i] for i in indices])
                    state_next_sample = np.array([state_next_history[i] for i in indices])
                    rewards_sample = [rewards_history[i] for i in indices]
                    action_sample = [action_history[i] for i in indices]
                    done_sample = tf.convert_to_tensor(
                        [float(done_history[i]) for i in indices]
                    )

                    # Build the updated Q-values for the sampled future states
                    # Use the target model for stability
                    future_rewards = model_target(state_next_sample)
                    future_rewards = tf.convert_to_tensor( np.clip(future_rewards , 0.0 , 1.0) ) * env.upper_bound
                    
                    predicted_rw = tf.reduce_max(
                        future_rewards, axis=1
                    )
                    
                    # Q value = reward + discount factor * expected future reward
                    updated_q_values = rewards_sample + gamma * predicted_rw

                    # If final frame set the last value to -1
                    updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                    # Create a mask so we only calculate loss on the updated Q-values
                    masks = tf.one_hot(action_sample , depth=num_actions )

                    with tf.GradientTape() as tape:
                        # Train the model on the states and updated Q-values
                        q_values_ = model(state_sample) 
                        
                        q_values = q_values_ *  env.upper_bound 
                        
                        # Apply the masks to the Q-values to get the Q-value for action taken
                        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                        q_action = tf.abs(q_action)
                        
                        # Calculate loss between new Q-value and old Q-value
                        loss = loss_function(updated_q_values, q_action)
                        
                        # os.system('cls')
                        loss_list.append( loss.numpy() )
                        
                        if max_memory_loss <= len(loss_list):
                            loss_list = loss_list[1:]
                        
                        avg_loss = np.mean(np.array( loss_list ))
                        
                        if best_avg_loss > avg_loss.T:
                            best_avg_loss = avg_loss.T
                            avg_count += 1
                            save_model()
                            
                        # show progress
                        os.system('cls')
                        print('______________________________________')
                        print(f"{progress}%")
                        print("model_version=", avg_count )
                        print("avg_loss:", avg_loss )
                        print("loss:", loss.numpy() )
                        
                            
                    # Backpropagation
                    grads = tape.gradient( loss , model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                if frame_count % update_target_network == 0:
                    # update the the target network with new weights
                    model_target.set_weights(model.get_weights())
            
                # Log details
                
                template = "running reward: {:.2f} at episode {}, frame count {}"
                # print(template.format(running_reward, episode_count, frame_count))

                # Limit the state and reward history
                if len(rewards_history) > max_memory_length:
                    del rewards_history[:1]
                    del state_history[:1]
                    del state_next_history[:1]
                    del action_history[:1]
                    del done_history[:1]

                if done:
                    break
            
            if done:
                break

            # Update running reward to check condition for solving
            episode_reward_history.append(episode_reward)
            
            running_reward = np.mean(episode_reward_history)
            
            print( f"avg reward per { frame_count % max_steps_per_episode } episode ==>" , running_reward )
            del episode_reward_history
            episode_reward_history = []
            
            avg_reward_list.append(running_reward)
            
            episode_count += 1
        
    if avg_loss <= 15:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break

# plotting results
import matplotlib.pyplot as plt

plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.show()