import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.keras import layers


import numpy as np

class Buffer:

    def __init__(self, num_states, num_actions, upper_bound, lower_bound, buffer_capacity, batch_size, anodes1, anodes2, slayer, alayer, layer2, layer3, alrate, clrate, gamma, tau):

        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.anodes1=anodes1
        self.anodes2=anodes2
        self.slayer=slayer
        self.alayer=alayer
        self.layer2=layer2
        self.layer3=layer2
        self.alrate=alrate
        self.clrate=clrate
        self.gamma=gamma
        self.tau=tau
        
        self.num_states=num_states
        self.num_actions=num_actions
        self.upper_bound=upper_bound
        self.lower_bound=lower_bound

        # Its tells us num of times record() was called.
        self.buffer_counter = 0
        self.SignCnt=0
        self.criticloss=0
        self.actorloss=0
        self.actor_action_avg=0
        self.actor_action_std=0
        self.y=np.zeros((batch_size))
        self.critic_value=np.zeros((batch_size))

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element

        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))
        
        self.actor_model,self.critic_model,self.target_actor,self.target_critic = self.build_models()
        self.critic_optimizer = tf.keras.optimizers.Adam(self.clrate)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.alrate)


    def build_models(self):
        actor_model = get_actor(self.anodes1,self.anodes2,self.num_states, self.num_actions, self.upper_bound)
        critic_model = get_critic(self.slayer,self.alayer,self.layer2,self.layer3,self.num_states, self.num_actions)
        target_actor = get_actor(self.anodes1,self.anodes2,self.num_states, self.num_actions, self.upper_bound)
        target_critic = get_critic(self.slayer,self.alayer,self.layer2,self.layer3,self.num_states, self.num_actions)

        return actor_model,critic_model,target_actor,target_critic

    # Takes (s,a,r,s') obervation tuple as input

    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, trainactor ):

        # Training and updating Actor & Critic networks.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
            self.criticloss=critic_loss   
            self.critic_value=critic_value
            self.y=y
   
            SignChk=np.zeros(len(y))
            for i in range(len(SignChk)):
                if (y[i].numpy()>0 and critic_value[i].numpy()<0) or (y[i].numpy()<0 and critic_value[i].numpy()>0):
                    SignChk[i]=1
                else:
                    SignChk[i]=0
            self.SignCnt=np.sum(SignChk)

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        # actor network update
        if trainactor:
            with tf.GradientTape() as tape:
                actions = self.actor_model(state_batch, training=True)
                critic_value = self.critic_model([state_batch, actions], training=True)
                # Used `-value` as we want to maximize the value given
                # by the critic for our actions
                actor_loss = -tf.math.reduce_mean(critic_value)
                self.actorloss=actor_loss
            actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor_model.trainable_variables)
            )

    # We compute the loss and update parameters
    def learn(self, trainactor):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)
        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch,trainactor)
        update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
        update_target(self.target_critic.variables, self.critic_model.variables, self.tau)

    def policy(self,state):
    
        sampled_actions = tf.squeeze(self.actor_model(state))
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() 
        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)
        return [np.squeeze(legal_action)]
        

def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

def get_actor(nodes1, nodes2, num_states, num_actions, upper_bound):
    # Initialize weights for last layer between -3e-3 and 3-e3
    initval = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(nodes1, activation="relu")(inputs) 
    out = layers.Dense(nodes2, activation="relu")(out)
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=initval)(out)
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    #print("actor model summary :",model.summary())
    return model


def get_critic(slayer, alayer, layer2, layer3, num_states,num_actions):
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(slayer, activation="relu")(state_input)
    # Action as input
    initval = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(alayer, activation="relu")(action_input)
    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])
    out = layers.Dense(layer2, activation="relu")(concat)
    if layer3>0: 
        out = layers.Dense(layer3, activation="relu")(out) 
    outputs = layers.Dense(1,kernel_initializer=initval)(out)
    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)
    return model

