import sys
import gym
import tensorflow as tf
import numpy as np
import random
import datetime

"""
Hyper Parameters
"""
GAMMA = 0.999  # discount factor for target Q
INITIAL_EPSILON = 1  # starting value of epsilon
FINAL_EPSILON = 0.1  # final value of epsilon
EPSILON_DECAY_STEPS = 100
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 50  # size of minibatch

TEST_FREQUENCY = 100  # How many episodes to run before visualizing test accuracy
SAVE_FREQUENCY = 1000  # How many episodes to run before saving model (unused)
NUM_EPISODES = 2000  # Episode limitation
EP_MAX_STEPS = 200  # Step limitation in an episode
# The number of test iters (with epsilon set to 0) to run every TEST_FREQUENCY episodes
NUM_TEST_EPS = 1

HIDDEN_NODES = 100

# task2
# I don't think I will change these value
PARTITION_LEN = 0.1 # used in init continuous, partition value.


def init(env, env_name):
    """
    Initialise any globals, e.g. the replay_buffer, epsilon, etc.
    return:
        state_dim: The length of the state vector for the env
        action_dim: The length of the action space, i.e. the number of actions

    NB: for discrete action envs such as the cartpole and mountain car, this
    function can be left unchanged.

    Hints for envs with continuous action spaces, e.g. "Pendulum-v0"
    1) you'll need to modify this function to discretise the action space and
    create a global dictionary mapping from action index to action (which you
    can use in `get_env_action()`)
    2) for Pendulum-v0 `env.action_space.low[0]` and `env.action_space.high[0]`
    are the limits of the action space.
    3) setting a global flag iscontinuous which you can use in `get_env_action()`
    might help in using the same code for discrete and (discretised) continuous
    action spaces
    """
    global replay_buffer, epsilon
    replay_buffer = []
    epsilon = INITIAL_EPSILON

    state_dim = env.observation_space.shape[0]

    # task 2
    global iscontinuous, action_dict
    
    # discrete action envs
    if env_name == 'CartPole-v0' or env_name == 'MountainCar-v0':
        action_dim = env.action_space.n
        iscontinuous = False

    # continuous action env
    # [-2.0, 2.0] divided by 0.1 => 40
    elif env_name == 'Pendulum-v0':
        high = env.action_space.high[0]
        low = env.action_space.low[0]
        
        action_dim = (int) ((high - low) / PARTITION_LEN)
        action_dict = np.arange(low + PARTITION_LEN / 2, \
                                high + PARTITION_LEN / 2, PARTITION_LEN)
        iscontinuous = True

    # unknow env name
    else:
        action_dim = env.action_space.n
        iscontinuous = False
    
    return state_dim, action_dim

def get_target_network(state_dim, action_dim, hidden_nodes=HIDDEN_NODES):
    """Define another target network

    Defined scope as target_net, and the original network scope as eval_net
    That's for replace the target param
    """
    state_in = tf.placeholder("float", [None, state_dim])
    action_in = tf.placeholder("float", [None, action_dim])  # one hot

    # q value for the target network for the state, action taken
    target_in = tf.placeholder("float", [None])

    with tf.variable_scope('target_net'):
        # implement dueling here
        # hidden layer
        w_h = tf.Variable(tf.random_normal([state_dim, hidden_nodes], stddev = 0.1))
        b_h = tf.Variable(tf.zeros([hidden_nodes]))
        
        h_layer = tf.tanh(tf.matmul(state_in, w_h) + b_h)

        # Q value layer for not dueling
        # Value layer and advantage layer for dueling
        #
        # value for state
        w_v = tf.Variable(tf.random_normal([hidden_nodes, 1], stddev = 0.1))
        b_v = tf.Variable(tf.zeros([1]))
        
        values = tf.matmul(h_layer, w_v) + b_v

        # advantage for action
        w_a = tf.Variable(tf.random_normal([hidden_nodes, action_dim], stddev = 0.1))
        b_a = tf.Variable(tf.zeros([action_dim]))
    
        advantage = tf.matmul(h_layer, w_a) + b_a

        # Q = V + A
        q_values = values + (advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True))

        q_selected_action = \
            tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)

        # TO IMPLEMENT: loss function
        # should only be one line, if target_in is implemented correctly
        loss = tf.reduce_sum(tf.square(target_in - q_selected_action))
        
        optimise_step = tf.train.AdamOptimizer().minimize(loss)
        
        train_loss_summary_op = tf.summary.scalar("TrainingLoss", loss)
        
    return state_in, action_in, target_in, q_values, q_selected_action, \
           loss, optimise_step, train_loss_summary_op    

def get_network(state_dim, action_dim, hidden_nodes=HIDDEN_NODES):
    """Define the neural network used to approximate the q-function

    The suggested structure is to have each output node represent a Q value for
    one action. e.g. for cartpole there will be two output nodes.

    Hints:
    1) Given how q-values are used within RL, is it necessary to have output
    activation functions?
    2) You will set `target_in` in `get_train_batch` further down. Probably best
    to implement that before implementing the loss (there are further hints there)
    """
    state_in = tf.placeholder("float", [None, state_dim])
    action_in = tf.placeholder("float", [None, action_dim])  # one hot

    # q value for the target network for the state, action taken
    target_in = tf.placeholder("float", [None])

    # TO IMPLEMENT: Q network, whose input is state_in, and has action_dim outputs
    # which are the network's esitmation of the Q values for those actions and the
    # input state. The final layer should be assigned to the variable q_values

    with tf.variable_scope('eval_net'):
        # implement dueling here
        # hidden layer
        w_h = tf.Variable(tf.random_normal([state_dim, hidden_nodes], stddev = 0.1))
        b_h = tf.Variable(tf.zeros([hidden_nodes]))
        
        h_layer = tf.tanh(tf.matmul(state_in, w_h) + b_h)

        # Q value layer for not dueling
        # Value layer and advantage layer for dueling
        #
        # value for state
        w_v = tf.Variable(tf.random_normal([hidden_nodes, 1], stddev = 0.1))
        b_v = tf.Variable(tf.zeros([1]))
        
        values = tf.matmul(h_layer, w_v) + b_v

        # advantage for action
        w_a = tf.Variable(tf.random_normal([hidden_nodes, action_dim], stddev = 0.1))
        b_a = tf.Variable(tf.zeros([action_dim]))
    
        advantage = tf.matmul(h_layer, w_a) + b_a

        # Q = V + A
        q_values = values + (advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True))

        q_selected_action = \
            tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)

        # TO IMPLEMENT: loss function
        # should only be one line, if target_in is implemented correctly
        loss = tf.reduce_sum(tf.square(target_in - q_selected_action))
        
        optimise_step = tf.train.AdamOptimizer().minimize(loss)

        train_loss_summary_op = tf.summary.scalar("TrainingLoss", loss)
    
    return state_in, action_in, target_in, q_values, q_selected_action, \
           loss, optimise_step, train_loss_summary_op


def init_session():
    global session, writer
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

    # Setup Logging
    logdir = "tensorboard/" + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, session.graph)


def get_action(state, state_in, q_values, epsilon, test_mode, action_dim):
    # get action by eval net
    Q_estimates = q_values.eval(feed_dict={state_in: [state]})[0]
    epsilon_to_use = 0.0 if test_mode else epsilon
    if random.random() < epsilon_to_use:
        action = random.randint(0, action_dim - 1)
    else:
        action = np.argmax(Q_estimates)
    return action


def get_env_action(action):
    """
    Modify for continous action spaces that you have discretised, see hints in
    `init()`
    """
    if iscontinuous:
        action = np.array([action_dict[action]])
    
    return action



def update_replay_buffer(replay_buffer, state, action, reward, next_state, done,
                         action_dim):
    """
    Update the replay buffer with provided input in the form:
    (state, one_hot_action, reward, next_state, done)

    Hint: the minibatch passed to do_train_step is one entry (randomly sampled)
    from the replay_buffer
    """
    # TO IMPLEMENT: append to the replay_buffer
    # ensure the action is encoded one hot
    one_hot_action = np.zeros(action_dim)
    one_hot_action[action] = 1
    
    # append to buffer
    replay_buffer.append([state, one_hot_action, reward, next_state, done])
    
    # Ensure replay_buffer doesn't grow larger than REPLAY_SIZE
    if len(replay_buffer) > REPLAY_SIZE:
        replay_buffer.pop(0)
    return None

def do_train_step_with_target_net(replay_buffer, state_in, state_in_t,
                           action_in, action_in_t,
                           target_in, target_in_t,
                           q_values, q_values_t,
                           q_selected_action, q_selected_action_t,
                           loss, loss_t,
                           optimise_step, optimise_step_t,
                           train_loss_summary_op, train_loss_summary_op_t,
                           batch_presentations_count):
    """
    Same structure as do_train_step, but implement double q here.
    """
    # get train batch
    minibatch = random.sample(replay_buffer, BATCH_SIZE)

    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]

    target_batch = []

    # use target net get target values
    Q_value_batch = q_values.eval(feed_dict={
        state_in: next_state_batch
    })

    Q_value_batch_t = q_values_t.eval(feed_dict={
        state_in_t: next_state_batch
    })

    for i in range(0, BATCH_SIZE):
        sample_is_done = minibatch[i][4]
        if sample_is_done:
            target_batch.append(reward_batch[i])
        else:
            # TO IMPLEMENT: set the target_val to the correct Q value update
            action_by_eval = np.argmax(Q_value_batch[i])

            target_val = reward_batch[i] + GAMMA * Q_value_batch_t[i][action_by_eval]
            # target_val = reward_batch[i] + GAMMA * np.max(Q_value_batch_t[i])
            target_batch.append(target_val)
    
    # train loss on eval net
    summary, _ = session.run([train_loss_summary_op, optimise_step], feed_dict={
        target_in: target_batch,
        state_in: state_batch,
        action_in: action_batch
    })
    writer.add_summary(summary, batch_presentations_count)
    
    

def do_train_step(replay_buffer, state_in, action_in, target_in,
                  q_values, q_selected_action, loss, optimise_step,
                  train_loss_summary_op, batch_presentations_count):
    target_batch, state_batch, action_batch = \
        get_train_batch(q_values, state_in, replay_buffer)

    summary, _ = session.run([train_loss_summary_op, optimise_step], feed_dict={
        target_in: target_batch,
        state_in: state_batch,
        action_in: action_batch
    })
    writer.add_summary(summary, batch_presentations_count)


def get_train_batch(q_values, state_in, replay_buffer):
    """
    Generate Batch samples for training by sampling the replay buffer"
    Batches values are suggested to be the following;
        state_batch: Batch of state values
        action_batch: Batch of action values
        target_batch: Target batch for (s,a) pair i.e. one application
            of the bellman update rule.

    return:
        target_batch, state_batch, action_batch

    Hints:
    1) To calculate the target batch values, you will need to use the
    q_values for the next_state for each entry in the batch.
    2) The target value, combined with your loss defined in `get_network()` should
    reflect the equation in the middle of slide 12 of Deep RL 1 Lecture
    notes here: https://webcms3.cse.unsw.edu.au/COMP9444/17s2/resources/12494
    """
    minibatch = random.sample(replay_buffer, BATCH_SIZE)

    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]

    target_batch = []
    Q_value_batch = q_values.eval(feed_dict={
        state_in: next_state_batch
    })
    for i in range(0, BATCH_SIZE):
        sample_is_done = minibatch[i][4]
        if sample_is_done:
            target_batch.append(reward_batch[i])
        else:
            # TO IMPLEMENT: set the target_val to the correct Q value update
            target_val = reward_batch[i] + GAMMA * np.max(Q_value_batch[i])
            
            target_batch.append(target_val)
    return target_batch, state_batch, action_batch


def qtrain(env, state_dim, action_dim,
           state_in, action_in, target_in, q_values, q_selected_action,
           loss, optimise_step, train_loss_summary_op,
           num_episodes=NUM_EPISODES, ep_max_steps=EP_MAX_STEPS,
           test_frequency=TEST_FREQUENCY, num_test_eps=NUM_TEST_EPS,
           final_epsilon=FINAL_EPSILON, epsilon_decay_steps=EPSILON_DECAY_STEPS,
           force_test_mode=False, render=True):
    global epsilon
    # Record the number of times we do a training batch, take a step, and
    # the total_reward across all eps
    batch_presentations_count = total_steps = total_reward = 0

    # average reward
    reward_100 = []

    # old things are eval Q network
    # new target Q network fom get_target_network()
    state_in_t, action_in_t, target_in_t, q_values_t, q_selected_action_t, \
           loss_t, optimise_step_t, train_loss_summary_op_t = get_target_network(state_dim, action_dim)

    # replace the target net with eval param
    target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
    eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

    target_replace_op = [tf.assign(t, e) for t, e in zip(target_params, eval_params)]

    # initialize
    session.run(tf.global_variables_initializer())
    
    for episode in range(num_episodes):
        # initialize task
        state = env.reset()
        if render: env.render()

        # Update epsilon once per episode - exp decaying
        epsilon -= (epsilon - final_epsilon) / epsilon_decay_steps

        # in test mode we set epsilon to 0
        test_mode = force_test_mode or \
                    ((episode % test_frequency) < num_test_eps and
                        episode > num_test_eps
                    )
        if test_mode: print("Test mode (epsilon set to 0.0)")

        ep_reward = 0
        for step in range(ep_max_steps):
            total_steps += 1

            # get an action and take a step in the environment
            action = get_action(state, state_in, q_values, epsilon, test_mode,
                                action_dim)
            env_action = get_env_action(action)
            
            next_state, reward, done, _ = env.step(env_action)
            ep_reward += reward

            # display the updated environment
            if render: env.render()  # comment this line to possibly reduce training time

            # add the s,a,r,s' samples to the replay_buffer
            update_replay_buffer(replay_buffer, state, action, reward,
                                 next_state, done, action_dim)
            
            state = next_state

            # perform a training step if the replay_buffer has a batch worth of samples
            if len(replay_buffer) > BATCH_SIZE:
##                do_train_step(replay_buffer, state_in, action_in, target_in,
##                              q_values, q_selected_action, loss, optimise_step,
##                              train_loss_summary_op, batch_presentations_count)

                
                do_train_step_with_target_net(replay_buffer, state_in, state_in_t,
                           action_in, action_in_t,
                           target_in, target_in_t,
                           q_values, q_values_t,
                           q_selected_action, q_selected_action_t,
                           loss, loss_t,
                           optimise_step, optimise_step_t,
                           train_loss_summary_op, train_loss_summary_op_t,
                           batch_presentations_count)

                # every 5 step, replace the target param
                if step % 3 == 0:
                    session.run(target_replace_op)
                
                batch_presentations_count += 1

            if done:
                break
        total_reward += ep_reward

        # for last 100 ave reward
        reward_100 += [ep_reward]
        if len(reward_100) > 100:
            reward_100.pop(0)
        
        
        test_or_train = "test" if test_mode else "train"
        print("end {0} episode {1}:\n\tep reward: {2}\n\tave reward: {3}\n\t\
last 100 ave reward: {4}\n\tBatch presentations: {5}\n\tepsilon: {6}".format(
            test_or_train, episode, ep_reward, total_reward / (episode + 1),
            np.array(reward_100).mean(),
            batch_presentations_count, epsilon
        ))


def setup():
    default_env_name = 'CartPole-v0'
    # default_env_name = 'MountainCar-v0'
    # default_env_name = 'Pendulum-v0'
    # if env_name provided as cmd line arg, then use that
    env_name = sys.argv[1] if len(sys.argv) > 1 else default_env_name
    env = gym.make(env_name)
    state_dim, action_dim = init(env, env_name)
    network_vars = get_network(state_dim, action_dim)
    init_session()
    return env, state_dim, action_dim, network_vars


def main():
    env, state_dim, action_dim, network_vars = setup()
    qtrain(env, state_dim, action_dim, *network_vars, render=False)
        

if __name__ == "__main__":
    main()
