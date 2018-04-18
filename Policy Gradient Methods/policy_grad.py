import numpy as np
import tensorflow as tf
import itertools
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

from pycolab import ascii_art
from pycolab.prefab_parts import sprites as prefab_sprites

epsilon = 1e-6

class PlayerSprite(prefab_sprites.MazeWalker):
    def __init__(self, corner, position, character):
        """Inform superclass that we can go anywhere, but not off the board."""
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable='#', confined_to_board=True)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del layers, backdrop, things  # Unused.

        # Apply motion commands.
        if actions == 0:  # walk upward?
            self._north(board, the_plot)
        elif actions == 1:  # walk downward?
            self._south(board, the_plot)
        elif actions == 2:  # walk leftward?
            self._west(board, the_plot)
        elif actions == 3:  # walk rightward?
            self._east(board, the_plot)
        else:
            # All other actions are ignored. Although humans using the CursesUi can
            # issue action 4 (no-op), agents should only have access to actions 0-3.
            # Otherwise staying put is going to look like a terrific strategy.
            return

        # See if the game is over.
        if self.position[0] == 0 and self.position[1] == 8:
            the_plot.add_reward(1.0)
            the_plot.terminate_episode()
        else:
            the_plot.add_reward(0.0)

class MazeEnv:
    def __init__(self, maze_index=0, useRandomMaze=False):
        self.ENV_GAME_ART = [
            ['.........',
             '.........',
             '.........',
             '########.',
             '.........',
             '...P.....'],

            ['.........',
             '.........',
             '.........',
             '.########',
             '.........',
             '...P.....'],

            ['.........',
             '.........',
             '.........',
             '.#######.',
             '.........',
             '...P.....']
        ]

        self.maze_index = np.random.randint(len(self.ENV_GAME_ART)) if useRandomMaze else maze_index

        self.game = self.make_game()

        self.observation_space_n = self.game.rows * self.game.cols

        self.actions = [0, 1, 2, 3]
        self.action_space_n = len(self.actions)

        self.episode = 0

    def make_game(self):
        """Builds and returns a cliff-walk game."""
        return ascii_art.ascii_art_to_game(
            self.ENV_GAME_ART[self.maze_index], what_lies_beneath='.',
            sprites={'P': PlayerSprite})

    def get_state(self, obs):
        allstates = np.array(obs.layers['P'], dtype=np.float)
        s = np.argmax(allstates, axis=None)
        return s

    def reset(self):
        self.game = self.make_game()  # blockingMaze.make_game(self.mazeIndex)
        self.episode = 0
        obs, r, gamma = self.game.its_showtime()
        return self.get_state(obs)

    def step(self, action):
        self.episode += 1
        obs, reward, gamma = self.game.play(action)
        next_state = self.get_state(obs)
        return next_state, reward, self.game.game_over

# define env
env = MazeEnv(useRandomMaze=True)

class EpisodeStats:
    def __init__(self, num_episodes):
        self.lengths = np.zeros(num_episodes)
        self.rewards = np.zeros(num_episodes)
        self.actions = [[] for _ in range(num_episodes)]

    def __str__(self):
        return "episode_lengths: " + str(self.lengths) + \
                "\nepisode_actions: " + str(self.actions)

    def add(self, i, reward, t, action):
        self.rewards[i] += reward
        self.lengths[i] = t
        self.actions[i].append(action)

class PolicyEstimator():
    def __init__(self,
                 alpha_theta=0.9,
                 lambda_trace=0.5,
                 gamma=1.0,
                 useTrace=False,
                 tag=""
                 ):
        self.scope = "policy_estimator-" + tag
        self.alpha_theta = alpha_theta
        self.lambda_trace = lambda_trace
        self.gamma = gamma
        self.useTrace = useTrace

        self._theta = np.zeros((env.observation_space_n, env.action_space_n), np.float32)
        self._trace = np.zeros((env.observation_space_n, env.action_space_n), np.float32)

        with tf.variable_scope(self.scope):
            # inputs
            self.state = tf.placeholder(tf.int32, [], "state")
            self.action = tf.placeholder(tf.int32, name="action")
            self.delta = tf.placeholder(tf.float32, name="delta")
            self.I = tf.placeholder(tf.float32, name="I")

            self.theta = tf.placeholder(tf.float32, name="theta",
                                        shape=(env.observation_space_n, env.action_space_n))
            self.trace = tf.placeholder(tf.float32, name="trace",
                                        shape=(env.observation_space_n, env.action_space_n))

            # prediction  computation
            state_one_hot = tf.one_hot(self.state, env.observation_space_n)
            self.action_probs = tf.squeeze(tf.nn.softmax(tf.tensordot(state_one_hot, self.theta, axes=1)))

            # updates
            action_prob = tf.log(tf.gather(self.action_probs, [self.action]))
            action_prob_grad = tf.gradients(action_prob, [self.theta])

            self.trace_new = tf.squeeze(self.gamma * self.lambda_trace * self.trace + self.I * action_prob_grad)

            if useTrace:
                self.theta_new = tf.squeeze(self.theta + self.alpha_theta * self.delta * self.trace_new)
            else:
                self.theta_new = tf.squeeze(self.theta + self.alpha_theta * self.I * self.delta * action_prob_grad)

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, {self.state: state, self.theta: self._theta})

    def update(self, state, delta, action, I, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state,
                     self.delta: delta,
                     self.action: action,
                     self.I: I,
                     self.theta: self._theta,
                     self.trace: self._trace}
        if self.useTrace:
            self._theta, self._trace = sess.run([self.theta_new, self.trace_new], feed_dict)
        else:
            self._theta = sess.run(self.theta_new, feed_dict)



class ValueEstimator():
    def __init__(self,
                 alpha_w=0.9,
                 lambda_trace=0.5,
                 gamma=1.0,
                 useTrace=False,
                 tag=""
                 ):
        self.scope = "value_estimator-" + tag
        self.alpha_w = alpha_w
        self.lambda_trace = lambda_trace
        self.gamma = gamma
        self.useTrace = useTrace

        self._w = np.zeros((env.observation_space_n), np.float32)
        self._trace = np.zeros((env.observation_space_n), np.float32)

        with tf.variable_scope(self.scope):
            # inputs
            self.state = tf.placeholder(tf.int32, [], "state")
            self.delta = tf.placeholder(tf.float32, name="delta")
            self.I = tf.placeholder(tf.float32, name="I")
            self.w = tf.placeholder(tf.float32, name="w", shape=(env.observation_space_n,))
            self.trace = tf.placeholder(tf.float32, name="trace", shape=(env.observation_space_n,))

            # prediction computation
            state_one_hot = tf.one_hot(self.state, env.observation_space_n)
            self.value = tf.squeeze(tf.reduce_sum(tf.multiply(state_one_hot, self.w)))

            # updates
            grad_v = tf.gradients(self.value, [self.w])

            if useTrace:
                self.trace_new = tf.squeeze(self.gamma * self.lambda_trace * self.trace + self.I * grad_v)
                self.w_new = tf.squeeze(self.w + self.alpha_w * self.delta * self.trace_new)
            else:
                self.w_new = tf.squeeze(self.w + self.I * self.alpha_w * self.delta * grad_v)

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.w: self._w}
        return sess.run(self.value, feed_dict)

    def update(self, state, delta, I, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state,
                     self.delta: delta,
                     self.I: I,
                     self.w: self._w,
                     self.trace: self._trace}
        if self.useTrace:
            self._w, self._trace = sess.run([self.w_new, self.trace_new], feed_dict)
        else:
            self._w = sess.run(self.w_new, feed_dict)


class ActorCritic:
    def __init__(self,
                 env,
                 policy_estimator,
                 value_estimator,
                 gamma=0.99,
                 num_episodes=100,
                 max_iters_per_ep=10000
                 ):
        # variables
        self.env = env
        self.policy_estimator = policy_estimator
        self.value_estimator = value_estimator
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.max_iters_per_ep = max_iters_per_ep

        self.stats = EpisodeStats(self.num_episodes)

    def run(self):
        for i_episode in range(self.num_episodes):
            state = env.reset()
            I = 1.0

            for t in itertools.count():

                action_probs = self.policy_estimator.predict(state)

                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, R, done = env.step(action)

                self.stats.add(i_episode, R, t, action)

                value_crt = self.value_estimator.predict(state)
                value_next = self.value_estimator.predict(next_state)

                td_delta = R + self.gamma * value_next - value_crt

                # dont update when its insignificant
                if np.abs(td_delta) > epsilon:
                    self.value_estimator.update(state, td_delta, I)
                    self.policy_estimator.update(state, td_delta, action, I)

                if t % 10 == 0:
                    print("\rStep {} @ Episode {}/{} ({})".format(
                        t, i_episode + 1, self.num_episodes, self.stats.rewards[i_episode - 1]), end="")

                if done or t > self.max_iters_per_ep:
                    break

                I *= self.gamma
                state = next_state


def plotstats(test_name,
              stats_dict,
              num_episodes,
              max_savgol_winsize=151,
              min_savgol_winsize=15
              ):

    # iterator table of available colors
    class ColorIterator:
        def __init__(self):
            self.colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                      'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                      'tab:olive', 'tab:cyan']
            self.index = 0
        def next(self):
            color = self.colors[self.index]
            self.index = (self.index + 1) % len(self.colors)
            return color

    colors = ColorIterator()

    fig, ax1 = plt.subplots()

    # setup visuals
    ax1.set_xlabel('Episode')
    ax1.set_ylabel("Log Episode Length")

    # process data
    episodes = np.arange(num_episodes)

    # determine smoothing window size
    savgol_winsize = int(num_episodes / 2)
    savgol_winsize = min_savgol_winsize if savgol_winsize < min_savgol_winsize else savgol_winsize
    savgol_winsize = max_savgol_winsize if savgol_winsize > max_savgol_winsize else savgol_winsize
    savgol_winsize = savgol_winsize + 1 if savgol_winsize % 2 == 0 else savgol_winsize  # ensure odd

    print("Plotting results; smoothed with {}-wide savgol filter.".format(savgol_winsize))

    for stats_name in stats_dict:
        color = colors.next()

        stats = stats_dict[stats_name]

        log_length = np.log10(stats.lengths)
        log_length_smooth = savgol_filter(log_length, savgol_winsize, 4)

        ax1.plot(episodes, log_length, 'o--', color=color, markersize=2, alpha=0.15)
        ax1.plot(episodes, log_length_smooth, 'o--', color=color, markersize=2, alpha=0.7, label=stats_name)

    fig.tight_layout()
    ax1.legend()
    plt.title(test_name)
    plt.show()


class TestConfig:
    def __init__(self,
                 use_trace_policy,
                 use_trace_value,
                 gamma,
                 lambda_trace
                 ):
        self.use_trace_policy = use_trace_policy
        self.use_trace_value = use_trace_value
        self.gamma = gamma
        self.lambda_trace = lambda_trace

    def __str__(self):
        def bool2str(b):
            return "T" if b else "F"

        def float2str(f, precision=4):
            return "0." + str(int((10 ** precision) * f))

        return "{}{}-{}-{}".format(
            bool2str(self.use_trace_policy),
            bool2str(self.use_trace_value),
            float2str(self.gamma),
            float2str(self.lambda_trace),
        )


def runSingleTest(policy_estimator, value_estimator, gamma, num_episodes, sess):
    sess.run(tf.global_variables_initializer())

    actor_critic = ActorCritic(env,
                               policy_estimator,
                               value_estimator,
                               gamma=gamma,
                               num_episodes=num_episodes)

    actor_critic.run()

    return actor_critic.stats

def runMultiTests(num_episodes_per_test, configurations):
    # run cpu only -> this is actually faster on my machine
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )

    tf.reset_default_graph()

    stats_dict = {} # record results here
    with tf.Session(config=config) as sess:
        for config in configurations:

            config_name = str(config)

            print("Running tests with configuration: {}".format(config_name))

            policy_estimator = PolicyEstimator(gamma=config.gamma,
                                               lambda_trace=config.lambda_trace,
                                               useTrace=config.use_trace_policy,
                                               tag=config_name)

            value_estimator = ValueEstimator(gamma=config.gamma,
                                             lambda_trace=config.lambda_trace,
                                             useTrace=config.use_trace_value,
                                             tag=config_name)

            stats_dict[config_name] = runSingleTest(policy_estimator,
                                                    value_estimator,
                                                    config.gamma,
                                                    num_episodes_per_test,
                                                    sess)

            print("")  # print a new line

    return stats_dict

def runTest1():
    """
    this test compares the efficiency of the algorithm when traces are used vs not used
    """
    test_name = "Effect of using Traces"

    num_episodes_per_test = 100

    gamma = 0.99
    lambda_trace = 0.5

    # see TestConfig class for how to change test configurations
    configurations = [
        TestConfig(True, True, gamma, lambda_trace),
        TestConfig(False, False, gamma, lambda_trace)
    ]

    # run tests and plot results
    stats_dict = runMultiTests(num_episodes_per_test, configurations)
    plotstats(test_name, stats_dict, num_episodes_per_test)

def runTest2():
    """
    this test compares different values of lambda
    """
    test_name = "Effect of using Traces"

    num_episodes_per_test = 100

    lambda_trace = 0.5

    # see TestConfig class for how to change test configurations
    # NOTE: small gammas do not allow to converge
    gammas = [0.9, 0.95, 0.99, 0.999, 0.9999]
    configurations = [TestConfig(True, True, gamma, lambda_trace) for gamma in gammas]

    # run tests and plot results
    stats_dict = runMultiTests(num_episodes_per_test, configurations)
    plotstats(test_name, stats_dict, num_episodes_per_test)


# run the experiments
def main():
    # runTest1()
    runTest2()

if __name__=="__main__":
    main()



