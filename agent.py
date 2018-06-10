class Agent(object):

    def __init__(
        self, env, actor, critic, target_actor=None, target_critic=None):
        self._env = env
        self._actor = actor
        self._critic = critic
        self._target_actor = target_actor
        self._target_critic = target_critic

        # 1) The environment gives the agent its current state.
        self._state = env.reset()

    def _run_helper(self, episodes=1, episode_max_length=1):
        # 2) The agent asks the actor what action it should take given the
        # current state.
        action = self._actor.predict(self._state)

        # 3) The agent uses that action to interact with the environment.
        # This interaction yields a reward, the next state, and whether
        # the next state is terminal.
        next_state, reward, done, _ = self._env.step(action)

        actions_for_next_state = self._target_actor.predict(next_state)
        next_state_q_value = self._target_critic.predict(next_state, actions_for_next_state)

        target_value = reward if done else reward + next_state_q_value

        # 4) The agent uses the reward to train the critic.
        self._critic.train(self._state, action, target_value)

        # 5) The trained critic can tell the actor how to select even better
        # actions in the future. This is done by looking at how the "goodness"
        # value returned by the critic changes as we fiddle (increase/decrease)
        # with the action (the critic's gradient).
        gradient = self._critic.compute_gradient(self._state, action)

        # 6) The gradient is then used to train the actor.
        self._actor.train(self._state, gradient)

        # Current state is updated.
        self._state = next_state

        return done

    def run(self, episodes=1, episode_max_length=1):
        for _ in range(episodes):
            # Episodes must start in initial state.
            self._env.reset()
            for _ in range(episode_max_length):
                if self._run_helper():
                    break

