"""Tests for DDPG agent.

Usage:
    python agent_test.py
"""
import unittest
import mock
import agent

NEXT_STATE = 2.00
ENV_STEP_RETURN = (NEXT_STATE, 5, False, None)
ENV_RESET_RETURN = 1.00
ACTOR_PREDICT_RETURN = 0
CRITIC_COMP_GRADIENT_RETURN = -1

TARGET_ACTOR_PREDICT_RETURN = 4
TARGET_CRITIC_PREDICT_RETURN = 7

class TrainerTestCase(unittest.TestCase):

    def setUp(self):
        # Create mock for env
        self._env = mock.MagicMock()
        self._env.step = mock.MagicMock(return_value=ENV_STEP_RETURN)
        self._env.reset = mock.MagicMock(return_value=ENV_RESET_RETURN)

        # Create mock for actor
        self._actor = mock.MagicMock()
        self._actor.predict = mock.MagicMock(return_value=ACTOR_PREDICT_RETURN)
        self._actor.train = mock.MagicMock()

        # Create mock for critic
        self._critic = mock.MagicMock()
        self._critic.train = mock.MagicMock()
        self._critic.compute_gradient = mock.MagicMock(
            return_value=CRITIC_COMP_GRADIENT_RETURN)

        # Create mock for target actor
        self._target_actor = mock.MagicMock()
        self._target_actor.predict = mock.MagicMock(return_value=TARGET_ACTOR_PREDICT_RETURN)
        self._target_actor.train = mock.MagicMock()

        # Create mock for target critic
        self._target_critic = mock.MagicMock()
        self._target_critic.train = mock.MagicMock()
        self._target_critic.predict = mock.MagicMock(return_value=TARGET_CRITIC_PREDICT_RETURN)

        self._agent = agent.Agent(
            self._env,
            self._actor,
            self._critic,
            target_actor=self._target_actor,
            target_critic=self._target_critic)

    def test_run(self):
        self._agent.run()

        self._env.reset.assert_called()
        self._env.step.assert_called()

        self._actor.predict.assert_called()
        self._actor.train.assert_called()

        self._critic.train.assert_called()
        self._critic.compute_gradient.assert_called()

    def test_env_step_called_with_predicted_action(self):
        self._agent.run()
        self._env.step.assert_called_with(ACTOR_PREDICT_RETURN)

    def test_critic_trained_with_correct_params(self):
        current_state = ENV_RESET_RETURN
        predicted_action = ACTOR_PREDICT_RETURN
        reward = ENV_STEP_RETURN[1] +  TARGET_CRITIC_PREDICT_RETURN

        self._agent.run()
        self._critic.train.assert_called_with(current_state, predicted_action, reward)

    def test_actor_trained_with_correct_state_and_gradient(self):
        current_state = ENV_RESET_RETURN

        self._agent.run()
        self._actor.train.assert_called_with(current_state, CRITIC_COMP_GRADIENT_RETURN)

    def test_run_with_episodes(self):
        # Reset interactions with mock environment.
        self._env.reset_mock()

        episodes = 10
        episode_max_length = 100
        self._agent.run(episodes=episodes, episode_max_length=episode_max_length)

        # Every now episode starts with the initial state (output of env.reset)
        assert self._env.reset.call_count == episodes

        # Assuming the environment won't reach a terminal state in the middle
        # of an episode.
        assert self._env.step.call_count == episodes * episode_max_length

    def test_episode_stops_when_terminal_state_is_reached(self):
        fake_next_state = 2.00
        fake_reward = 5

        # The following array entries are the responses to `env.step` call.
        self._env.step.side_effect = [
            (fake_next_state, fake_reward, False, None), # return of 1st call.
            (fake_next_state, fake_reward, False, None), # return of 2st call.
            (fake_next_state, fake_reward, True, None),  # return of 3st call.
        ]

        self._agent.run(episodes=1, episode_max_length=100)

        # Ensure agent stopped during 3rd interaction with environment and
        # did not complete the 100 interactions defined by episode_max_length.
        assert self._env.step.call_count == 3

    def test_target_actor_network_used_to_predict_next_state_action(self):
        self._agent.run()
        self._target_actor.predict.assert_called_with(NEXT_STATE)

    def test_target_critic_network_used_to_get_next_state_q_value(self):
        self._agent.run()
        self._target_critic.predict.assert_called_with(NEXT_STATE, TARGET_ACTOR_PREDICT_RETURN)


if __name__ == '__main__':
    unittest.main()