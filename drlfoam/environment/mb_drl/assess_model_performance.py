"""
here possible criteria to assess the model performance and to determine when to switch
"""
import torch as pt

from ...agent import PPOAgent
from ...constants import EPS_SP
from ...agent.agent import compute_gae


class AssessModelPerformance:
    def __init__(self, agent: PPOAgent):
        self._agent = agent

    def check(self, s_model: list, a_model: list, r_model: list) -> list:
        """
        computes the policy loss of the current MB episode for each model in the ensemble

        :param s_model: predicted states by each environment model
        :param a_model: actions predicted by policy network for each environment model
        :param r_model: predicted rewards by each environment model
        :return: policy loss wrt environment models
        """
        policy_loss = []

        # assess the policy loss for each model
        for m in range(a_model[0].size()[-1]):
            values = [self._agent.value(s[:, :, m]) for s in s_model]

            # if we have more than 1 action, a_model = n_buffer * [len_traj, n_actions, n_models],
            # else a_model = n_buffer * [n_traj, n_models]
            if len(a_model[0].size()) > 2:
                a_model_tmp = [i[:, :, m] for i in a_model]
            else:
                a_model_tmp = [i[:, m] for i in a_model]
            log_p_old = pt.cat([self._agent.policy.predict(s[:-1, :, m], a[:-1])[0] for s, a in
                                zip(s_model, a_model_tmp)])
            gaes = pt.cat([compute_gae(r[:, m], v, self._agent.gamma, self._agent.lam) for r, v in
                           zip(r_model, values)])
            gaes = (gaes - gaes.mean()) / (gaes.std(0) + EPS_SP)

            states_wf = pt.cat([s[:-1, :, m] for s in s_model])
            actions_wf = pt.cat([a[:-1] for a in a_model_tmp])
            log_p_new, entropy = self._agent.policy.predict(states_wf, actions_wf)
            p_ratio = (log_p_new - log_p_old).exp()
            policy_objective = gaes * p_ratio
            policy_objective_clipped = gaes * p_ratio.clamp(1.0 - self._agent.policy_clip, 1.0 + self._agent.policy_clip)
            policy_loss.append(-pt.min(policy_objective, policy_objective_clipped).mean().item())

        return policy_loss


if __name__ == "__main__":
    pass
