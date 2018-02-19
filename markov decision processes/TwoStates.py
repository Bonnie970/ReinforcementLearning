import random
import numpy as np

verbose = False

class MDP:
    # generic mdp class
    def __init__(self,
                 gamma):
        self.states = []
        self.actions = []
        self.p = []
        self.r = None
        self.v = []
        self.gamma = gamma

class TwoStateMdp(MDP):
    def __init__(self,
                 gamma=0.95):
        MDP.__init__(self, gamma)

        self.states = [0, 1]
        self.actions = [0, 1, 2]

        # construct environment transition probabilities
        # 6 state-action pairs, 2 states
        # [s0-a0, s0-a1, s0-a2], [s1-a0, s1-a1, s1-a2]
        self.p = np.array([[[0.5, 0, 0], [0.5, 1, 0]], [[0, 0, 0], [0, 0, 1]]], np.float)
        # rewards r(s, a)
        self.r = np.array([[5, 10, 0], [0, 0, -1]], np.float)

        # initial value and policy
        self.v = np.array([0, 0], np.float)
        # 1 means valid action, 0 means invalid
        self.pi = np.array([[1, 1, 0], [0, 0, 1]], np.float)

        print(self.p.shape)

class GridStateMdp(MDP):
    def __init__(self,
                 gamma=0.95):
        MDP.__init__(self, gamma)

        # TODO: finish this

# policy evaluation
def e(mdp, verbose=verbose):
    threshold = 0.001
    error = 0
    flag = True
    while (flag or (error > threshold)):
        flag = False
        error = 0
        for s in mdp.states:
            v_old = mdp.v[s]
            mdp.v[s] = mdp.p[s, 0].dot(mdp.r[s] + mdp.gamma*mdp.v[0]) + mdp.p[s, 1].dot(mdp.r[s] + mdp.gamma*mdp.v[1])
            error = max(error, abs(v_old - mdp.v[s]))
        if verbose:
            print(mdp.v)
            print(error)
    i(mdp)

# policy improvement
def i(mdp):
    print('Policy is', mdp.pi)
    policy_stable = True
    for s in mdp.states:
        old_action = mdp.pi[s].copy()
        q = {}
        for a in mdp.actions:
            if old_action[a]==0:
                # invalid action
                # mdp.pi[s,a] = 0
                continue
            else:
                qa = sum([mdp.p[s,s2,a] * (mdp.r[s,a] + mdp.gamma*mdp.v[s2]) for s2 in mdp.states])
                q[a] = qa
        qm = max(q.values())
        for a in q.keys():
            if q[a]==qm:
                mdp.pi[s,a] = 1
            else:
                mdp.pi[s,a] = 0
        if (old_action != mdp.pi[s]).all():
            policy_stable = False

    if policy_stable == True:
        print('Done.')
        print('v* = ', mdp.v, '\npi* = ', mdp.pi)
    else:
        e(mdp)

mdp = TwoStateMdp()
e(mdp)

