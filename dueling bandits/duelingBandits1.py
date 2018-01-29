from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np


class DuelBandit:
    def __init__(self, kArm=10, T=1000, trueReward=0, max_time_steps = 1000000):
    
        self.k = kArm
        self.trueReward = trueReward
        self.T = T
        self.max_time_steps = max_time_steps
        
        # initial real reward for each action based on N((0,1)
        self.arm_reward = []
        for i in range(0, self.k):
            self.arm_reward.append(np.random.randn() + trueReward)
            
        # find the best action 
        self.best_action = np.argmax(self.arm_reward)
        # print("True values of action sets: ",self.armReward)
        
        # initial parameters required for EXPLORE algorithm
        self.sigma = 1/self.T/self.k**2
        
        # random pick the first action as b_hat, remove b_hat from action set W
        self.b_hat = np.random.randint(0,self.k)
        self.W = [b for b in range(self.k) if b!=self.b_hat]
        
        # initial winning probability P and confidence interval C, and comparison counter t
        self.P = [self.k*[0.5] for _ in range(self.k)]
        self.t = [self.k*[0] for _ in range(self.k)]
        self.C = [self.k*[[0,1]] for _ in range(self.k)]
        
        # initial step count
        self.step = 0

        # regret
        self.total_regret_strong = 0
        self.total_regret_weak = 0

    def duelIF1(self):
        t = 0
        while len(self.W)>0 and t < self.max_time_steps:
            #if step%10 == 0:
                #print("Progress {}, len(W) {}, b_hat {}\n".format(step, len(self.W), self.b_hat))
            for b in self.W:
                P1, P2 = self.P[self.b_hat][b], self.P[b][self.b_hat]
                t = self.t[self.b_hat][b]
                reward1 = np.random.randn() + self.arm_reward[self.b_hat]
                reward2 = np.random.randn() + self.arm_reward[b]
                self.step += 1
                if reward1 > reward2:
                    self.P[self.b_hat][b] = (P1 * t + 1)/(t + 1)
                    self.t[self.b_hat][b] += 1
                    self.P[b][self.b_hat] = (P2 * t)/(t + 1)
                    self.t[b][self.b_hat] += 1
                else:
                    self.P[self.b_hat][b] = (P1 * t) / (t + 1)
                    self.t[self.b_hat][b] += 1
                    self.P[b][self.b_hat] = (P2 * t + 1) / (t + 1)
                    self.t[b][self.b_hat] += 1
                c_t = np.sqrt(np.log(1 / self.sigma) / (t + 1))
                self.C[self.b_hat][b] = [self.P[self.b_hat][b] - c_t, self.P[self.b_hat][b] + c_t]
                self.C[b][self.b_hat] = [self.P[b][self.b_hat] - c_t, self.P[b][self.b_hat] + c_t]

                # -1 and -0.5 because P = Epsilon + 0.5
                self.total_regret_strong += self.P[self.best_action][b] + self.P[self.best_action][self.b_hat] - 1
                self.total_regret_weak += max(self.P[self.best_action][b], self.P[self.best_action][self.b_hat]) - 0.5

            for b in self.W:
                cc = self.C[self.b_hat][b]
                if (self.P[self.b_hat][b] > 0.5) and (not ((0.5 > cc[0]) and (0.5 < cc[1]))):
                    self.W = [x for x in self.W if x != b]
                if (self.P[self.b_hat][b] < 0.5) and (not ((0.5 > cc[0]) and (0.5 < cc[1]))):
                    self.b_hat = b
                    self.W = [x for x in self.W if x != b]


def dualSimulation(nBandits, time, kArms=10, num_t_buckets=10):
    t_buckets = np.zeros(num_t_buckets, np.uint) # log10 scale, histogram of number of comparisons
    regrets = []
    correct = 0
    for _ in range(0, nBandits):
        bandit = DuelBandit(kArm=kArms, T=time)
        bandit.duelIF1()
        # print("Best action is {}, Result found is {}, Time used is {}".format(bandit.bestAction,bandit.b_hat,bandit.step))
        #print("len w is {}".format(len(bandit.W)))
        if bandit.best_action == bandit.b_hat:
            correct += 1
        t_buckets[int(np.log10(bandit.step)) % num_t_buckets] += 1
        regrets.append([bandit.total_regret_strong, bandit.total_regret_weak])
        if _ % 10 == 0:
            print("\rCompleted {}/{} iterations.".format(_, tests), end="")
    print("")
    return correct, t_buckets, regrets

tests = 100
correct, t_buckets, regrets = dualSimulation(tests, 1000, 25)
print("correctiness: {}/{}={}".format(correct, tests, correct/tests))
print(t_buckets)
print(regrets)
