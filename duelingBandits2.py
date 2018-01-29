import numpy as np

class Bandits:
    def __init__(self, num_bandits, time_steps, max_time_steps):
        self.num_bandits = num_bandits
        self.time_steps = time_steps
        self.max_time_steps = max_time_steps
        # self.current_time = 0

        self.rewards = np.random.rand(num_bandits) # randomize the reward value for each bandit
        # self.eps = np.zeros((time_steps, num_bandits, num_bandits), np.float) # lots of memory, can be more efficient

        # print(self.rewards)

        self.t, self.Pbb, self.bb = self.explore_IF1()

    # def time_step(self, time_step = None):
    #     return self.current_time if time_step == None else time_step

    def reset_pbb_cbb(self, t, bb, delta, W):
        c = self.confidence_interval(t, delta)

        # set everything to 1/2
        Pbb = {b : [1, 2] for b in W}
        Cbb = {b : [Pbb[b][0] / Pbb[b][1] - c, Pbb[b][0] / Pbb[b][1] + c] for b in W}

        return Pbb, Cbb

    def explore_IF1(self): # interleaved filter 1 strategy
        t = 1
        delta = 1 / self.time_steps / (self.num_bandits ** 2)

        # get a random bandit
        bb = int(np.random.rand() * self.num_bandits)

        W = [b for b in range(self.num_bandits) if b != bb]
        Pbb, Cbb = self.reset_pbb_cbb(1, bb, delta, W)

        while len(W) > 0 and t < self.max_time_steps:
            for b in W:
                # compare, get winner
                bw = self.duel(bb, b)

                # update Pbb and Cbb
                if bw == bb:
                    # bb win
                    Pbb[b][0] += 1
                    Pbb[b][1] += 1
                else:
                    # bb loss
                    Pbb[b][1] += 1

                c = self.confidence_interval(Pbb[b][1], delta)
                Cbb[b][0] = Pbb[b][0] / Pbb[b][1] - c
                Cbb[b][1] = Pbb[b][0] / Pbb[b][1] + c

                t += 1

            # print(W)
            # print("current bb", bb, "with reward", self.rewards[bb])
            # print(Pbb)
            # print(Cbb)

            for b in W:
                if Pbb[b][0] / Pbb[b][1] > 1 / 2 and not (Cbb[b][0] < 1 / 2 and Cbb[b][1] > 1 / 2):
                    W.remove(b)
                    Pbb.pop(b, None)
                    Cbb.pop(b, None)
                    # print("removed", b)

            for b in W:
                if Pbb[b][0] / Pbb[b][1] < 1 / 2 and not (Cbb[b][0] < 1 / 2 and Cbb[b][1] > 1 / 2):
                    bb = b
                    W.remove(b)

                    # reset
                    Pbb, Cbb = self.reset_pbb_cbb(1, bb, delta, W)

                    # print("Reset bb to", b)
                    # print(Pbb)
                    # print(Cbb)
                    break

        return t, Pbb, bb

    # returns the winner of the duel between the two bandits
    def duel(self, i, j):
        return i if np.random.rand() < self.P_Bradley_Terry(i, j) else j

    def P_Bradley_Terry(self, i, j):
        return self.rewards[i] / (self.rewards[i] + self.rewards[j])

    def confidence_interval(self, t, delta):
        return np.sqrt(np.log(1 / delta) / t)

    # def P(self, i, j, time_step = None):
    #     time_step = self.time_step(time_step)
    #     return self.eps[time_step, i, j] + 1 / 2

    # returns the index of the actual best policy
    def best_bandit_actual(self):
        return np.argmax(self.rewards)

def main():
    tests = 100
    num_bandits = 10
    time_steps = 1000
    max_time_steps = 1000000

    correct = 0
    for i in range(tests):
        bandits = Bandits(num_bandits, time_steps, max_time_steps)
        if bandits.bb == np.argmax(bandits.rewards):
            correct += 1

        # for b in range(num_bandits):
        #     print("Pbb for bandit", b)
        #     print(bandits.Pbb[b])

        # print("Finished after", bandits.t, "time steps.")
        # print("Computed best bandit:", bandits.bb)
        # print("Actual best bandit:", np.argmax(bandits.rewards))

        if i % 10 == 0:
            print("\rCompleted {}/{} iterations.".format(i, tests), end="")

    print("")
    print("Correctness:", correct/tests)

if __name__=="__main__":
    main()