# Force
Force is a library for deep reinforcement learning (RL) research, built on [PyTorch](https://pytorch.org/) and [OpenAI Gym](https://gym.openai.com/). It is under active development. Basic features at present:
 * Implementations of various deep RL algorithms ([DDPG](https://arxiv.org/abs/1509.02971), [TD3](https://arxiv.org/abs/1802.09477), [SAC](https://arxiv.org/abs/1812.05905), [FQE](https://arxiv.org/abs/1903.08738))
 * Tools for managing configuration and launching jobs on the [Slurm](https://slurm.schedmd.com/overview.html) scheduler

The name *Force* was originally derived from the word *reinforcement*, but, by a fun coincidence, it is also related to [one of my advisors](https://ai.stanford.edu/~tengyuma/) via [Newton's second law](https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion#Newton's_second_law).

I am using Python 3.8 and PyTorch 1.4.0. I think any Python 3.6+ would be fine, although I haven't verified.
