# PCHID_code
Code for Policy Continuation with Hindsight Inverse Dynamics (PCHID)

## There are three environments
- Bitflip

The bitflip environment was introduced by [Hindsight experience replay (HER)](http://papers.nips.cc/paper/7090-hindsight-experience-replay)	

Our implementation of bitflip and HER refers [RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2/blob/master/9.her.ipynb) and [Baselines](https://github.com/openai/baselines/tree/master/baselines/her)

- GridWorld

Our experiment on the GridWorld domain is based on previous work of [Value Iteration Networks (VIN)](https://papers.nips.cc/paper/6046-value-iteration-networks.pdf) and the [PyTorch implementation of VIN](https://github.com/kentsommer/pytorch-value-iteration-networks)

- FetchReach

The FetchReach environment is one of four OpenAI [Fetch envs](https://github.com/openai/gym/tree/master/gym/envs/robotics/fetch). As for the other three environments (Push, Slide, PickAndPlace), synchoronous improvement is needed to further improve PCHID's learning efficiency.
