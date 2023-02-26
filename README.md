# OmniAirl

This is a trustworthy reinforcement learning library only for XJTU IAIR homework, currently maintained by Jiayi Zhou from Xi'an Jiaotong University. Contributions are welcome.

![Logo](https://github.com/Gaiejj/omniairl/blob/main/images/logo.png)

## Installation

The following libraries are required for installation:

- numpy
- yaml

To install the dependencies, run:
```bash
git clone https://github.com/Gaiejj/omniairl.git
cd omniairl
pip install -r requirements.txt
cd examples
python Q_Learning_K_Bandits.py
```

## Algorithms

Currently, the following reinforcement learning algorithms have been implemented:

- Tabular Q-Learning

## Enviornment

- K-Armed Bandit

## Result


| Fig1: Epoch Reward | Fig2: Rolling Average Reward (K=5) |
|--------------------|-------------------------------------|
| ![Epoch Reward](https://github.com/Gaiejj/omniairl/blob/main/images/Q_Learning_K_Bandits/window_0.png "Epoch Reward") | ![Rolling Average Reward (K=5)](https://github.com/Gaiejj/omniairl/blob/main/images/Q_Learning_K_Bandits/window_5.png "Rolling Average Reward (K=5)") |

| Fig3: Rolling Average Reward (K=10) | Fig4: Rolling Average Reward (K=20) |
|----------------------------------------|----------------------------------------|
| ![Rolling Average Reward (K=10)](https://github.com/Gaiejj/omniairl/blob/main/images/Q_Learning_K_Bandits/window_10.png "Rolling Average Reward (K=10)") | ![Rolling Average Reward (K=20)](https://github.com/Gaiejj/omniairl/blob/main/images/Q_Learning_K_Bandits/window_20.png "Rolling Average Reward (K=20)") |



## Contributing

Contributions to this project are welcome! If you find a bug or would like to propose a new feature, please open an issue or submit a pull request.

### Contributors

Here's a list of the current contributors to the project:

- Jiayi Zhou

## License

This project is licensed under the MIT License - see the LICENSE file for details.

