# dl-visu
Script for the Visualization of Deep Learning Progress
![preview](https://github.com/titus-leistner/dl-visu/blob/master/dl_vis.gif)

# Requirements
* python3
* virtualenv

# Installation
For the installation I recommend virtualenv.

```sh
git clone git@github.com:titus-leistner/dl-visu.git
cd dl-visu/
python3 -m venv .
source bin/activate
pip install -r requirements.txt
```

# Usage
To test the visualization, simply run the script
```sh
python dl_visu.py
```

```python
# initialize
plot = Plot()

# training loop
for i in range(101):
    # train your network
    loss_train, loss_test, weights, biases = train(i)

    # plot the progress
    plot.plot(i, loss_train, loss_test, weights, biases)
```
