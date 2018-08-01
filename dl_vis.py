import numpy as np
import matplotlib.pyplot as plt

def extract_params_pytorch(net):
    """
    Extract weights and biases from a pytorch network.
    Returns a tuple(weights, biases) of numpy arrays

    net: pytorch network
    """
    net = net.cpu()

    # extracting weights, biases
    weights = np.concatenate([p.data.numpy().flatten()
        for n, p in net.named_parameters() if 'weight' in n])
    biases = np.concatenate([p.data.numpy().flatten()
        for n, p in net.named_parameters() if 'bias' in n])

    return weights, biases

class Plot():
    """
    Helper class to plot training progress of neural networks
    """
    def __init__(self):
        self.loss_train = np.empty([0])
        self.loss_test = np.empty([0])

        self.perc_w = np.empty([0, 9])
        self.perc_b = np.empty([0, 9])

        self.steps = np.empty([0])

        # initialize matplotlib
        plt.rc('axes', axisbelow=True)
        plt.rc('axes', facecolor='lightgray')

        # setup columns
        fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
        self.loss = axes[0]
        self.ws = axes[1]
        self.bs = axes[2]
        
        fig.canvas.set_window_title('Training Progress')
        
        # init subplots
        self.plot_loss()
        self.plot_ws()
        self.plot_bs()
        
        # set layout and show window
        plt.tight_layout()
        plt.show(block=False)
        
    def percentiles(self, arr):
        """
        Computes 9 percentiles of the data

        arr: array of the distribution values
        """
        ps = np.empty([9])
        for i in range(9):
            ps[i] = np.percentile(arr, 10.0 * i + 10.0)

        return ps

    def plot_loss(self):
        """
        Plot the loss axis
        """
        self.loss.clear()
        self.loss.set_title('Loss')
        self.loss.grid(linestyle='dotted', color='white')

        self.loss.plot(self.steps, self.loss_train, label='training loss',
                       color='red')
        self.loss.plot(self.steps, self.loss_test, label='test loss',
                       color='blue')
        self.loss.legend()

    def plot_ws(self):
        """
        Plot the weights axis
        """
        self.ws.clear()
        self.ws.set_title('Weight Distribution')
        self.ws.grid(linestyle='dotted', color='white')

        for i in range(5):
            self.ws.fill_between(self.steps, self.perc_w.T[i],
                                 self.perc_w.T[8 - i], color='red', alpha=0.2)

    def plot_bs(self):
        """
        Plot the biases axis
        """
        self.bs.clear()
        self.bs.set_title('Bias Distribution')
        self.bs.grid(linestyle='dotted', color='white')

        for i in range(5):
            self.bs.fill_between(self.steps, self.perc_b.T[i],
                                 self.perc_b.T[8 - i], color='red', alpha=0.2)

    def plot(self, i, loss_train, loss_test, weights, biases):
        """
        Plot new values for the next iteration

        i:          iteration
        loss_train: new training loss
        loss_test:  new test loss

        weights:    numpy array of weights
        biases:     numpy array of biases
        """
        # update members
        self.steps = np.append(self.steps, [i])
        self.loss_train = np.append(self.loss_train, [loss_train])
        self.loss_test = np.append(self.loss_test, [loss_test])
        self.perc_w = np.vstack([self.perc_w, self.percentiles(weights)])
        self.perc_b = np.vstack([self.perc_b, self.percentiles(biases)])

        # plot
        self.plot_loss()
        self.plot_ws()
        self.plot_bs()

        plt.tight_layout()
        plt.pause(0.05)

        # print values
        print('Iteration %i' % (i))
        print('Loss:     train data: %f,    \ttest data: %f' %
              (loss_train, loss_test))
        print('Weights:  mean:       %f,    \tstdev:     %f' %
              (np.mean(weights), np.std(weights)))
        print('Biases:   mean:       %f,    \tstdev:     %f' %
              (np.mean(biases), np.std(biases)))
        print('-----------------------------------------------------------')

if __name__ == '__main__':
    from math import sqrt
    from random import uniform
    def train(i):
        """
        Generate some dummy data to test the Plot class
        
        i: training iteration

        Returns a tuple(loss_train, loss_test, weights, biases)
        """

        loss_train = 1.0 / (i + 1) + uniform(-0.01, 0.01)
        loss_test = 1.0 / (i + 1) + 0.0025 * i + uniform(-0.01, 0.01)
        

        weights = np.random.normal(0.0, 0.01 + 0.01 * sqrt(i), (1024,))
        
        biases = np.random.normal(0.0, 0.01, (1024,))
        biases += np.random.uniform(0, 0.01 * sqrt(i), (1024,))

        return (loss_train, loss_test, weights, biases)
    

    # training our fake network
    plot = Plot()

    # training loop
    for i in range(51):
        # train the network
        loss_train, loss_test, weights, biases = train(i)

        # plot the progress
        plot.plot(i, loss_train, loss_test, weights, biases)
    
    # keep plotting window open after the learning is finished
    plt.show()
