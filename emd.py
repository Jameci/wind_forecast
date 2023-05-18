# 导入工具包
import numpy as np
from PyEMD import EEMD, EMD, Visualisation
import pylab as plt





def define_signal(y):
    t = np.array([i for i in range(len(y))])
    s = np.array(y)
    return t, s





def execute_EMD_on_signal(t, s):
    IMF = EMD().emd(s,t)
    return IMF





def remove_IMF1(t, s):
    IMF = EMD().emd(s, t)
    y = np.array([0.0 for i in range(len(s))])
    for n, imf in enumerate(IMF):
        if n > 0:
            y += imf
    print(y.shape)
    return y





def draw_new_signal(t, s, ns):
    # Plot results
    plt.subplot(2, 1, 1)
    plt.plot(t, s, 'r')
    plt.title("Input signal:")
    plt.xlabel("Time [s]")
    
    
    plt.subplot(2, 1, 2)
    plt.plot(t, ns, 'g')
    plt.title("new signal:")
    plt.xlabel("Time [s]")
    
    plt.tight_layout()
    plt.show()





if __name__ == '__main__':
    t, s = define_signal([1, 2, 5, 12, 1, 9])
    IMF = execute_EMD_on_signal(t, s)
    