import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Latex
    
"""This file contains some utility functions to handle results given
by the classes coded in the file phs_simulator.
"""
    
def plot_results(results, title, efforts_labels, flows_labels, state_labels):
    """Function to plot flow, efforts, energy and power as function
    of time using results given in results

    Args:
        results (dict): dictionnary containing all results
        title (string): plot title
        efforts_labels (list): labels for efforts variables
        flow_labels (list): labels for flow variables
        state_labels (list): labels for state variables
    """


    fig = plt.figure(figsize=(16,12))
    plt.suptitle(title, fontsize = 16)

    plt.subplot(2,3,1)
    plt.plot(results["Time"],results["Flows"], label = flows_labels)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitudes')
    plt.title('Variables flows')

    plt.subplot(2,3,2)
    plt.plot(results["Time"],results["Efforts"], label = efforts_labels)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitudes')
    plt.title('Efforts')

    plt.subplot(2,3,3)
    plt.plot(results["Time"],results["State"], label=state_labels)
    plt.title('State variables')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitudes')
    plt.legend()


    ax = plt.subplot(2,3,4)
    plt.plot(results["Time"],results["Estored"], label = "Energy")
    plt.title("Energy (value of the hamiltonian)")
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.ylim((np.min(results["Estored"]), np.max(results["Estored"])))
    ax.ticklabel_format(useOffset=False)
    plt.legend()

    plt.subplot(2,3,5)
    plt.plot(results["Time"],results["Pdiss"], label='Power of dissipated energy')
    plt.plot(results["Time"],results["Pstored"], label='Power of stored energy')
    plt.plot(results["Time"],results["Pext"], label='Power of exterior energy')
    plt.title('Powers')
    plt.xlabel('Time (s)')
    plt.ylabel('Power (W)')
    plt.legend()

    plt.subplot(2,3,6)
    Perror = results["Ptot"]
    plt.plot(results["Time"],Perror)
    print(f"Mean error on power balance = {np.mean(Perror)}")
    plt.title("Error on power balance")
    plt.xlabel('Time (s)')
    plt.ylabel('Error (W)')

    plt.tight_layout()
    return fig

def plot_power_error(results):
    """Function to plot error on the power balance as a function of time

    Args:
        results (dict): dictionnary containing all results
    """
    fig = plt.figure()
    plt.title("Error on power balance",fontsize = 16)
    Perror = np.sqrt(results["Ptot"]**2)
    plt.plot(results["Time"],Perror)
    plt.xlabel("Time (s)")
    plt.ylabel("Error")

    return fig

def display_quadratization_init(phsquad, positive = True):
    """Displays change of state Q, jacobian matrix Jqx and matrices Jx(x), Rx(x) and Qx(x) for
    the given instance of the PHSQuadratiser class. Can be used directly after instanciation
    of the class. 
    ! Only works in jupyter notebooks as it uses Ipython.display to show better 
    looking expressions ! 

    Args:
        phsquad (object): instance of the PHSQuadratiser class
        positive (bool): if true, expressions and matrices for positive values of the state are
            displayed, otherwise expressions and matrices for negative values of the state are
            displayed. Default to True
    """
    display(Latex('Expression of the change of state Q for each state variables for positive $x$: '))
    display(*phsquad.QPos)

    display(Latex('Expression of the jacobian matrix $J_{qx}(x)$ for positive $x$: '))
    display(phsquad.JqxPos)

    display(Latex('Expression of the matrix $J_{x}(x)$ for positive $x$: '))
    display(phsquad.JxPos)

    display(Latex('Expression of the matrix $R_{x}(x)$ for positive $x$: '))
    display(phsquad.RxPos)

    display(Latex('Expression of the matrix $M_{x}(x)$ for positive $x$: '))
    display(phsquad.MxPos)

def display_quadratization_JRMq(phsquad, positive=True):
    """Displays matrices Jq(q), Rq(q) and Mq(q) for the given instance of the PHSQuadratiser 
    class. Note that these attributes only exist after the use of the computeJRMq function of 
    the class. 
    ! Only works in jupyter notebooks as it uses Ipython.display to show better 
    looking expressions ! 

    Args:
        phsquad (object): instance of the PHSQuadratiser class
        positive (bool): if true, matrices for positive values of the state are
            displayed, otherwise  matrices for negative values of the state are
            displayed. Default to True
    """
    display(Latex('Expression of the matrix $J_{q}(q)$ for positive $x$: '))
    display(phsquad.JqPos)

    display(Latex('Expression of the matrix $R_{q}(q)$ for positive $x$: '))
    display(phsquad.RqPos)

    display(Latex('Expression of the matrix $M_{q}(q)$ for positive $x$: '))
    display(phsquad.MqPos)