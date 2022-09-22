import numpy as np
from utils.sampling import sample_binary_matrix

def simulated_annealinng(objective, n_vars, cooling_rate : float = 0.985, n_iter : int=100):
    x_iter = np.zeros((n_iter, ))
    obj_iter = np.zeros((n_iter, ))

    # set initial temperature and cooling schedule
    T = 1.
    def cool(T): return cooling_rate * T

    curr_x = sample_binary_matrix(1, n_vars)
    curr_obj = objective(curr_x)

    best_x = curr_x
    best_obj = curr_obj

    for i in range(n_iter):

        # decrease T according to cooling schedule
        T = cool(T)

        new_x = sample_binary_matrix(1, n_vars)
        new_obj = objective(new_x)

        # update current solution
        if (new_obj > curr_obj) or (np.random.rand()
                                    < np.exp((new_obj - curr_obj) / T)):
            curr_x = new_x
            curr_obj = new_obj

        # Update best solution
        if new_obj > best_obj:
            best_x = new_x
            best_obj = new_obj

        # save solution
        x_iter[i] = best_x
        obj_iter[i] = best_obj

    return x_iter, obj_iter
