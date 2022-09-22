import numpy as np

def simulated_annealinng(objective, cooling_rate : float, n_iter : int=100):
    x_iter = np.zeros((n_iter, ))
    obj_iter = np.zeros((n_iter, ))

    # set initial temperature and cooling schedule
    T = 1.
    def cool(T): return cooling_rate * T

    curr_x = data_x[index]
    curr_obj = objective(curr_x)

    best_x = curr_x
    best_obj = curr_obj

    for i in range(n_iter):

        # decrease T according to cooling schedule
        T = cool(T)

        index = np.random.choice(n, 1)
        new_x = data_x[index]
        new_obj = objective(new_x)

        # update current solution iterate
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
