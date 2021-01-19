import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib.animation as animation


def cost(x):
    return 1/(2*m)*np.linalg.norm(A.dot(x) - y, 2)**2


def grad(x):
    return 1/m * (A.T).dot(A.dot(x) - y)


def gradient_descent(x_init, learning_rate, tolerate):
    interation = 0
    x_list = [x_init]
    while True:
        interation += 1
        x_new = x_list[-1] - learning_rate*grad(x_list[-1])
        x_list.append(x_new)
        # When to stop
        if np.linalg.norm(grad(x_list[-1]))/m <= tolerate:
            print(np.linalg.norm(grad(x_list[-1]))/m)
            break

    return x_list, interation


def gra_test(x):
    eps = 1e-4
    g = np.zeros_like(x)
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += eps
        x2[i] += - eps
        g[i] = (cost(x1) - cost(x2))/(2*eps)
    grad_x = grad(x)
    if np.linalg.norm(g-grad_x) > 1e-7:
        print(np.linalg.norm(g-grad_x))
        print('Gradient  caculation warning')


def gradient_descent_with_fixed_inter(x_init, learning_rate, iteration):
    x_list = [x_init]
    for i in range(iteration):
        x_new = x_list[-1] - learning_rate*grad(x_list[-1])
        x_list.append(x_new)
    return x_list


def show_cost():
    # show f(x) va x
    f_list = []
    for i in range(iter):
        f_list.append(cost(x_list[i]))
        # print(f_list)
    plt.figure("f(x)")
    plt.xlabel("Iteration")
    plt.ylabel("Cost value")
    plt.plot(np.arange(iter), f_list, color="blue")
    plt.show()


# Random data
A = np.array([[2, 5, 7, 9, 11, 16, 19, 23, 22, 29, 29, 35, 37, 40, 46]]).T
y = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]).T

# Linear regression
lr = linear_model.LinearRegression()
lr.fit(A, y)

coeffiecient = lr.coef_[0][0]  # he so a
intercept = lr.intercept_[0]  # b
# print(coeffiecient, intercept)
x0_gd = np.linspace(0, 46, 2)
y0_sklearn = x0_gd * coeffiecient + intercept

fig1 = plt.figure("GD for Linear Regression")
ax = plt.axes(xLim=(-10, 60), yLim=(-1, 20))

plt.plot(A, y, 'ro')
plt.plot(x0_gd, y0_sklearn, color="green")

# Gradiant descent\
# khoi tao gia tri dau random
x_init = np.array([[1.], [2.]])
y_init = x_init[0][0] * x0_gd + x_init[1][0]
plt.plot(x0_gd, y_init, color="black")

m = A.shape[0]
# print(m)

ones = np.ones_like(A)
A = np.concatenate((A, ones), axis=1)
# print(A)

gra_test(x_init)

x_list, iter = gradient_descent(x_init, learning_rate=0.0001, tolerate=0.3)

for i in range(iter):
    y0_xlist = x_list[i][0] * x0_gd + x_list[i][1]
    plt.plot(x0_gd, y0_xlist, color="black", alpha=0.3)

# Draw animation gd
ln, = ax.plot([], [], color='blue')


def update(frame):
    y0 = x_list[frame][0][0] * x0_gd + x_list[frame][1][0]
    ln.set_data(x0_gd, y0)
    return ln,


iteration = np.arange(iter)
# print(iteration)
ani = animation.FuncAnimation(
    fig1, update, iteration, interval=50, blit=True)

# Add Legend
plt.title('Gradient Descent Animation')
plt.legend(('Data',  'Solution by formular', 'Initial value for GD', 'Value in each GD iteration'
            ), loc=(0.52, 0.01))
ltext = plt.gca().get_legend().get_texts()

plt.show()
