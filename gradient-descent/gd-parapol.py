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


def gradient_descent_with_fixed_inter(x_init, learning_rate, iteration):
    x_list = [x_init]
    for i in range(iteration):
        x_new = x_list[-1] - learning_rate*grad(x_list[-1])
        print(x_new)
        x_list.append(x_new)
    return x_list


def linear_regression_fomular():
    x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(y)
    print(x)
    # print(x0)
    y0 = x[0][0] * x0 ** 2 + x[1][0] * x0 + x[2][0]

    plt.plot(x0, y0, color="green")


def check_gradient(x):
    eps = 1e-4
    g = np.zeros_like(x)
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += eps
        x2[i] += - eps
        g[i] = (cost(x1) - cost(x2))/(2*eps)
    grad_x = grad(x)
    if np.linalg.norm(g-grad_x) > 1e-6:
        print(np.linalg.norm(g-grad_x))
        print('Gradient  caculation warning')


# random data
y = np.array([[2, 5, 7, 9, 11, 16, 19, 23, 22, 29, 29, 35,
               37, 40, 46, 42, 39, 31, 30, 28, 20, 15, 10, 6]]).T
X = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]]).T

# Create A square
x_square = np.array([X[:, 0]**2]).T
A = np.concatenate((x_square, X), axis=1)

# Create vector 1
ones = np.ones((A.shape[0], 1), dtype=np.int8)

# Combine 1 and A
A = np.concatenate((A, ones), axis=1)

fig1 = plt.figure("GD with parapol")
ax = plt.axes(xLim=(-10, 30), yLim=(-5, 50))

x0 = np.linspace(1.0, 25.0, 10000)

plt.plot(X, y, 'ro', color='red')
linear_regression_fomular()

# Gradient descent
m = A.shape[0]
# print(m)
x_init = np.array([[-2.1], [5.1], [-2.1]])
# Notes de co duoc ket qua chap nhan duoc can thu nhieu x_init : bai toan that
y_init = x_init[0][0] * x0 ** 2 + x_init[1][0] * x0 + x_init[2][0]
plt.plot(x0, y_init, color="black")


x_list = gradient_descent_with_fixed_inter(
    x_init, learning_rate=0.000001, iteration=70)
for i in range(len(x_list)):
    y0_gd = x_list[i][0][0] * x0 ** 2 + x_list[i][1][0] * x0 + x_list[i][2][0]
    plt.plot(x0, y0_gd, color='black', alpha=0.5)

# x_list, iter = gradient_descent(x_init, learning_rate=0.00001, tolerate=0.3)
# print('Iteration:' + str(iter))
# for i in range(iter):
#     y0_gd = x_list[i][0][0] * x0 ** 2 + x_list[i][1][0] * x0 + x_list[i][2][0]
#     plt.plot(x0, y0_gd, color='black')

ln, = ax.plot([], [], color='blue')


def update(frame):
    y0 = x_list[frame][0][0] * x0 * x0 + \
        x_list[frame][1][0] * x0 + x_list[frame][2][0]
    ln.set_data(x0, y0)
    return ln,


frames = np.arange(1, len(x_list), 1)
ani = animation.FuncAnimation(fig1, update, frames, interval=80, blit=True)

# Add Legend
plt.title('Gradient Descent Parapol Animation')
plt.legend(('Data',  'Solution by formular', 'Initial value for GD', 'Value in each GD iteration'
            ), loc=(0.01, 0.75))
ltext = plt.gca().get_legend().get_texts()

check_gradient(x_init)
plt.show()
