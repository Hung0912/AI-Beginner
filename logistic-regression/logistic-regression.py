import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import LogisticRegression


def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))


def net_input(theta, x):
    # Computes the weighted sum of inputs
    return np.dot(x, theta)


def probability(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(net_input(theta, x))


def cost(theta, x, y):
    cos = -1/m*np.sum(y*np.log(probability(theta, x)) +
                      (1-y)*np.log(1-probability(theta, x)))
    return cos


def grad(theta, x, y):
    return 1/m * (np.dot(x.T, probability(theta, x) - y))


def grad_test(theta, x, y):
    eps = 1e-4
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        t1 = theta.copy()
        t2 = theta.copy()
        t1[i] += eps
        t2[i] += - eps
        g[i] = (cost(t1, x, y) - cost(t2, x, y))/(2*eps)
    grad_x = grad(theta, x, y)
    if np.linalg.norm(g-grad_x) > 1e-7:
        print(np.linalg.norm(g-grad_x))
        print('Gradient  caculation warning')


def logistic_regression(theta, x, y, learning_rate, tolerance):
    interation = 0
    theta_list = [theta]
    while True:
        interation += 1
        theta_new = theta_list[-1] - learning_rate*grad(theta_list[-1], x, y)
        theta_list.append(theta_new)
        # When to stop
        if np.linalg.norm(grad(theta_list[-1], x, y)) <= tolerance:
            # print(np.linalg.norm(grad(theta_list[-1], x, y)))
            break

    return theta_list, interation


fig1 = plt.figure("GD with logistic regression")
ax = plt.axes(xLim=(-3, 40), yLim=(-2, 2))

X0 = np.array([[2], [4], [5], [6], [10], [11], [12], [20], [30], [36]])
y0 = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]).T
m = X0.shape[0]
plt.plot(X0, y0, 'ro')

# dung sklearn
lr = LogisticRegression().fit(X0, y0)
coeffiecient = lr.coef_[0][0]
intercept = lr.intercept_[0]
print(coeffiecient, intercept)
x = np.linspace(0, 46, 1000)
y0_sklearn = sigmoid(x * coeffiecient + intercept)

plt.plot(x, y0_sklearn, color="green")
ones = np.ones_like(X0)
X0 = np.concatenate((ones, X0), axis=1)


# khoi tao theta
theta_init = np.array([[-1], [.7]])
y_init = sigmoid(theta_init[0][0] + theta_init[1][0] * x)
plt.plot(x, y_init, color="blue")

theta_list, interation = logistic_regression(
    theta_init, X0, y0, learning_rate=0.01, tolerance=0.3)
print(interation)


for i in range(interation):
    y_theta = sigmoid(theta_list[i][0] + theta_list[i][1] * x)
    plt.plot(x, y_theta, color='black', alpha=0.3)


# animation
ln, = ax.plot([], [], color='blue')


def update(frame):
    y0 = sigmoid(theta_list[frame][0][0] + theta_list[frame][1][0] * x)
    ln.set_data(x, y0)
    return ln,


frames = np.arange(interation)
ani = animation.FuncAnimation(fig1, update, frames, interval=50, blit=True)

# Add Legend
plt.title('Gradient Descent for Logistic Regression Animation')
plt.legend(('Data',  'Solution by formular', 'Initial value for GD', 'Value in each GD iteration'
            ), loc=(0.52, 0.01))
ltext = plt.gca().get_legend().get_texts()
# predict
x0 = 18
y0 = sigmoid(theta_list[-1][0] + theta_list[-1][1] * x0)
print(y0)
plt.show()
