import numpy as np
from scipy.special import expit
from gurobipy import *
import random


def lassothres(x: np.array, lam: float) -> np.array:
    """
    computes the lasso thresholding function
    :param x: input parameter matrix
    :param lam: tuning parameter
    :return: lasso penalized input
    """
    dim1, dim2 = np.shape(x)
    for i in range(dim1):
        for j in range(dim2):
            x[i, j] = np.sign(x[i, j]) * np.max([0, np.abs(x[i, j]) - lam])
    return x


def groupsum(x: np.array) -> float:
    """
    computes the group sum of a matrix x (sum of l2-norm of each column)
    :param x: the parameter matrix
    :return: groupsum, float
    """
    dim2 = np.shape(x)[1]
    grpsum = 0
    for i in range(dim2):
        grpsum = grpsum + np.linalg.norm(x[:, i])
    return grpsum


def grouplassothres(x: np.array, lam: float) -> np.array:
    """
    computes the group thresholding function
    :param x: the input parameter matrix
    :param lam: tuning parameter
    :return: group lasso penalized input
    """
    dim2 = np.shape(x)[1]
    for j in range(dim2):
        x[:, j] = np.max([0, 1 - lam / np.linalg.norm(x[:, j])]) * x[:, j]
    return x


def generate_data(
        seed, n: int = 50, p: int = 100, cor: float = 0.6, ntotal: int = 100) -> [
        np.array, np.array, np.array, np.array]:
    """
    generate training and testing data
    :param seed: seed
    :param n: training sample size
    :param p: number of features
    :param cor: feature correlation
    :param ntotal: total sample size
    :return: x_train, x_test, y_train, y_test
    """
    random.seed(seed)
    # initial design matrix, parameters and response. Design matrix are drawn
    # independently from uniform(-1,1) distribution. Random errors are drawn from
    # normal(0,1) distribution.
    x = np.random.uniform(-1, 1, (ntotal, p))
    xu = np.random.uniform(-1, 1, ntotal)
    xv = np.random.uniform(-1, 1, ntotal)
    t = np.sqrt(cor / (1 - cor))
    for j in range(4):
        x[:, j] = (x[:, j] + t * xu) / (1 + t)  # generate correlated features
    for j in range(4, p):
        x[:, j] = (x[:, j] + t * xv) / (1 + t)  # generate correlated features
    truefx = 6 * x[:, 0] + np.sqrt(84) * x[:, 1] ** 3 + np.sqrt(12 / (np.sin(6) / 12 + 1 / 2)) * np.sin(
        3 * x[:, 2]) + np.sqrt(48 / (np.exp(2) + np.exp(-1) - np.exp(-2) - np.exp(1))) * np.exp(x[:, 3])
    prob = expit(truefx)  # compute probability
    y = np.random.binomial(1, p=prob)  # generate labels from Binomial distribution
    xtotal = x[:, :]
    x_train = xtotal[:n, :]
    x_test = xtotal[n:, :]
    ytotal = y[:]
    y_train = ytotal[:n]
    y_test = ytotal[n:]
    return x_train, x_test, y_train, y_test


class nnSglL0:
    """
    implement the sparse group lasso l0 l1 neural network
    """

    def __init__(self,
                 x: np.array,
                 y: np.array,
                 lam1: float,
                 lam2: float,
                 alpha: float,
                 k: int,
                 n_nodes: int = 50,
                 tau: float = 0.1,
                 shrink_factor: float = 0.8,
                 stepsize: float = 0.1,
                 l: float = 1,
                 m: float = 100):
        """
        initialization
        :param x: training data
        :param y: labels
        :param lam1: tuning parameter for sparse group lasso
        :param lam2: tuning parameter for l1
        :param alpha: balancing parameter in sparse group lasso
        :param k: upper bound for l0
        :param n_nodes: number of hidden nodes
        :param tau: tolerance
        :param shrink_factor: between 0 and 1, in gradient descent step
        :param stepsize: step size in gradient descent step
        :param l: parameter in convex relaxation
        :param m: upper bound for beta
        """
        self.x = x
        self.y = y
        self.n, self.p = self.x.shape
        self.lam1 = lam1
        self.lam2 = lam2
        self.alpha = alpha
        self.K = k
        self.n_nodes = n_nodes
        self.tau = tau
        self.shrikn_factor = shrink_factor
        self.stepsize = stepsize
        self.L = l
        self.M = m
        self.theta = None
        self.t = None
        self.beta = None
        self.b = None
        self.VEC1P = np.ones(self.p)
        self.VEC1N = np.ones(self.n)
        self.VEC1NNODES = np.ones(self.n_nodes)

    def initialize_par(self) -> [np.array, np.array, np.array, float]:
        """
        initialize parameters
        :return: initial values for theta, beta, t and b
        """
        thetaold = np.random.uniform(-1, 1, [self.n_nodes, self.p])
        betaold = np.random.uniform(-10, 10, self.n_nodes)
        told = np.random.uniform(-1, 1, self.n_nodes)
        bold = 1
        print('Parameters initialized')
        return thetaold, betaold, told, bold

    def smooth_gradient(
            self, thetaold: np.array, betaold: np.array, told: np.array, bold: float,
            x: np.array = None, y: np.array = None, stepsize: float = None) -> [np.array, np.array, float]:
        """
        solve the smooth part and sparse group lasso part
        :param thetaold: theta
        :param betaold: beta
        :param told: t
        :param bold: b
        :param x: data matrix
        :param y: label
        :param stepsize: step size in gradient descent
        :return: updated theta, t and b
        """
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if stepsize is None:
            stepsize = self.stepsize
        errtbtheta = self.tau * 100
        while errtbtheta > self.tau:
            linesearch = 0
            while linesearch == 0:
                # compute values needed to compute the gradients
                xiold = np.matmul(thetaold, np.transpose(x))
                for i in range(self.n_nodes):
                    xiold[i] = xiold[i] + told[i]
                etaold = np.matmul(np.transpose(expit(xiold)), betaold) + bold
                muold = expit(etaold)
                # compute gradient for t, b and theta
                dert = -np.matmul(np.diag(betaold), np.matmul(expit(xiold) * (1 - expit(xiold)), y - muold)) / self.n
                derb = -np.matmul(y - muold, self.VEC1N) / self.n
                dertheta = np.zeros([self.n_nodes, self.p])
                for i in range(self.n_nodes):
                    dertheta[i] = -betaold[i] * np.matmul(np.transpose(x), np.matmul(
                        np.diag(y - muold), expit(xiold[i]) * (1 - expit(xiold[i])))) / self.n
                # update t, b and theta using gradient descent
                tnew = told - stepsize * dert
                bnew = bold - stepsize * derb
                thetainter1 = thetaold - stepsize * dertheta
                # use threshold function to update theta lasso penalty
                thetainter2 = lassothres(thetainter1, stepsize * self.lam1 * (1 - self.alpha))
                # use threshold function to update theta group lasso penalty
                thetanew = grouplassothres(thetainter2, stepsize * self.lam2 * self.alpha)
                # calculate the target function and check the line search criterion
                targetold = -np.matmul(y * etaold - np.log(1 + np.exp(etaold)), self.VEC1N) / self.n + self.lam1 * (
                        1 - self.alpha) * np.sum(np.absolute(thetaold)) + self.lam1 * self.alpha * groupsum(thetaold)
                xinew = np.matmul(thetanew, np.transpose(x))
                for i in range(self.n_nodes):
                    xinew[i] = xinew[i] + tnew[i]
                etanew = np.matmul(np.transpose(expit(xinew)), betaold) + bnew
                targetnew = -np.matmul(y * etanew - np.log(1 + np.exp(etanew)), self.VEC1N) / self.n + self.lam1 * (
                        1 - self.alpha) * np.sum(np.absolute(thetanew)) + self.lam1 * self.alpha * groupsum(thetanew)
                parametersold = np.append(np.matrix.flatten(thetaold), told)
                parametersold = np.append(parametersold, bold)
                parametersnew = np.append(np.matrix.flatten(thetanew), tnew)
                parametersnew = np.append(parametersnew, bnew)
                if targetnew <= targetold - stepsize * 0.5 * np.linalg.norm(parametersold - parametersnew) ** 2:
                    linesearch = 1
                else:
                    stepsize = self.shrikn_factor* stepsize
            thetadiff = thetanew - thetaold
            tdiff = tnew - told
            bdiff = bnew - bold
            thetaold = thetanew
            told = tnew
            bold = bnew
            errtbtheta = np.sqrt(np.sum(thetadiff ** 2) + np.sum(tdiff ** 2) + bdiff ** 2)
        return thetanew, tnew, bnew

    def gurobi_beta(
            self, betaold: np.array, xi: np.array, bnew: float, y: np.array = None) -> np.array:
        """
        use gurobi to solve the mixed integer second order cone problem for beta
        :param betaold: beta from previous step
        :param xi: the output of hidden layer
        :param bnew: b from previous step
        :param y: labels
        :return: new beta
        """
        if y is None:
            y = self.y
        errbeta = self.tau * 1000
        while errbeta > self.tau:
            # gurobi model
            model = Model("nnbeta")
            model.setParam('OutputFlag', False)
            model.setParam('TimeLimit', 10)
            # add variables
            u = model.addVar(vtype=GRB.CONTINUOUS, name="u")
            v = model.addVar(vtype=GRB.CONTINUOUS, name="v")
            beta = model.addVars(self.n_nodes, lb=-self.M, ub=self.M, vtype=GRB.CONTINUOUS, name="beta")
            z = model.addVars(self.n_nodes, vtype=GRB.BINARY, name="z")
            betabar = model.addVars(self.n_nodes, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="betabar")
            model.update()
            betav = model.getVars()[2:(self.n_nodes + 2)]
            zv = model.getVars()[(self.n_nodes + 2):(self.n_nodes + self.n_nodes + 2)]
            betabarv = model.getVars()[(self.n_nodes * 2 + 2):(self.n_nodes * 3 + 2)]
            model.update()
            # add constraints
            etabetaold = np.matmul(np.transpose(expit(xi)), betaold) + bnew
            mubetaold = expit(etabetaold)
            alpha = betaold + np.matmul(expit(xi), y - mubetaold) / self.L / self.n
            rss = QuadExpr(np.matmul(alpha, alpha) * self.L / 2)
            rss.addTerms(-self.L * alpha, betav)
            rss.addTerms(self.VEC1NNODES * self.L / 2, betav, betav)
            zsum = LinExpr(self.VEC1NNODES, zv)
            betabarsum = LinExpr(self.VEC1NNODES, betabarv)
            model.addConstr(rss, GRB.LESS_EQUAL, u, name="rss")
            model.addConstrs((beta[i] <= self.M * z[i] for i in range(self.n_nodes)), name="betamax")
            model.addConstrs((beta[i] >= -self.M * z[i] for i in range(self.n_nodes)), name="betamin")
            model.addConstr(zsum, GRB.EQUAL, self.K, name="nnzero")
            model.addConstrs((beta[i] <= betabar[i] for i in range(self.n_nodes)), name="betabarmax")
            model.addConstrs((beta[i] >= -betabar[i] for i in range(self.n_nodes)), name="betabarmin")
            model.addConstr(betabarsum, GRB.LESS_EQUAL, v, name="betanorm")
            model.update()
            model.setObjective(u + self.lam2 * v)
            print('before optimize')
            model.optimize()
            print('after optimize')
            betanew = model.getAttr("X", model.getVars()[2:(self.n_nodes + 2)])
            betadiff = np.array(betanew) - np.array(betaold)
            errbeta = np.linalg.norm(betadiff)
            print('betaold: ', betaold)
            betaold = betanew
            print('betanew: ', betanew)
            print('errbeta: ', errbeta)
        return betanew

    def l1l0nn(self, seed: float = 1):
        """
        run the full optimization algorithm
        :param seed: random seed
        :return: None
        """
        random.seed(seed)
        thetaold, betaold, told, bold = self.initialize_par()
        errtotal = self.tau * 100
        targetfullold = 0
        while errtotal > self.tau:
            thetanew, tnew, bnew = self.smooth_gradient(
                thetaold=thetaold,
                betaold=betaold,
                told=told,
                bold=bold)
            xi = np.matmul(thetanew, np.transpose(self.x))
            for i in range(self.n_nodes):
                xi[i] += tnew[i]
            betanew = self.gurobi_beta(
                betaold=betaold,
                xi=xi,
                bnew=bnew
            )
            eta = np.matmul(np.transpose(expit(xi)), betanew) + bnew
            targetfullnew = -np.matmul(self.y * eta - np.log(1 + np.exp(eta)), self.VEC1N) / self.n + self.lam1 * (
                    1 - self.alpha) * np.sum(np.absolute(thetanew)) + self.lam1 * self.alpha * groupsum(
                thetanew) + self.lam2 * np.linalg.norm(betanew, 1)
            errtotal = abs(targetfullnew - targetfullold)
            targetfullold = targetfullnew
            print('errtotal :', errtotal)
        self.theta = thetanew
        self.t = tnew
        self.beta = betanew
        self.b = bnew

    def predict(self, x: np.array, y: np.array = None) -> dict:
        """
        predict and evaluate
        :param x: testing data
        :param y: testing label
        :return: the prediction (and the accuracy if test label is provided)
        """
        xipred = np.matmul(self.theta, np.transpose(x))
        for i in range(len(xipred)):
            xipred[i] += self.t[i]
        etapred = np.matmul(np.transpose(expit(xipred)), self.beta) + self.b
        mupred = expit(etapred)
        print(mupred)
        ypred = np.ones(x.shape[0])
        for i in range(len(ypred)):
            if mupred[i] < 0.5:
                ypred[i] = 0
        if y is not None:
            accuracy = 1 - np.mean(np.abs(y.ravel() - ypred.ravel()))
            return {'ypred': ypred, 'accuracy': accuracy}
        else:
            return {'ypred': ypred}
