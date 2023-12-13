# Standard imports
import networkx as nx
import numpy as np
from tqdm import tqdm

class FGC:
    def __init__(self, L, X, n, gamma, lambd, alpha, seed = 666, quiet = False):
        # Save parameter fields.
        self.L = L
        self.X = X
        self.N = X.shape[0]
        self.n = n
        self.d = X.shape[1]
        
        # Initialize optimization variables as random matrices.
        np.random.seed(seed)
        self.X_tilde = np.random.normal(0, 1, (self.n, self.d))
        
        self.clip = 1e-10
        self.C = np.random.normal(0, 1, (self.N, self.n))
        self.C[self.C < self.clip] = self.clip

        # Save regularization variables.
        self.alpha = alpha
        self.lambd = lambd
        self.gamma = gamma
        
        # Optimization parameters.
        self.num_iter = 0
        self.lr = 1e-5
        self.num_c_iter = 100
        
        # Toggle progress bars.
        self.quiet = quiet

    def objective(self):
        f = 0
        
        # Bregman divergence.
        J = np.outer(np.ones(self.n), np.ones(self.n)) / self.n
        f += -self.gamma * np.linalg.slogdet(self.C.T @ self.L @ self.C + J)[1]
        
        # Dirichlet energy.
        L_tilde = self.C.T @ self.L @ self.C
        f += np.trace(self.X_tilde.T @ L_tilde @ self.X_tilde)

        # Regularization for C.
        f += self.lambd * np.linalg.norm(self.C @ np.ones((self.n, 1)))**2 / 2
        
        # Regularization for X_tilde.
        f += self.alpha * np.linalg.norm(self.X - self.C @ self.X_tilde)**2 / 2
        
        return f

    def update_X_tilde(self):
        L_tilde = self.C.T @ self.L @ self.C
        A = 2 * L_tilde / self.alpha + self.C.T @ self.C
        self.X_tilde = np.linalg.pinv(A) @ self.C.T @ X

        for i in range(len(self.X_tilde)):
            self.X_tilde[i] = self.X_tilde[i] / np.linalg.norm(self.X_tilde[i])
        
        return None

    def gradient_C(self):
        grad = np.zeros(self.C.shape)
        
        J = np.outer(np.ones(self.n), np.ones(self.n)) / self.n
        v = np.linalg.pinv(self.C.T @ self.L @ self.C + J)
        grad += -2*self.gamma * self.L @ self.C @ v
        
        grad += self.alpha * (self.C @ self.X_tilde - self.X) @ self.X_tilde.T
        grad += 2*self.L @ self.C @ self.X_tilde @ self.X_tilde.T
        grad += self.lambd * np.abs(self.C) @ (np.ones((self.n, self.n)))

        return grad

    def update_C(self):
        self.C = self.C - self.lr * self.gradient_C()
        self.C[self.C < self.clip] = self.clip

        for i in range(len(self.C)):
            self.C[i] = self.C[i] / np.linalg.norm(self.C[i],1)
        
        return None

    def fit(self, num_iters):
        loss = np.zeros(num_iters)
        for i in tqdm(range(num_iters), disable = self.quiet):
            for _ in range(self.num_c_iter):
                self.update_C()
            self.update_X_tilde()
            loss[i] = self.objective()
            self.num_iter += 1
        return (self.C, self.X_tilde, loss)

class GC:
    def __init__(self, L, n, gamma, lambd, alpha, seed = 666, quiet = False):
        # Save parameter fields.
        self.L = L
        self.N = L.shape[0]
        self.n = n
        
        # Initialize optimization variables as random matrices.
        self.clip = 1e-10
        self.C = np.random.normal(0, 1, (self.N, self.n))
        self.C[self.C < self.clip] = self.clip

        # Save regularization variables.
        self.alpha = alpha
        self.lambd = lambd
        self.gamma = gamma
        
        # Optimization parameters.
        self.num_iter = 0
        self.lr = 1e-5
        self.num_c_iter = 100
        
        # Toggle progress bars
        self.quiet = quiet

    def objective(self):
        f = 0
        
        # Bregman divergence.
        J = np.outer(np.ones(self.n), np.ones(self.n)) / self.n
        f += -self.gamma * np.linalg.slogdet(self.C.T @ self.L @ self.C + J)[1]

        # Regularization for C.
        f += self.lambd * np.linalg.norm(self.C @ np.ones((self.n, 1)))**2 / 2
        
        return f

    def gradient_C(self):
        grad = np.zeros(self.C.shape)
        
        J = np.outer(np.ones(self.n), np.ones(self.n)) / self.n
        v = np.linalg.pinv(self.C.T @ self.L @ self.C + J)
        grad += -2*self.gamma * self.L @ self.C @ v
        grad += self.lambd * np.abs(self.C) @ (np.ones((self.n, self.n)))

        return grad

    def update_C(self):
        self.C = self.C - self.lr * self.gradient_C()
        self.C[self.C < self.clip] = self.clip

        for i in range(len(self.C)):
            self.C[i] = self.C[i] / np.linalg.norm(self.C[i],1)
        
        return None

    def fit(self, num_iters):
        loss = np.zeros(num_iters)
        for i in tqdm(range(num_iters), disable = self.quiet):
            for _ in range(self.num_c_iter):
                self.update_C()
            loss[i] = self.objective()
            self.num_iter += 1
        return (self.C, loss)
    
def make_LX(G, d, seed = None):
    # Create edge weights
    np.random.seed(seed)
    
    N = len(G.nodes)
    W = np.zeros((N, N))
    for (x, y) in G.edges:
        W[x][y] = np.random.randint(1,10)
    W = W + W.T

    # Compute Laplacian
    D = np.diag(W @ np.ones((W.shape[0])))
    L = D - W

    # Create features for the synthetic graph
    X = np.random.multivariate_normal(np.zeros(N), np.linalg.pinv(L), d).T
    
    return L, X

def make_erdos_renyi_graph(N, p, d, seed = None):
    G = nx.erdos_renyi_graph(N, p, directed = False)
    return make_LX(G, d, seed)

def make_barabasi_albert_graph(N, m, d, seed = None):
    G = nx.barabasi_albert_graph(n = N, m = m, seed = seed)
    return make_LX(G, d, seed)

def make_watts_strogatz_graph(N, k, p, seed = None):
    G = nx.watts_strogatz_graph(N, k, p, seed = seed)
    return make_LX(G, d, seed)

def make_random_geometric_graph(N, r, d, seed = None):
    # Make graph
    G = nx.random_geometric_graph(N, r, seed = seed)
    return make_LX(G, d, seed)

def reconstruction_error(L, L_hat):
    return np.linalg.norm(L - L_hat)

N = 100
p = 0.1
d = 500

n = 10
num_iter = 100

fgc_err = np.zeros(100)
gc_err = np.zeros(100)
for i in range(100):
    # Make a new graph
    L, X = make_erdos_renyi_graph(N, p, d)
    
    # Compute reconstruction error with features
    fgc = FGC(L, X, n, 500, 500, d/2, quiet = True) 
    C, X_t, loss = fgc.fit(num_iter)
    P = np.linalg.pinv(C)
    CP = C @ P
    fgc_err[i] = reconstruction_error(L, CP.T @ L @ CP)
    print(round(fgc_err[i], 2), end = '\t')
    
    # Compute reconstruction error without features
    gc = GC(L, n, 500, 500, d/2, quiet = True) 
    C, loss = gc.fit(num_iter)
    P = np.linalg.pinv(C)
    CP = C @ P
    gc_err[i] = reconstruction_error(L, CP.T @ L @ CP)
    print(round(gc_err[i], 2))
    
    np.save('../results/er_fgc.npy', fgc_err)
    np.save('../results/er_gc.npy', gc_err)
    np.save('../results/er_diff.npy', fgc_err - gc_err)