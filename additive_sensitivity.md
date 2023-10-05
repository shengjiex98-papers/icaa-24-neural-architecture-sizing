
## Background
We assume the system is $x \in R^n$ and $x[t+1] = Ax[t] + Bu[t]$ where $u[t]=K\hat{x}[t]$ is computed from the neural-network percepted states $\hat{x}[t] \approx x[t]$.

## Multiplicative noise model
We model noise by $\hat{x}[t] = Ex[t]$ where $E$ is a diagonal matrix that is close to the identity matrix. Specifically, given a small $\epsilon_i$ such that $\hat{x}_i[t] = (1+\epsilon_i)x_i[t]$,
<!-- given $\lambda_i \approx 1$ such that $\hat{x}_i[t] \in (\frac{x_i[t]}{\lambda_i}, x_i[t]\lambda_i)$ -->
$$
E = \begin{bmatrix}
    1+\epsilon_1 & & \huge0 \\
    & \ddots &  \\
    \huge0 & & 1+\epsilon_n
\end{bmatrix},
$$
and the system model becomes
$$
x[t+1] = Ax[t] + BK(Ex[t]) \\
       = (A+BKE)x[t].
$$

We use a special notation $E_i$ to denote when only **one** state, $x_i$, is affected by uncertainty and all other states are assumed to be perfectly sensed.
$$
E_i = \begin{bmatrix}
    I_{i-1} & & \huge0 \\
    & 1+\epsilon_i &  \\
    \huge0 & & I_{n-i}
\end{bmatrix}.
$$

For example, when $A=
\begin{bmatrix}
a_{11} & a_{11} \\
a_{21} & a_{22}
\end{bmatrix}
$, $B=\begin{bmatrix}
    b_1 \\ b_2
\end{bmatrix}$, and $K = \begin{bmatrix}
    k_1 &  k_2
\end{bmatrix}$,
$$
A+BKE_1 =
\begin{bmatrix}
a_{11}+b_1k_1 (1+\epsilon_1) & a_{11}+b_1k_2 \\
a_{21}+b_2k_1 (1+\epsilon_1) & a_{22}+b_2k_2
\end{bmatrix},
$$
and
$$
A+BKE_2 =
\begin{bmatrix}
a_{11}+b_1k_1 & a_{11}+b_1k_2 (1+\epsilon_2) \\
a_{21}+b_2k_1 & a_{22}+b_2k_2 (1+\epsilon_2)
\end{bmatrix}.
$$

### Computing sensitivity with multiplicative noise model
To use the technique that computes the sensitivity of maximum singular value to an error $\epsilon$ between $X$ and $X + \epsilon Y + O(\epsilon^2)$, we set $X = A+BK$ and $Y_i = \begin{bmatrix}
    & b_1k_i & \\
    \huge 0 & \vdots & \huge 0 \\
    & b_nk_i & \\
\end{bmatrix}$. 

For example, when $A=
\begin{bmatrix}
a_{11} & a_{11} \\
a_{21} & a_{22}
\end{bmatrix}
$, $B=\begin{bmatrix}
    b_1 \\ b_2
\end{bmatrix}$, and $K = \begin{bmatrix}
    k_1 &  k_2
\end{bmatrix}$,
$$
X = A+BK =
\begin{bmatrix}
a_{11}+b_1k_1 & a_{11}+b_1k_2 \\
a_{21}+b_2k_1 & a_{22}+b_2k_2
\end{bmatrix} \\
Y_1 =
\begin{bmatrix}
b_1k_1 & 0 \\
b_2k_1 & 0
\end{bmatrix} \\
Y_2 =
\begin{bmatrix}
0 & b_1k_2 \\
0 & b_2k_2
\end{bmatrix}.
$$

## Additive noise model
In the additive noise model, the noise $w$ is not dependent on the current state $x[t]$. *i.e.*, $\hat{x}_i[t] = x_i[t] + w_i$.
$$
\begin{align*}
x[t+1] &= Ax[t] + BK(x[t]+w) \\
    &= (A+BK)x[t] + BKw
\end{align*}
$$

Since
$
BK =
\begin{bmatrix} b_1k_1 & b_1k_2\\ 
b_2k_1 & b_2k_2 \end{bmatrix},
$
The two parts of the expressions are
$$
(A+BK)x=
\begin{bmatrix} (a_{11}+b_1k_1)x_1+(a_{12}+b_1k_2)x_2 \\ 
(a_{21}+b_2k_1)x_1+(a_{22}+b_2k_2)x_2 \end{bmatrix},
$$
and
$$
BKw =
\begin{bmatrix} b_1k_1w_1+b_1k_2w_2 \\
 b_2k_1w_1+b_2k_2w_2 \end{bmatrix}
$$

Let $w = \begin{bmatrix}
    1 \\ 0
\end{bmatrix}$, then
$$
BKw =
\begin{bmatrix} b_1k_1 \\
 b_2k_1 \end{bmatrix}
$$

$$
(A+BK)x+BKw=
\begin{bmatrix} (a_{11}+b_1k_1)x_1+(a_{12}+b_1k_2)x_2+b_1k_1w_1+b_1k_2w_2 \\
 (a_{21}+b_2k_1)x_1+(a_{22}+b_2k_2)x_2+b_2k_1w_1+b_2k_2w_2 \end{bmatrix}
$$

### Computing sensitivity with additive noise model

Our method requires the perturbation to be reflected in the augmented matrix. One way to achieve this is to add a "fake" state in $x$ and modify $A$, $B$ and $K$ matrices accordingly:

$$
\bar{x} = \begin{bmatrix} x_1 \\ x_2 \\ 1 \end{bmatrix} \\
\bar{A} = \begin{bmatrix}
    a_{11} & a_{12} & 0 \\
    a_{21} & a_{22} & 0 \\
    0 & 0 & 1 
\end{bmatrix} \\
\bar{B}\bar{K} = \begin{bmatrix}
    b_1k_1 & b_1k_2 & 0 \\
    b_2k_1 & b_2k_2 & 0 \\
    0 &    0 &    0 
\end{bmatrix}
$$

Recall that $x[t+1] = (A+BK)x[t] + BKw$
$$
w' = \begin{bmatrix}
    0 & 0 & b_1k_1w_1+b_1k_2w_2 \\
    0 & 0 & b_2k_1w_1+b_2k_2w_2 \\
    0 & 0 & 0
\end{bmatrix} \\
w'1 = \begin{bmatrix}
    0 & 0 & b_1k_1 \\
    0 & 0 & b_2k_1 \\
    0 & 0 & 0
\end{bmatrix} \\
x[t+1] = (A+BK+w)x[t] = \begin{bmatrix}
    (a_{11}+b_1k_1)x_1+(a_{12}+b_1k_2)x_2+b_1k_1w_1+b_1k_2w_2 \\
    (a_{21}+b_2k_1)x_1+(a_{22}+b_2k_2)x_2+b_2k_1w_1+b_2k_2w_2 \\
    1
\end{bmatrix}
   
$$

X = A+BK
Y = w'i

B <- nxm
K <- mxn

x <- R^n
w <- R^n, only 1 element in w is non-zero

Y =
[0   0   1
 0   0   1]

$$
\begin{bmatrix} a_{11}+b_1k_1 & a_{12}+b_1k_2+l\\ 
a_{21}+b_2k_1 & a_{22}+b_2k_2+l \end{bmatrix}
$$