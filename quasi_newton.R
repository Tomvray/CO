################################################################################
# Limited-memory BFGS minimiser with strong-Wolfe line-search
#
#  fn  : function(x)          – returns objective value  f(x)
#  gr  : function(x)          – returns gradient vector ∇f(x)
#  x0  : numeric vector       – starting point
#  m   : memory size (pairs)  – default 10 (typical: 3–20)
#  c1, c2 : Wolfe constants   – def 1e-4, 0.9  (strong Wolfe: |g·p| <= −c2 g0·p)
#  tol : ‖∇f‖∞ stopping tol   – def 1e-6
#  maxit : max iterations     – def 1000
#
# Returns a list with: x, f, g, niter, nfun, ngr, converged, trace
################################################################################
lbfgs <- function(fn, gr, x0,
                  m = 10, c1 = 1e-4, c2 = 0.9,
                  tol = 1e-6, maxit = 1000, max_ls = 25,
                  verbose = FALSE) {

  ## ---- helpers -------------------------------------------------------------
  dot <- function(a, b) sum(a * b)

  strong_wolfe <- function(f0, g0dotp, xk, pk, alpha0 = 1) {
    # Backtracking variant that guarantees Armijo + strong-curvature
    alpha <- alpha0
    f_prev <- f0
    for (i in 1:max_ls) {
      x_new <- xk + alpha * pk
      f_new <- fn(x_new)
      g_new <- gr(x_new)
      if (f_new > f0 + c1 * alpha * g0dotp || (i > 1 && f_new >= f_prev)) {
        alpha <- alpha * 0.5        # fails Armijo: shrink
      } else {
        gdotp <- dot(g_new, pk)
        if (abs(gdotp) <= -c2 * g0dotp) {
          return(list(alpha = alpha, f = f_new, g = g_new, nfev = i, ngev = i))
        }
        if (gdotp >= 0) {
          alpha <- alpha * 0.5      # slope has wrong sign: shrink
        } else {
          f_prev <- f_new           # keep searching forward
          alpha <- alpha * 1.1      # modest expansion
        }
      }
    }
    warning("Line search failed to satisfy Wolfe after ", max_ls, " trials")
    list(alpha = alpha, f = f_new, g = g_new, nfev = max_ls, ngev = max_ls)
  }

  two_loop_recursion <- function(q, s_list, y_list, rho_list) {
    # Computes H_k * q  where H_k is L-BFGS inverse-Hessian approx.
    k <- length(rho_list)
    alpha <- numeric(k)
    if (k) {
      for (i in k:1) {
        alpha[i] <- rho_list[i] * dot(s_list[[i]], q)
        q <- q - alpha[i] * y_list[[i]]
      }
    }
    # initial Hessian scaling (used here: γ_k = (s_{k-1}·y_{k-1})/(y_{k-1}·y_{k-1}))
    if (k) {
      gamma <- dot(s_list[[k]], y_list[[k]]) / dot(y_list[[k]], y_list[[k]])
      H0q <- gamma * q
    } else {
      H0q <- q                     # start with Identity
    }
    r <- H0q
    if (k) {
      for (i in 1:k) {
        beta <- rho_list[i] * dot(y_list[[i]], r)
        r <- r + s_list[[i]] * (alpha[i] - beta)
      }
    }
    r
  }

  ## ---- initialisation ------------------------------------------------------
  xk <- x0
  fk <- fn(xk)
  gk <- gr(xk)
  nfun <- 1
  ngr  <- 1
  n    <- length(xk)
  s_list <- list()
  y_list <- list()
  rho_list <- numeric()
  trace <- data.frame(iter = 0, f = fk, grad_inf = max(abs(gk)))

  if (verbose)
    cat(sprintf("%4s %12s %12s\n", "iter", "f", "‖g‖∞"))

  if (max(abs(gk)) < tol) {
    return(list(x = xk, f = fk, g = gk, niter = 0,
                nfun = nfun, ngr = ngr, converged = TRUE, trace = trace))
  }

  ## ---- main loop -----------------------------------------------------------
  for (k in 1:maxit) {

    # ---- search direction  p_k = -H_k * g_k
    pk <- - two_loop_recursion(gk, s_list, y_list, rho_list)

    # ---- line search (strong Wolfe)
    ls <- strong_wolfe(fk, dot(gk, pk), xk, pk)
    alpha <- ls$alpha
    x_new <- xk + alpha * pk
    f_new <- ls$f
    g_new <- ls$g
    nfun <- nfun + ls$nfev
    ngr  <- ngr  + ls$ngev

    # ---- L-BFGS update
    sk <- x_new - xk
    yk <- g_new - gk
    ys <- dot(yk, sk)

    if (ys > 1e-10) {  # curvature condition
      if (length(s_list) == m) {   # drop the oldest pair
        s_list <- s_list[-1]
        y_list <- y_list[-1]
        rho_list <- rho_list[-1]
      }
      s_list[[length(s_list) + 1]] <- sk
      y_list[[length(y_list) + 1]] <- yk
      rho_list[length(rho_list) + 1] <- 1 / ys
    }

    # ---- convergence test
    grad_inf <- max(abs(g_new))
    trace <- rbind(trace, data.frame(iter = k, f = f_new, grad_inf = grad_inf))

    if (verbose && (k %% 5 == 0 || k == 1))
      cat(sprintf("%4d %12.6e %12.4e\n", k, f_new, grad_inf))

    if (grad_inf < tol)
      return(list(x = x_new, f = f_new, g = g_new, niter = k,
                  nfun = nfun, ngr = ngr, converged = TRUE, trace = trace))

    # ---- prepare next iteration
    xk <- x_new; fk <- f_new; gk <- g_new
  }

  warning("Maximum iterations reached without convergence")
  list(x = xk, f = fk, g = gk, niter = maxit,
       nfun = nfun, ngr = ngr, converged = FALSE, trace = trace)
}

################################################################################
# Example: minimise the 2-D Rosenbrock function
################################################################################

rosenbrock <- function(x) {
  100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
}
rosen_grad <- function(x) {
  c(-400 * x[1] * (x[2] - x[1]^2) - 2 * (1 - x[1]),
     200 * (x[2] - x[1]^2))
}
set.seed(1)
x0 <- runif(2, -1.2, 1.2)          # random start
res <- lbfgs(rosenbrock, rosen_grad, x0, verbose = TRUE)

cat("\nResult:\n")
print(res$x)                        # ≈ c(1, 1)
cat("f =", res$f, "  ‖g‖∞ =", max(abs(res$g)), "\n")