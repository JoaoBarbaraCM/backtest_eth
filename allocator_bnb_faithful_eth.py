import math
import threading
import heapq
import itertools
import numpy as np
from scipy.optimize import minimize, linprog

def x_kink(B,Sm, A, U0=0.9):
    x0 = (B / U0 - Sm) / A
    return x0    

def alpha_beta_low(B, kd=4, U0=0.9):
    '''
    Calculates alpha, beta constants for low utilization 
    x >= x0 -> U <= U0
    '''
    alpha = B * (1 - 1 / kd) / U0
    beta = 1 / kd
    return alpha, beta

def alpha_beta_high(B, kd=4, U0=0.9):
    '''
    Calculates alpha, beta constants for high utilization 
    x < x0 -> U > U0
    '''
    alpha = B * (kd - 1) / (1 - U0)
    beta = (1 - U0 * kd) / (1 - U0)
    return alpha, beta

def mu_itv(alpha, Sm, R0, T=365 * 24 * 3600):
    mu = Sm / (R0 * T * alpha)
    return mu

def curve(alpha, beta, Sm, A, x):
    c = alpha / (Sm + A * x) + beta
    return c

def exponent_rtc(R0, c, T=365 * 24 * 3600):
    exp_rtc = math.exp(R0 * T * c)
    return exp_rtc

def d2f_dx2(B, A, Sm, R0, alpha, beta, x, phi=0, T=365 * 24 * 3600):
    c = curve(alpha, beta, Sm, A, x)
    exp_rtc = exponent_rtc(R0, c, T)
    S = Sm + x * A

    f2 = -B * (1 - phi) * (
        (-2 * A * Sm) * (exp_rtc - 1) / S**3 -
        2 * A * Sm * R0 * T * alpha * exp_rtc / S**4 +
        A**2 * x * R0 * T * alpha * (2 * S + R0 * T * alpha) 
        * exp_rtc / S**5
    )

    return f2

def bisect_d2f(B, A, Sm, R0, alpha, beta, lower, upper,
               lower_val, phi=0, T=365 * 24 * 3600, tol=1e-10):
    '''
    Finds the unique root of d2f_dx2 in (lower, upper) by bisection.
    lower_val / upper_val are the already-computed endpoint values (opposite signs).
    Converges in ~33 iterations for tol=1e-10 on [0, 1].
    '''
    lo, hi = lower, upper
    f_lo = lower_val
    while hi - lo > tol:
        mid = (lo + hi) / 2
        f_mid = d2f_dx2(B, A, Sm, R0, alpha, beta, mid, phi, T)
        if f_mid == 0:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo = mid
            f_lo = f_mid
    return (lo + hi) / 2


def no_positive_roots_f2(B, A, Sm, R0, alpha, beta, lower, upper, phi=0, T=365 * 24 * 3600):
    x_mean = (lower + upper) / 2
    d2fdx2 = d2f_dx2(B, A, Sm, R0, alpha, beta, x_mean, phi, T)
    if d2fdx2 >= 0:
        # f is strictly convex in the interval
        # to find the minimum according to this variable we
        # use gradient descent
        # added the '=' to the condition in case it's so close to 0 that
        # it get's approximated by 0 and in that case we treat it as convex as a safeguard
        return 'convex'
    elif d2fdx2 < 0:
        # f is strictly concave in the interval
        # to find the minimum on this variable we look in
        # the boundaries
        return 'concave'


def bnb_intervals(B, Sm, A, R0, phi=0, T=365 * 24 * 3600, U0=0.9, kd=4):

    x0 = x_kink(B, Sm, A, U0)
    
    intervals = []
    if x0 > 0 and x0 < 1:
        alpha, beta = alpha_beta_high(B, kd, U0)
        intervals.append({'lower': 0, 'upper': x0, 
                    'alpha': alpha, 
                    'beta': beta})
        alpha, beta = alpha_beta_low(B, kd, U0)
        intervals.append({'lower': x0, 'upper': 1, 
                    'alpha': alpha, 
                    'beta': beta})
    elif x0 <= 0:
        # In the cases where we are at the kink of the IRM or under 
        # The kink Utilization is under the target and we would need
        # To remove supply to bring it to the target it is a low  
        # Utilization scenario
        alpha, beta = alpha_beta_low(B, kd, U0)
        intervals.append({'lower': 0, 'upper': 1, 
                    'alpha': alpha, 'beta': beta})
    elif x0 >= 1:
        # In the cases where our assets can at most bring the curve to
        # The kink but never under this is a high utilization scenario
        alpha, beta = alpha_beta_high(B, kd, U0)
        intervals.append({'lower': 0, 'upper': 1, 
                    'alpha': alpha, 'beta': beta})

    bnb_itv = []
    for itv in intervals:
        mu = mu_itv(itv['alpha'], Sm, R0, T)

        if mu >= 1:
            # The second derivative  d2fdx2 has no zeros in x >= 0 so
            # Meaning f is strictly convex or concave
            curvature = no_positive_roots_f2(B, A, Sm, R0, itv['alpha'], itv['beta'], itv['lower'], itv['upper'], phi, T)
            holder = itv.copy()
            holder['curvature'] = curvature
            bnb_itv.append(holder)

        elif mu < 1 and mu >= 0:
            # In this case there is only one root of d2fdx2 in x >= 0
            # we start by verifying if the root is in the interval of interest
            lower_d2fdx2 = d2f_dx2(B, A, Sm, R0, itv['alpha'], itv['beta'], 
                             itv['lower'], phi, T)
            upper_d2fdx2 = d2f_dx2(B, A, Sm, R0, itv['alpha'], itv['beta'], 
                             itv['upper'], phi, T)
            
            # I need to add the edge case where the zero is at the interval boundry
            if upper_d2fdx2*lower_d2fdx2 > 0:
                # same sign means no root
                if lower_d2fdx2 > 0:
                    holder = itv.copy()
                    holder['curvature'] = 'convex'
                    bnb_itv.append(holder)
                else:
                    holder = itv.copy()
                    holder['curvature'] = 'concave'
                    bnb_itv.append(holder)

            elif upper_d2fdx2*lower_d2fdx2 < 0:
                # there is a single root in the interval, find it and split
                x_star = bisect_d2f(B, A, Sm, R0, itv['alpha'], itv['beta'],
                                    itv['lower'], itv['upper'],
                                    lower_d2fdx2, phi, T)
                # left sub-interval [lower, x_star]: curvature follows sign at lower
                holder_lo = itv.copy()
                holder_lo['upper'] = x_star
                holder_lo['curvature'] = 'convex' if lower_d2fdx2 > 0 else 'concave'
                bnb_itv.append(holder_lo)
                # right sub-interval [x_star, upper]: curvature follows sign at upper
                holder_hi = itv.copy()
                holder_hi['lower'] = x_star
                holder_hi['curvature'] = 'convex' if upper_d2fdx2 > 0 else 'concave'
                bnb_itv.append(holder_hi)
            
            else:
                # edge case, the root is the zero, behavior follows mu >= 1
                curvature = no_positive_roots_f2(B, A, Sm, R0, itv['alpha'], itv['beta'], itv['lower'], itv['upper'], phi, T)
                holder = itv.copy()
                holder['curvature'] = curvature
                bnb_itv.append(holder)

    return bnb_itv

def is_feasible(leaf):
    # True if is feasible False otherwise
    return sum(itv['lower'] for itv in leaf) <= 1 <= sum(itv['upper'] for itv in leaf)

def check_curvature(leaf):
    curv = list(dict.fromkeys([itv['curvature'] for itv in leaf]))
    if len(curv) == 1:
        return curv[0]
    else:
        return 'mixed'

def f_market(x_val, itv):
    '''
    Single-market objective:
      f(x) = -B*(1-phi) * (x/S) * (exp(R0*T*c) - 1)
    where S = Sm + x*A, c = alpha/S + beta, x in [0,1] is the allocation fraction.
    x/S is the fraction of total supply contributed by the new allocation.
    Minimising f maximises effective supply APY weighted by allocation fraction.
    '''
    S = itv['Sm'] + x_val * itv['A']
    c = itv['alpha'] / S + itv['beta']
    return -itv['B'] * (1 - itv['phi']) * (x_val / S) * (math.exp(itv['R0'] * itv['T'] * c) - 1)


def df_market(x_val, itv):
    '''
    Exact first derivative of f with respect to x for one market:
      f'(x) = -B*(1-phi) * [Sm/S² * (E-1) - x*A*R0*T*alpha*E/S³]
    '''
    S = itv['Sm'] + x_val * itv['A']
    c = itv['alpha'] / S + itv['beta']
    E = math.exp(itv['R0'] * itv['T'] * c)
    return -itv['B'] * (1 - itv['phi']) * (
        itv['Sm'] / S**2 * (E - 1)
        - x_val * itv['A'] * itv['R0'] * itv['T'] * itv['alpha'] * E / S**3
    )


def convex_solver(leaf):
    '''
    Solve  min Σ f_i(x_i)  subject to  Σx_i = 1,  l_i <= x_i <= u_i
    for a fully convex leaf.
    All f_i are convex on their sub-intervals, so the sum is convex on a
    convex domain: any local minimum is global. SLSQP is exact here.

    Returns (x_opt, f_opt):
        x_opt : dict {Id -> optimal allocation fraction}
        f_opt : float, total objective value (negative of total yield)
    '''
    n = len(leaf)

    def objective(x):
        return sum(f_market(x[i], leaf[i]) for i in range(n))

    def jacobian(x):
        return np.array([df_market(x[i], leaf[i]) for i in range(n)])

    bounds      = [(itv['lower'], itv['upper']) for itv in leaf]
    constraints = {'type': 'eq', 'fun': lambda x: x.sum() - 1,
                                 'jac': lambda _: np.ones(n)}

    # Starting point: interval midpoints normalised to the simplex.
    # SLSQP tolerates mildly infeasible starts and converges to the
    # feasible minimum regardless.
    x0 = np.array([(itv['lower'] + itv['upper']) / 2 for itv in leaf])
    x0 = x0 / x0.sum()

    result = minimize(
        objective, x0,
        method='SLSQP',
        jac=jacobian,
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-14, 'maxiter': 1000},
    )

    x_opt = {leaf[i]['Id']: result.x[i] for i in range(n)}
    return x_opt, result.fun

def concave_solver(leaf):
    '''
    Solve  min Σ f_i(x_i)  subject to  Σx_i = 1,  l_i <= x_i <= u_i
    for a fully concave leaf.

    The minimum of a concave function over a convex polytope is attained at
    a vertex of that polytope.  On the box-simplex the vertices have the form:
    pick one "free" market i; fix every other market j at either l_j or u_j;
    set x_i = 1 - Σ_{j≠i} x_j and check l_i ≤ x_i ≤ u_i.

    At most k × 2^(k-1) candidates are evaluated.

    Returns (x_opt, f_opt):
        x_opt : dict {Id -> optimal allocation fraction}
        f_opt : float, total objective value (negative of total yield)
    '''
    n = len(leaf)
    best_val = math.inf
    best_x   = None

    for free_idx in range(n):
        others = [i for i in range(n) if i != free_idx]
        for bits in range(1 << (n - 1)):          # 2^(n-1) bound combos
            x = np.empty(n)
            fixed_sum = 0.0
            for j, idx in enumerate(others):
                x[idx] = leaf[idx]['upper'] if (bits >> j) & 1 else leaf[idx]['lower']
                fixed_sum += x[idx]
            x_free = 1.0 - fixed_sum
            lo, hi = leaf[free_idx]['lower'], leaf[free_idx]['upper']
            eps = 1e-9
            if lo - eps <= x_free <= hi + eps:
                # if x_free < lo: x_free = lo
                # if x_free > hi: x_free = hi
                x[free_idx] = x_free
                val = sum(f_market(x[i], leaf[i]) for i in range(n))
                if val < best_val:
                    best_val = val
                    best_x   = x.copy()

    if best_x is None:
        return None, math.inf

    x_opt = {leaf[i]['Id']: best_x[i] for i in range(n)}
    return x_opt, best_val


def mixed_solver(leaf, max_threads=8, tol=1e-8, n_starts=8, max_nodes=500):
    '''
    Solve  min Σ f_i(x_i)  subject to  Σx_i = 1,  l_i <= x_i <= u_i
    for a mixed-curvature leaf (some f_i convex, some concave).

    Algorithm — Branch-and-Bound with LP relaxation
    ------------------------------------------------
    Each B&B node is a sub-box of the original leaf bounds.  At every node:

    Lower bound (LP relaxation)
      Each f_i is replaced by a linear underestimator L_i:
        • Convex  f_i → tangent at the sub-interval midpoint.
          By convexity, f_i(x) ≥ L_i(x) everywhere on the interval.
        • Concave f_i → chord through the two endpoints.
          By concavity (Jensen), f_i(x) ≥ chord(x) everywhere on the interval.
      Minimising Σ L_i(x_i) over the box-simplex is a linear program (LP)
      solved with scipy linprog.  Its optimal value is a valid lower bound.

    Upper bound (multi-start SLSQP)
      n_starts random feasible points are generated inside the current
      sub-box and used as starting points for SLSQP.  The best feasible
      value found updates the global incumbent.

    Branching
      The dimension with the largest linearisation gap
        gap_i = max_{x ∈ {l_i, mid_i, u_i}} (f_i(x) − L_i(x))
      is split at its midpoint.  This is where the relaxation is loosest
      and splitting there tightens the bound fastest.

    Pruning
      A node is discarded if its LP lower bound ≥ current incumbent − tol.

    Convergence
      A node is not branched further if (incumbent − lower_bound) < tol
      or the global node counter reaches max_nodes.

    Parallelism
      When a node is branched, each child is either:
        (a) dispatched immediately to a new thread, if fewer than
            max_threads threads are currently live, or
        (b) pushed onto a min-heap priority queue ordered by lower bound.
      Threads drain the queue on completion, so the pool self-balances
      and the best nodes are always processed first once the thread
      budget is saturated.

    Returns (x_opt, f_opt):
        x_opt : dict {Id -> optimal allocation fraction}  or None
        f_opt : float, total objective value
    '''
    n = len(leaf)

    # ------------------------------------------------------------------ #
    # Per-market helpers (close over leaf; leaf[i] params never change)
    # ------------------------------------------------------------------ #

    def _f(i, x_val):
        return f_market(x_val, leaf[i])

    def _df(i, x_val):
        return df_market(x_val, leaf[i])

    def _linear_underestimator(i, lo_i, hi_i):
        '''
        Return (a, b) such that L_i(x) = a + b*x  underestimates  f_i  on [lo_i, hi_i].

        Convex  → tangent at midpoint:  f(mid) + f'(mid)*(x − mid)
        Concave → chord:                f(lo) + slope*(x − lo),   slope = (f(hi)−f(lo))/(hi−lo)
        '''
        mid = (lo_i + hi_i) / 2.0
        if leaf[i]['curvature'] == 'convex':
            f_mid  = _f(i, mid)
            df_mid = _df(i, mid)
            b = df_mid
            a = f_mid - df_mid * mid
        else:  # concave
            f_lo = _f(i, lo_i)
            f_hi = _f(i, hi_i)
            b = (f_hi - f_lo) / (hi_i - lo_i) if hi_i > lo_i else 0.0
            a = f_lo - b * lo_i
        return a, b

    def _lower_bound(bounds):
        '''
        Solve the LP relaxation on the current sub-box.
        Returns the LP optimal value (valid lower bound on F over this node).
        '''
        a_vec = np.empty(n)
        b_vec = np.empty(n)
        for i, (lo_i, hi_i) in enumerate(bounds):
            a_vec[i], b_vec[i] = _linear_underestimator(i, lo_i, hi_i)

        res = linprog(
            c       = b_vec,
            A_eq    = np.ones((1, n)),
            b_eq    = [1.0],
            bounds  = bounds,
            method  = 'highs',
        )
        # If LP is infeasible or fails, return −∞ so the node is never pruned
        if res.status != 0:
            return -math.inf
        return float(res.fun) + float(a_vec.sum())

    def _gap(i, lo_i, hi_i):
        '''
        Max pointwise linearisation gap for market i over {lo_i, mid_i, hi_i}.
        Always ≥ 0 because L_i is an underestimator.
        '''
        mid = (lo_i + hi_i) / 2.0
        a, b = _linear_underestimator(i, lo_i, hi_i)
        return max(_f(i, x) - (a + b * x) for x in (lo_i, mid, hi_i))

    def _find_split(bounds):
        '''Return (dim, split_value): split the dimension with the largest gap at its midpoint.'''
        gaps  = [_gap(i, bounds[i][0], bounds[i][1]) for i in range(n)]
        dim   = int(np.argmax(gaps))
        value = (bounds[dim][0] + bounds[dim][1]) / 2.0
        return dim, value

    def _random_feasible(bounds):
        '''
        Generate one random feasible point inside the box-simplex intersection.
        Strategy: start at lower bounds, then randomly distribute the remaining
        slack across dimensions (in random order, clipped to each upper bound).
        '''
        x   = np.array([b[0] for b in bounds], dtype=float)
        slack = 1.0 - x.sum()
        if slack < -1e-9:
            return None
        order = np.random.permutation(n)
        for i in order:
            add   = min(float(slack), bounds[i][1] - x[i])
            x[i] += add
            slack -= add
            if slack < 1e-12:
                break
        return x if slack < 1e-8 else None

    def _upper_bound(bounds):
        '''
        Multi-start SLSQP upper bound.  Returns (best_val, best_x) or
        (inf, None) if every run fails.
        '''
        best_val = math.inf
        best_x   = None
        for _ in range(n_starts):
            x0 = _random_feasible(bounds)
            if x0 is None:
                continue
            cons = {'type': 'eq', 'fun': lambda x: x.sum() - 1,
                                  'jac': lambda _: np.ones(n)}
            res = minimize(
                lambda x: sum(f_market(x[i], leaf[i]) for i in range(n)),
                x0,
                method='SLSQP',
                jac=lambda x: np.array([df_market(x[i], leaf[i]) for i in range(n)]),
                bounds=bounds,
                constraints=cons,
                options={'ftol': 1e-14, 'maxiter': 500},
            )
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x   = res.x.copy()
        return best_val, best_x

    # ------------------------------------------------------------------ #
    # Shared B&B state  (all writes protected by _lock)
    # ------------------------------------------------------------------ #
    _lock         = threading.Lock()
    _pq           = []                   # min-heap: (lb, uid, bounds)
    _uid          = itertools.count()
    _incumbent    = [math.inf]
    _best_x       = [None]
    _thread_count = [0]                  # live worker threads
    _task_count   = [0]                  # nodes queued + in-progress
    _nodes_done   = [0]
    _all_done     = threading.Event()

    # ------------------------------------------------------------------ #
    # Thread body: process one node, then drain the queue while items remain
    # ------------------------------------------------------------------ #
    def _thread_func(initial_bounds):
        bounds = initial_bounds
        while bounds is not None:
            _process_node(bounds)

            with _lock:
                _task_count[0] -= 1          # this node is finished

                # Skip pruned items; find the next valid node to process
                next_bounds = None
                while _pq:
                    lb, _, candidate = heapq.heappop(_pq)
                    if lb < _incumbent[0] - tol:
                        next_bounds = candidate  # will process in next loop iteration
                        break
                    else:
                        _task_count[0] -= 1      # pruned item — drain its count

                if next_bounds is None and _task_count[0] == 0:
                    _all_done.set()              # nothing left anywhere

            bounds = next_bounds

        with _lock:
            _thread_count[0] -= 1

    # ------------------------------------------------------------------ #
    # Schedule a node: spawn a thread if under the limit, else push to queue
    # ------------------------------------------------------------------ #
    def _schedule(bounds, lb):
        with _lock:
            if lb >= _incumbent[0] - tol:
                return                           # pruned before even queuing
            _task_count[0] += 1
            if _thread_count[0] < max_threads:
                _thread_count[0] += 1
                do_spawn = True
            else:
                heapq.heappush(_pq, (lb, next(_uid), bounds))
                do_spawn = False
        if do_spawn:
            threading.Thread(target=_thread_func, args=(bounds,), daemon=True).start()

    # ------------------------------------------------------------------ #
    # Process a single B&B node
    # ------------------------------------------------------------------ #
    def _process_node(bounds):
        lb = _lower_bound(bounds)

        with _lock:
            if lb >= _incumbent[0] - tol:
                return                           # pruned by fresh incumbent

        ub, x_cand = _upper_bound(bounds)

        with _lock:
            if ub < _incumbent[0]:
                _incumbent[0] = ub
                _best_x[0]    = x_cand
            current_inc = _incumbent[0]
            _nodes_done[0] += 1
            at_limit = _nodes_done[0] >= max_nodes

        # Stop branching if gap is tight or node budget exhausted
        if current_inc - lb < tol or at_limit:
            return

        dim, split_val = _find_split(bounds)

        lo_bounds       = list(bounds)
        lo_bounds[dim]  = (bounds[dim][0], split_val)
        hi_bounds       = list(bounds)
        hi_bounds[dim]  = (split_val, bounds[dim][1])

        for child in (lo_bounds, hi_bounds):
            child_lb = _lower_bound(child)
            _schedule(child, child_lb)

    # ------------------------------------------------------------------ #
    # Kick off with the root node (the full leaf box)
    # ------------------------------------------------------------------ #
    root_bounds = [(itv['lower'], itv['upper']) for itv in leaf]
    root_lb     = _lower_bound(root_bounds)
    _schedule(root_bounds, root_lb)
    _all_done.wait()

    if _best_x[0] is None:
        return None, math.inf

    x_opt = {leaf[i]['Id']: _best_x[0][i] for i in range(n)}
    return x_opt, _incumbent[0]


# ══════════════════════════════════════════════════════════════════════════════
# Faithful-allocator extensions
# ══════════════════════════════════════════════════════════════════════════════

IDLE_MARKET_ID = 'idle'   # sentinel Id for the zero-yield virtual market


def clip_intervals(intervals, x_min=0.0, x_max=1.0):
    '''
    Restrict a list of curvature-labelled sub-intervals to [x_min, x_max].

    This enforces locked-fund constraints: a lender who cannot reduce their
    position below x_min (because high utilisation blocks withdrawal) has an
    effective domain [x_min, x_max] instead of [0, 1].  Intervals that
    become empty after clipping are discarded entirely.

    Parameters
    ----------
    intervals : list of interval dicts from bnb_intervals()
    x_min     : hard lower bound on allocation fraction  (default 0)
    x_max     : hard upper bound on allocation fraction  (default 1)

    Returns
    -------
    list of clipped interval dicts (may be shorter than the input list)
    '''
    result = []
    for itv in intervals:
        lo = max(itv['lower'], x_min)
        hi = min(itv['upper'], x_max)
        if lo < hi:                       # non-empty after clipping
            clipped          = itv.copy()
            clipped['lower'] = lo
            clipped['upper'] = hi
            result.append(clipped)
    return result


def _make_idle_intervals(A):
    '''
    Single interval representing the zero-yield idle "market".

    Setting B = 0 makes f_market(x) = 0 and df_market(x) = 0 for all x,
    so the optimizer treats allocation to idle as free of yield.  The idle
    slot absorbs whatever fraction real markets cannot profitably use and
    guarantees the simplex stays feasible even under tight locked-fund
    constraints.

    Sm = 1e30 prevents division-by-zero inside f_market / df_market.
    curvature = 'convex' routes the idle market through the SLSQP path.
    bnb_intervals() is intentionally bypassed (B = 0, R0 = 0 would cause
    division-by-zero in mu_itv).
    '''
    return [{
        'lower': 0.0, 'upper': 1.0,
        'alpha': 0.0, 'beta':  0.0,
        'curvature': 'convex',
        'Id':  IDLE_MARKET_ID,
        'B':   0.0, 'Sm': 1e30, 'R0': 0.0,
        'A':   A,   'phi': 0.0,
        'T':   365 * 24 * 3600, 'U0': 0.9, 'kd': 4.0,
    }]


def allocator(market_states, A, x_min_per_market=None, include_idle=True):
    '''
    Multi-market allocator with locked-fund support and optional idle slot.

    Drop-in replacement for the original allocator() with two extra parameters.

    Parameters
    ----------
    market_states    : list of market state dicts (same schema as before —
                       keys Id, B, Sm, R0, phi, T, U0, kd)
    A                : total allocation size (raw token units)
    x_min_per_market : dict {market_Id -> x_min_fraction}
                       Minimum allocation fraction for each market, set by the
                       caller to reflect funds that cannot be withdrawn due to
                       high utilisation.  Missing keys default to x_min = 0.
    include_idle     : bool (default True)
                       When True a virtual zero-yield market is appended so
                       the optimizer can choose to leave capital uninvested.
                       This also guarantees feasibility when real markets have
                       tight locked lower bounds.

    Returns
    -------
    (best_alloc, best_val)
        best_alloc : dict {market_Id -> fraction}   or  None if infeasible
        best_val   : float  (negative of total expected yield)

    Infeasibility
    -------------
    If the sum of locked lower bounds already ≥ 1, no allocation satisfying
    Σx_i = 1 exists within the constraints.  The function returns (None, inf)
    rather than raising, so the caller can skip the rebalance and hold the
    current positions.
    '''
    if x_min_per_market is None:
        x_min_per_market = {}

    # ── guard: if total locked fraction >= 1, no feasible rebalance ───────
    total_locked = sum(x_min_per_market.get(m['Id'], 0.0) for m in market_states)
    if total_locked >= 1.0:
        return None, math.inf

    # ── build curvature-labelled intervals for each real market ────────────
    for market in market_states:
        x_min = x_min_per_market.get(market['Id'], 0.0)

        intervals = bnb_intervals(
            market['B'], market['Sm'], A, market['R0'],
            phi=market['phi'], T=market['T'],
            U0=market['U0'],   kd=market['kd'],
        )

        # Clip to [x_min, 1]: enforce the locked-fund lower bound
        intervals = clip_intervals(intervals, x_min=x_min, x_max=1.0)

        for itv in intervals:
            itv['Id']  = market['Id']
            itv['B']   = market['B']
            itv['Sm']  = market['Sm']
            itv['R0']  = market['R0']
            itv['A']   = A
            itv['phi'] = market['phi']
            itv['T']   = market['T']
            itv['U0']  = market['U0']
            itv['kd']  = market['kd']

        market['intervals'] = intervals

    # ── append idle virtual market (bypasses bnb_intervals) ───────────────
    if include_idle:
        all_markets = list(market_states) + [
            {'Id': IDLE_MARKET_ID, 'intervals': _make_idle_intervals(A)}
        ]
    else:
        all_markets = list(market_states)

    # ── Cartesian product of sub-intervals; keep feasible leaves ──────────
    all_intervals    = [m['intervals'] for m in all_markets]
    cartesian_leaves = list(itertools.product(*all_intervals))
    feasible_leaves  = [leaf for leaf in cartesian_leaves if is_feasible(leaf)]

    if not feasible_leaves:
        return None, math.inf

    # ── tag each leaf with its solver, sort convex → concave → mixed ──────
    leafs_solver = [
        {'solver': check_curvature(leaf), 'leaf': leaf}
        for leaf in feasible_leaves
    ]
    custom_order = {'convex': 0, 'concave': 1, 'mixed': 2}
    leafs_solver.sort(key=lambda item: custom_order[item['solver']])

    # ── solve each leaf; track global incumbent ────────────────────────────
    best_val   = math.inf
    best_alloc = None

    for item in leafs_solver:
        if item['solver'] == 'convex':
            x_opt, f_opt = convex_solver(item['leaf'])
        elif item['solver'] == 'concave':
            x_opt, f_opt = concave_solver(item['leaf'])
        else:
            x_opt, f_opt = mixed_solver(item['leaf'])

        if x_opt is not None and f_opt < best_val:
            best_val   = f_opt
            best_alloc = x_opt

    return best_alloc, best_val
    
def _print_market_intervals(markets, A):
    '''Print the curvature classification for every interval of every market.'''
    print(f"\n{'─'*60}")
    print(f"  Interval curvature diagnostics  (A = {A:.3e})")
    print(f"{'─'*60}")
    for m in markets:
        itvs = bnb_intervals(m['B'], m['Sm'], A, m['R0'],
                             phi=m['phi'], T=m['T'], U0=m['U0'], kd=m['kd'])
        tags = [(f"[{i['lower']:.4f}, {i['upper']:.4f}] {i['curvature']}") for i in itvs]
        print(f"  Market {m['Id']}  B={m['B']:.2e}  Sm={m['Sm']:.2e}  →  {tags}")


def main():
    # ------------------------------------------------------------------ #
    # Curvature theory recap (second-derivative sign):
    #
    #   d²f/dx² sign is driven by the competition between:
    #     • Terms 1+2  ∝  A·Sm / S³   (negative contribution)
    #     • Term 3     ∝  A²·x / S³   (positive contribution → convex)
    #
    #   Ratio Term3 / Term1  ≈  A·x / Sm
    #
    #   A >> Sm  →  Term3 wins  →  SUM > 0  →  d²f/dx² = -B·SUM < 0  → CONCAVE
    #   A << Sm  →  Terms1+2 win →  SUM < 0 →  d²f/dx² = -B·SUM > 0  → CONVEX
    #
    # Strategy: with A = 1e15, set:
    #   Market 1  Sm = 1e16  (Sm >> A  → convex)
    #   Market 2  Sm = 1e13  (Sm << A  → concave)
    # Both markets yield a single interval [0,1], giving one Cartesian leaf
    # labelled 'mixed'.  mixed_solver is therefore exercised directly.
    # ------------------------------------------------------------------ #

    T = 365 * 24 * 3600
    A = 1e15

    markets = [
        # High liquidity market: Sm >> A → small A/Sm ratio → convex
        {'Id': 1, 'B': 9.4e14, 'Sm': 1e16, 'R0': 1.35e-9,
         'phi': 0, 'T': T, 'U0': 0.9, 'kd': 4},
        # Low liquidity market: Sm << A → large A/Sm ratio → concave
        {'Id': 2, 'B': 8e12,   'Sm': 1e13, 'R0': 1.35e-9,
         'phi': 0, 'T': T, 'U0': 0.9, 'kd': 4},
    ]

    _print_market_intervals(markets, A)

    print(f"\n{'─'*60}")
    print("  Running allocator …")
    print(f"{'─'*60}")
    alloc, val = allocator(markets, A)
    print(f"  best allocation : {alloc}")
    print(f"  best objective  : {val:.6e}")
    
if __name__ == '__main__':
    main()