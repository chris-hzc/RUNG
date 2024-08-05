

def w_l1(y):
    # d_{y^2} y = 1 / 2y
    return 1 / (2 * y)

def w_ruge(y, gamma, max_z=1e3):
    # rho of MCP, used in RUNG
    rho = 1 / (2 * y) - 1 / (2 * gamma)
    return rho.clamp(min=0, max=max_z)

def get_w_ruge(gamma, **kwargs):
    return lambda y: w_ruge(y, gamma, **kwargs)