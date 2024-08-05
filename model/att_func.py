import torch

def get_default_att_func(p=1, tau=0.2, T=2, soft_att=False, **kwargs):
    def att(w):

        w = torch.pow(w, 1 - 0.5 * p)

        w[(w < tau)] = tau
        if T > 0:
            w[(w > T)] = float("inf")

        w = 1 / w

        if not (w == w).all():
            raise "nan occured!"

        return w

    def soft_att_func(w):
        w += 1e-9
        w = w.sqrt()

        e1 = torch.exp(-20 * (w - tau)) 
        e2 = torch.exp(-10 * (T - w)) 
        Z = e1 + e2 + 1
        w = e1 / Z * tau ** (p - 2) + 1 / Z * torch.pow(w, p - 2)

        assert (w == w).all(), 'nan in att func'

        return w
    
    return soft_att_func if soft_att else att

def get_mask_att_func(tau, **kwargs):
    assert tau > 0, 'invalid attention parameter'
    def att(w):
        w[w > tau] = 0
        w[w > 0] = 1
        return w
    return att

def get_log_att_func(ep, **kwargs):
    assert ep > 0
    def att(w):
        return 1 / (w + ep)
    return att

def get_step_p_norm_att_func(p, tau, **kwargs):
    '''This is essentially l_21 norm when p = 1'''
    assert p < 2 and tau > 0
    def att(w):
        w[w < tau] = -1
        w[w >= tau] = p / 2 * (w[w >= tau] / tau ** 2) ** (p / 2 - 1)
        # Or maybe (w[w >= tau] / tau) ** (p - 2) would be better? 
        # in which case \rho not continuous but attention does

        w[w < 0] = tau
        return w
    return att


'''Input: w = z^2 '''


def get_soft_step_l21_att_func(ep, **kwargs):
    '''This is essentially l_21 norm or the weight in adaptive lasso'''
    assert ep > 0
    def att(w):
        return 1 / (torch.sqrt(w) + ep)
    return att

def get_mcp_att_func(gamma, ep=0.01, soft=False, beta=None, **kwargs):
    # x / gamma - x ^ 2 / gamma ^ 2 / 2
    def att(w):
        w += ep
        z = w.sqrt() # convert w to l_21 to match mcp & scad formulation. check with continuity of attention func.

        if torch.where(z < ep)[0].shape != (0,):
            raise ValueError('w should be smaller than ep')

        high_idx = torch.where(z > gamma)
        z[z <= gamma] = 1 / (2 * (z[z <= gamma])) - 1 / (2 * gamma)
        z[high_idx] = 0
        return z
    
    if soft:
        assert beta is not None
    def soft_att(w):
        w += ep
        z = w.sqrt()
        # softmax(1 / 2z - 1 / 2gamma, 0)
        x = 1 / (2 * z) - 1 / (2 * gamma)
        weight = torch.exp(beta * x)
        
        assert (weight == weight).all(), 'nan in soft mcp'
        
        
        # # make sure x is positive xe^bx / (1 + e^bx) = (bx+1)(1+e^bx)-xbe^bx = 0
        # # bx + e^bx + 1 = 0
        # # bx + 1 + bx + b^2 x^2 / 2 ... + 1 = 0
        # # x ~= (-2b +- \sqrt{4b^2 - 4b^2}) / b^2 = -2 / b
        # bias = 2 / beta

        # x = (x + bias) * weight / (1 + weight)
        
        # Well that was stupid...
        x = torch.log(1 + weight) / beta
        return x
    
    return att if not soft else soft_att

def get_l12_att_func(norm, ep=0.01, soft=False, beta=None, **kwargs):
    # x / gamma - x ^ 2 / gamma ^ 2 / 2
    def att(w):
        if norm == 'L1':
            w += ep
            z = w.sqrt() # convert w to l_21 to match mcp & scad formulation. check with continuity of attention func
            z = 1 / (2 * z)
        elif norm == 'L2':
            z = w*0+1
        
        return z
    
    return att 

def get_scad_att_func(lam_att, gamma, ep=0.01, **kwargs):
    assert gamma > 2 and lam_att > 0
    def att(w):
        w += ep
        z = w.sqrt() # convert w to l_21 to match mcp & scad formulation. check with continuity of attention func.

        high_idx = torch.where(z > gamma * lam_att)
        mid_idx = torch.where((lam_att < z) & (z <= lam_att * gamma))
        z[z <= lam_att] = lam_att / 2 / (z[z <= lam_att])
        z[mid_idx] = gamma * lam_att / (2 * (gamma - 1) * z[mid_idx]) - 1 / (2 * (gamma - 1))
        z[high_idx] = 0
        return z
    return att