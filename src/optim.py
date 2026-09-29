import math 
import torch 
 
 
class AdamW(torch.optim.Optimizer): 
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0): 
        if lr < 0: 
            raise ValueError("lr must be non-negative") 
        if eps < 0: 
            raise ValueError("eps must be non-negative") 
        if weight_decay < 0: 
            raise ValueError("weight_decay must be non-negative") 
        if not (0.0 <= betas[0] < 1.0): 
            raise ValueError("beta1 must be in [0, 1)") 
        if not (0.0 <= betas[1] < 1.0): 
            raise ValueError("beta2 must be in [0, 1)") 
 
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay) 
        super().__init__(params, defaults) 
 
        for group in self.param_groups: 
            self._validate_group(group) 
  
    def _validate_group(self, group): 
        if group["lr"] < 0: 
            raise ValueError("lr must be non-negative") 
        if group["eps"] < 0: 
            raise ValueError("eps must be non-negative") 
        if group["weight_decay"] < 0: 
            raise ValueError("weight_decay must be non-negative") 
        beta1, beta2 = group["betas"] 
        if not (0.0 <= beta1 < 1.0): 
            raise ValueError("beta1 must be in [0, 1)") 
        if not (0.0 <= beta2 < 1.0): 
            raise ValueError("beta2 must be in [0, 1)") 
 
    def step(self, closure=None): 
        loss = None 
        if closure is not None: 
            with torch.enable_grad(): 
                loss = closure() 
 
        for group in self.param_groups: 
            self._validate_group(group) 
            lr = group["lr"] 
            beta1, beta2 = group["betas"] 
            eps = group["eps"] 
            weight_decay = group["weight_decay"] 
 
            for p in group["params"]: 
                if p.grad is None: 
                    continue 
                grad = p.grad 
                if grad.is_sparse: 
                    raise RuntimeError("AdamW does not support sparse gradients") 
 
                state = self.state[p] 
                if len(state) == 0: 
                    state["step"] = 0 
                    state["exp_avg"] = torch.zeros_like(p) 
                    state["exp_avg_sq"] = torch.zeros_like(p) 
 
                state["step"] += 1 
                t = state["step"] 
                exp_avg = state["exp_avg"] 
                exp_avg_sq = state["exp_avg_sq"] 
 
                with torch.no_grad(): 
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1) 
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) 
 
                    bias_correction1 = 1 - beta1 ** t 
                    bias_correction2 = 1 - beta2 ** t 
 
                    m_hat = exp_avg / bias_correction1 
                    v_hat = exp_avg_sq / bias_correction2 
 
                    p.mul_(1 - lr * weight_decay) 
                    p.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr) 
 
        return loss 
 
 
def get_lr_cosine_schedule(step, learning_rate_max, learning_rate_min, warmup_steps, cosine_steps): 
    if not isinstance(step, int): 
        raise TypeError("step must be an integer") 
    if step < 0: 
        raise ValueError("step must be non-negative") 
    if not (0 <= learning_rate_min <= learning_rate_max): 
        raise ValueError("must have 0 <= learning_rate_min <= learning_rate_max") 
    if warmup_steps < 0: 
        raise ValueError("warmup_steps must be non-negative") 
    if warmup_steps >= cosine_steps: 
        raise ValueError("warmup_steps must be less than cosine_steps") 
 
    if step < warmup_steps: 
        return step / warmup_steps * learning_rate_max 
    elif step <= cosine_steps: 
        progress = (step - warmup_steps) / (cosine_steps - warmup_steps) 
        return learning_rate_min + 0.5 * (1 + math.cos(math.pi * progress)) * (learning_rate_max - learning_rate_min) 
    else: 
        return learning_rate_min 
 
 
def gradient_clipping(parameters, max_l2_norm): 
    if max_l2_norm <= 0: 
        raise ValueError("max_l2_norm must be positive") 
 
    grads = [p.grad for p in parameters if p.grad is not None] 
    if not grads: 
        return 0.0 
 
    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads)) 
    eps = 1e-6 
    if total_norm > max_l2_norm: 
        scale = max_l2_norm / (total_norm + eps) 
        for g in grads: 
            g.mul_(scale) 
 
    return total_norm.item()