import torch
from collections import defaultdict

from system.utils.model_utils import param_to_vector


class ASAM:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self, init_model=None):
        grad_record = []
        if init_model is None:
            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                grad_record.append(p.grad.data.clone().flatten())
                p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()
        return torch.cat(grad_record)

class SAM(ASAM):
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        grads_record = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads_record.append(p.grad.data.clone().view(-1))
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
            # grads_record.append(eps.data.clone().view(-1))
        self.optimizer.zero_grad()
        return torch.cat(grads_record)


class MoSAM(SAM):
    def __init__(self, optimizer, model, rho, beta, delta):
        super().__init__(optimizer, model, rho)
        self.beta = beta
        self.delta = delta  # 全局-全局更新
        # self.model_parameters_np = model_parameters_np

    @torch.no_grad()
    def descent_step(self):
        idx = 0
        for n, p in self.model.named_parameters():
            layer_size = p.grad.numel()
            shape = p.grad.shape

            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])

            p.grad.mul_(self.beta)
            momentum_grad = self.delta[idx:idx + layer_size].view(shape)[:]
            momentum_grad = momentum_grad.mul_(1 - self.beta).cuda()

            p.grad.add_(momentum_grad)

            idx += layer_size
        self.optimizer.step()
        self.optimizer.zero_grad()

class NagSAM(SAM):
    def __init__(self, optimizer, model, rho, beta, delta_i):
        super().__init__(optimizer, model, rho)
        self.beta = beta
        self.delta_i = delta_i  # 全局-全局更新
        # self.model_parameters_np = model_parameters_np
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        # grads_record = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        idx = 0
        for n, p in self.model.named_parameters():
            layer_size = p.grad.numel()
            shape = p.grad.shape
            if p.grad is None:
                continue
            # grads_record.append(p.grad.data.clone().view(-1))
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            if self.delta_i is not None:
                eps.mul_(self.beta)
                momentum_grad = self.delta_i[idx:idx + layer_size].view(shape)[:]
                momentum_grad = momentum_grad.mul_(1 - self.beta).cuda()
                eps.add_(momentum_grad)
            p.add_(eps)
            idx += layer_size
        self.optimizer.zero_grad()
        # return torch.cat(grads_record)

class GF_ADMM(SAM):
    @torch.no_grad()
    def descent_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()


class LESAM(ASAM):

    @torch.no_grad()
    def ascent_step(self, g_update):
        idx = 0
        for n, p in self.model.named_parameters():
            layer_size = p.grad.numel()
            shape = p.grad.shape
            eps = g_update[idx:idx + layer_size].view(shape)[:]
            eps = eps.mul_(self.rho).cuda()
            p.add_(eps)
            idx += layer_size
        self.optimizer.zero_grad()

class LESAM_D(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"
        self.max_norm = 10

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(LESAM_D, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        # self.g_update=None
        for group in self.param_groups:
            group["rho"] = rho
            # group["adaptive"] = adaptive
        self.paras = None

    @torch.no_grad()
    def first_step(self, g_update):
        # first order sum
        grad_norm = 0
        for group in self.param_groups:
            for idx, p in enumerate(group["params"]):
                p.requires_grad = True
                if g_update == None:
                    continue
                else:
                    grad_norm += g_update[idx].norm(p=2)

        for group in self.param_groups:
            # if g_update !=None:
            scale = group["rho"] / (grad_norm + 1e-7)
            for idx, p in enumerate(group["params"]):
                p.requires_grad = True
                if g_update == None:
                    continue
                # original SAM
                # e_w = p.grad * scale.to(p)
                # ASAM

                # e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                else:
                    e_w = -g_update[idx] * scale.to(p)
                # climb to the local maximum "w + e(w)"
                p.add_(e_w * 1)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]:
                    continue
                # go back to "w" from "w + e(w)"
                p.sub_(self.state[p]["e_w"])
                self.state[p]["e_w"] = 0

    def step(self, g_update):

        inputs, labels, loss_func, model, delta_list, lamb = self.paras

        self.zero_grad()

        self.first_step(g_update)

        param_list = param_to_vector(model)
        predictions = model(inputs)
        loss = loss_func(predictions, labels, param_list, delta_list, lamb)
        self.zero_grad()
        loss.backward()

        self.second_step()


class LESAM_S(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"
        self.max_norm = 10

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(LESAM_S, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        # self.g_update=None
        for group in self.param_groups:
            group["rho"] = rho
            # group["adaptive"] = adaptive
        self.paras = None

    @torch.no_grad()
    def first_step(self, g_update):
        # first order sum
        grad_norm = 0
        for group in self.param_groups:
            for idx, p in enumerate(group["params"]):
                p.requires_grad = True
                if g_update == None:
                    continue
                else:
                    grad_norm += g_update[idx].norm(p=2)

        for group in self.param_groups:
            # if g_update !=None:
            scale = group["rho"] / (grad_norm + 1e-7)
            for idx, p in enumerate(group["params"]):
                p.requires_grad = True
                if g_update == None:
                    continue
                # original SAM
                # e_w = p.grad * scale.to(p)
                # ASAM

                # e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                else:
                    e_w = -g_update[idx] * scale.to(p)
                # climb to the local maximum "w + e(w)"
                p.add_(e_w * 1)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]:
                    continue
                # go back to "w" from "w + e(w)"
                p.sub_(self.state[p]["e_w"])
                self.state[p]["e_w"] = 0

    def step(self, g_update):

        inputs, labels, loss_func, model, delta_list, lamb = self.paras

        self.zero_grad()

        self.first_step(g_update)

        param_list = param_to_vector(model)
        predictions = model(inputs)
        loss = loss_func(predictions, labels, param_list, delta_list, lamb)
        self.zero_grad()
        loss.backward()

        self.second_step()