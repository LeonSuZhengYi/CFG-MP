import torch
import numpy as np

class AndersonFlowScheduler:
    """
    A scheduler implementing Flow Matching with Anderson Acceleration (AA).
    This class handles the noise schedule, timestep alignment based on D2F (Diffusion-to-Flow), 
    fixed-point iteration for position correction, and ODE stepping.
    """

    def __init__(self, num_inference_steps=12, beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000):
        """
        Initializes the scheduler with a discrete diffusion noise schedule and aligns 
        Flow-Matching timesteps.

        Args:
            num_inference_steps (`int`):
                The number of denoising steps during inference.
            beta_start (`float`):
                The starting value of the beta schedule.
            beta_end (`float`):
                The ending value of the beta schedule.
            num_train_timesteps (`int`):
                The number of timesteps used during model training (typically 1000).
        """
        self.num_inference_steps = num_inference_steps
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sigmas_all, self.timesteps_all = self._get_flow_matching_sigmas(num_inference_steps, alphas_cumprod)

    def _get_flow_matching_sigmas(self, num_steps, alphas_cumprod):
        """
        Aligns the inference sampling points with the training distribution by finding 
        nearest neighbor indices in Flow-equivalent space (D2F Alignment).

        Args:
            num_steps (`int`):
                The number of inference steps.
            alphas_cumprod (`torch.Tensor`):
                Cumulative product of (1 - beta) from the training schedule.

        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`:
                - **sigmas**: The mapped noise levels (float32).
                - **indices**: The corresponding training timestep indices (long).
        """
        sigmas_base = ((1.0 - alphas_cumprod) / alphas_cumprod) ** 0.5
        alpha_s = 1.0 / ((sigmas_base**2 + 1.0) ** 0.5)
        scale_s = 1.0 / (alpha_s + (1.0 - alpha_s**2) ** 0.5)
        floweq_s = alpha_s * scale_s
        ideal_t = np.linspace(0, 1, num_steps, endpoint=False)
        diff = np.abs(ideal_t[:, None] - floweq_s.numpy()[None, :])
        indices = np.argmin(diff, axis=1)
        sigmas = np.concatenate([sigmas_base[indices].numpy(), [0.0]]).astype(np.float32)
        return torch.from_numpy(sigmas), torch.from_numpy(indices).long()

    def get_step_params(self, step_idx, device):
        """
        Calculates physical coefficients for the current step, including 
        alpha, sigma, scale, and flow-equivalent timesteps.

        Args:
            step_idx (`int`):
                The current step index in the inference loop.
            device (`torch.device`):
                The device (e.g., 'cuda') for the output tensors.

        Returns:
            `Dict[str, torch.Tensor]`:
                A dictionary containing parameters for ODE stepping (alpha_c, sigma_tc, scale_c, etc.)
        """
        s_curr, s_next = self.sigmas_all[step_idx], self.sigmas_all[step_idx + 1]
        def get_single(sigma_val):
            sigma = torch.tensor(sigma_val, dtype=torch.float64, device=device)
            alpha_t = 1.0 / ((sigma**2 + 1.0) ** 0.5)
            scale = 1.0 / (alpha_t + (1.0 - alpha_t**2) ** 0.5)
            return alpha_t.float(), (sigma * alpha_t).float(), scale.float(), (alpha_t * scale).float()
        ac, stc, sc, tc = get_single(s_curr)
        _, _, sn, tn = get_single(s_next)
        return {"alpha_c": ac, "sigma_tc": stc, "scale_c": sc, "t_curr": tc, 
                "t_next": tn, "scale_n": sn, "dt": tn - tc, "t_index": self.timesteps_all[step_idx].to(device)}

    def solve_anderson_weights(self, residuals_stack, regularization=1e-4):
        """
        Solves the least-squares optimization for Anderson Acceleration to find the 
        optimal linear combination of historical residuals.

        Args:
            residuals_stack (`torch.Tensor`):
                A stack of residual tensors (G(x) - x) of shape (Batch, Window, Dim).
            regularization (`float`):
                The regularization factor for matrix inversion stability.

        Returns:
            `torch.Tensor`:
                The calculated coefficients (weights) for the history window.
        """
        B, M, D = residuals_stack.shape
        H = torch.bmm(residuals_stack, residuals_stack.transpose(1, 2))
        H = H + regularization * torch.eye(M, device=residuals_stack.device).unsqueeze(0)
        ones = torch.ones(B, M, 1, device=residuals_stack.device)
        try:
            alpha_unnorm = torch.linalg.solve(H, ones)
        except RuntimeError: alpha_unnorm = ones
        coeffs = alpha_unnorm / (alpha_unnorm.sum(dim=1, keepdim=True) + 1e-8)
        return coeffs.unsqueeze(-1).unsqueeze(-1)

    def step_anderson_correction(self, model, latents, params, labels, null_labels, aa_steps, latent_c, m=1, use_aa=False, damping_beta=1.0):
        """
        Phase 1: Refines the latent position using fixed-point iteration (Picard) or 
        Anderson Acceleration to minimize trajectory error.

        Args:
            model (`nn.Module`):
                The Transformer model.
            latents (`torch.Tensor`):
                The current latent tensor.
            params (`Dict`):
                Step parameters from `get_step_params`.
            labels (`torch.Tensor`):
                Conditional class labels.
            null_labels (`torch.Tensor`):
                Unconditional labels (null class).
            aa_steps (`int`):
                Number of correction iterations.
            latent_c (`int`):
                Number of latent channels.
            m (`int`):
                The history window size for Anderson Acceleration.
            use_aa (`bool`):
                Whether to use Anderson Acceleration (True) or Picard iteration (False).
            damping_beta (`float`):
                The damping factor (relaxation factor). 1.0 means full AA update, 
                lower values increase stability.

        Returns:
            `torch.Tensor`:
                The corrected latent tensor.
        """
        bs = latents.shape[0]
        G_list, F_list, X_list = [], [], []
        x_k = latents
        t_idx = params['t_index'].expand(bs)
        
        for k in range(aa_steps):
            # 1. Fixed-point operator g(x) calculation
            out_u = model(x_k, timestep=t_idx, class_labels=null_labels).sample
            if out_u.shape[1] // 2 == latent_c: out_u = out_u.chunk(2, dim=1)[0]
            x0_u = (x_k - params['sigma_tc'] * out_u) / params['alpha_c']
            v_u = (x0_u - params['scale_c'] * x_k) / (1.0 - params['t_curr'] + 1e-7)
            x_half = (params['scale_c'] * x_k - (params['dt'] / 2) * v_u) / params['scale_c']
            
            out_c = model(x_half, timestep=t_idx, class_labels=labels).sample
            if out_c.shape[1] // 2 == latent_c: out_c = out_c.chunk(2, dim=1)[0]
            x0_c = (x_half - params['sigma_tc'] * out_c) / params['alpha_c']
            v_c = (x0_c - params['scale_c'] * x_half) / (1.0 - params['t_curr'] + 1e-7)
            g_k = (params['scale_c'] * x_half + (params['dt'] / 2) * v_c) / params['scale_c']
            
            if not use_aa:
                # Pure Picard Iteration (CFG-MP)
                x_k = g_k
            else:
                # Anderson Acceleration (CFG-MP+)
                f_k = g_k - x_k
                G_list.append(g_k)
                F_list.append(f_k.reshape(bs, -1))
                X_list.append(x_k)
                
                # Maintain Sliding Window
                if len(G_list) > (m + 1):
                    G_list.pop(0)
                    F_list.pop(0)
                    X_list.pop(0)
                
                if k == 0:
                    x_k = g_k
                else:
                    coeffs = self.solve_anderson_weights(torch.stack(F_list, dim=1))
                    
                    # Compute weighted averages
                    sum_g = (coeffs * torch.stack(G_list, dim=1)).sum(dim=1)
                    
                    if damping_beta == 1.0:
                        x_k = sum_g
                    else:
                        # Apply damping factor: (1-beta)*sum(alpha*x) + beta*sum(alpha*g)
                        sum_x = (coeffs * torch.stack(X_list, dim=1)).sum(dim=1)
                        x_k = (1.0 - damping_beta) * sum_x + damping_beta * sum_g
                        
        return x_k

    def step_cfg_flow(self, model, latents, params, labels, null_labels, guidance_scale, latent_c, step_idx):
        """
        Phase 2: Performs Classifier-Free Guidance (CFG) and updates the latent using 
        the Flow Matching stepping logic.

        Args:
            model (`nn.Module`):
                The Transformer model.
            latents (`torch.Tensor`):
                Current latent tensor.
            params (`Dict`):
                Step parameters.
            labels (`torch.Tensor`):
                Conditional class labels.
            null_labels (`torch.Tensor`):
                Unconditional labels.
            guidance_scale (`float`):
                The CFG scale.
            latent_c (`int`):
                Number of latent channels.
            step_idx (`int`):
                Current step index.

        Returns:
            `torch.Tensor`:
                The updated latent tensor for the next timestep.
        """
        curr_bs = latents.shape[0]
        model_input = torch.cat([latents, latents], dim=0)
        t_in = params['t_index'].expand(curr_bs * 2)
        lbl_in = torch.cat([labels, null_labels], dim=0)
        
        model_out = model(model_input, timestep=t_in, class_labels=lbl_in).sample
        eps = model_out[:, :latent_c]
        eps_c, eps_u = eps.chunk(2, dim=0)
        
        eps_cfg = eps_u + guidance_scale * (eps_c - eps_u)
        x0_cfg = (latents - params['sigma_tc'] * eps_cfg) / params['alpha_c']
        v_cfg = (x0_cfg - params['scale_c'] * latents) / (1.0 - params['t_curr'] + 1e-7)
        
        dt_val = params['t_next'] if step_idx == 0 else params['dt']
        next_flow_x = params['scale_c'] * latents + dt_val * v_cfg
        
        return (next_flow_x / params['scale_n']).to(latents.dtype)