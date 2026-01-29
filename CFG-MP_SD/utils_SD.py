import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.configuration_utils import register_to_config, ConfigMixin
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput


class CFGMPScheduler(SchedulerMixin, ConfigMixin):
    """
    Various functions for implementing CFG-MP/MP+.
    
    """
    _compatibles = []
    order = 1 

    @register_to_config
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 3.0):
        """
        Initializes the scheduler.

        Args:
            num_train_timesteps (`int`): Total training timesteps.
            shift (`float`): The mu/shift parameter for the sigma distribution.
        """
        self.timesteps = None
        self.sigmas = None
        self.num_inference_steps = None
        self.shift = shift 

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None, mu: Optional[float] = None, **kwargs):
        """
        Sets the discrete timesteps used for the diffusion chain.

        Args:
            num_inference_steps (`int`): The number of diffusion steps.
            device (`str` or `torch.device`): The device to move timesteps to.
            mu (`float`, optional): Overrides the default shift value.
        """
        self.num_inference_steps = num_inference_steps
        total_steps = num_inference_steps * 2
        sigmas = np.linspace(1.0, 1e-4, total_steps)
        mu = mu if mu is not None else self.shift
        sigmas = mu * sigmas / (1 + (mu - 1) * sigmas)
        sigmas = np.append(sigmas, 0.0) 
        self.sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        self.timesteps = self.sigmas * 1000.0

    def get_fine_grained_times(self, step_index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves the current, midpoint, and next timestamps for the current loop.

        Args:
            step_index (`int`): The current step index in the inference loop.

        Returns:
            `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: (t_curr, t_mid, t_next).
        """
        idx = step_index * 2
        return self.sigmas[idx], self.sigmas[idx + 1], self.sigmas[idx + 2]
    
    def solve_anderson_mixing(
        self, 
        z_history: List[torch.Tensor], 
        f_history: List[torch.Tensor], 
        beta: float = 1.0, 
        ridge: float = 1e-4
    ) -> torch.Tensor:
        """
        Solves the Anderson Acceleration mixing to find the next extrapolated state.
        Formula: $x_{k+1} = (1 - \beta) \sum_{i=0}^{m} \alpha_i x_{k-i} + \beta \sum_{i=0}^{m} \alpha_i g(x_{k-i})$

        Args:
            z_history (`List[torch.Tensor]`): History of input states [x_{k-m}, ..., x_k].
            f_history (`List[torch.Tensor]`): History of residuals [g(x_{k-m})-x_{k-m}, ..., f_k].
            beta (`float`): Mixing/Damping factor (beta in the formula).
            ridge (`float`): Tikhonov regularization factor for the least-squares problem.

        Returns:
            `torch.Tensor`: The accelerated next state $x_{k+1}$.
        """
        num_samples = len(f_history)
        m_formula = num_samples - 1 
        
        # Fallback to Picard iteration if insufficient history
        if m_formula < 1:
            return z_history[-1] + beta * f_history[-1]

        B = f_history[0].shape[0]
        device = f_history[0].device
        orig_dtype = f_history[0].dtype
        
        # 1. Flatten and cast to float32 for numerical stability in the linear solver
        z_flat = torch.stack([z.view(B, -1).to(torch.float32) for z in z_history], dim=1)
        f_flat = torch.stack([f.view(B, -1).to(torch.float32) for f in f_history], dim=1)
        g_flat = z_flat + f_flat

        # 2. Construct the Least Squares problem
        f_k = f_flat[:, -1:, :]
        Y = f_flat[:, -1:, :] - f_flat[:, :-1, :] # Delta F matrix
        
        input_matrix = Y.transpose(1, 2) # [B, D, m]
        target_matrix = f_k.transpose(1, 2) # [B, D, 1]

        if ridge > 0:
            eye = torch.eye(m_formula, device=device, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1) * ridge
            input_matrix = torch.cat([input_matrix, eye], dim=1)
            target_matrix = torch.cat([target_matrix, torch.zeros(B, m_formula, 1, device=device, dtype=torch.float32)], dim=1)

        # Solve for coefficients gamma
        gamma = torch.linalg.lstsq(input_matrix, target_matrix).solution # [B, m, 1]
        
        # 3. Map gamma to alpha weights (sum of alpha = 1)
        sum_gamma = gamma.sum(dim=1) 
        alpha_latest = 1.0 - sum_gamma
        alpha_history = gamma.squeeze(-1)
        alphas = torch.cat([alpha_history, alpha_latest], dim=1) # [B, num_samples]

        # 4. Compute weighted sums
        x_avg = torch.bmm(alphas.unsqueeze(1), z_flat).squeeze(1)
        g_avg = torch.bmm(alphas.unsqueeze(1), g_flat).squeeze(1)
        
        # 5. Final mixing
        z_next_flat = (1.0 - beta) * x_avg + beta * g_avg
        return z_next_flat.to(orig_dtype).view_as(z_history[0])

    def step(self, v_final: torch.Tensor, z_star: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Performs a standard Euler integration step.

        Args:
            v_final (`torch.Tensor`): The predicted velocity vector.
            z_star (`torch.Tensor`): The current state.
            dt (`float`): The time step size.

        Returns:
            `torch.Tensor`: The updated latents for the next timestep.
        """
        return z_star + v_final * dt



class CFGMPSD3Pipeline(StableDiffusion3Pipeline):
    """
    Inference pipeline for Stable Diffusion 3.5 utilizing CFG-MP/MP+.
    """
    
    def _get_velocity(self, latents: torch.Tensor, t_val: float, hidden_states: torch.Tensor, pooled: torch.Tensor, joint_kwargs: Dict) -> torch.Tensor:
        """Helper to invoke the Transformer model for velocity prediction."""
        t_tensor = torch.tensor([t_val * 1000.0], device=latents.device, dtype=latents.dtype)
        t_expand = t_tensor.expand(latents.shape[0])
        return self.transformer(
            hidden_states=latents,
            timestep=t_expand,
            encoder_hidden_states=hidden_states,
            pooled_projections=pooled,
            joint_attention_kwargs=joint_kwargs,
            return_dict=False,
        )[0]

    def iterate_fixed_point(
        self, 
        z_k: torch.Tensor, 
        t_curr: float, 
        t_mid: float, 
        p_embeds: torch.Tensor, 
        p_pooled: torch.Tensor, 
        n_embeds: torch.Tensor, 
        n_pooled: torch.Tensor, 
        joint_kwargs: Dict,
        z_history: List[torch.Tensor], 
        f_history: List[torch.Tensor],
        use_aa: bool = True,
        aa_window_size: int = 2,
        aa_damping: float = 1.0,
        aa_ridge: float = 1e-4,
        op_type: str = "G"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the manifold projection operator G and applies Anderson Mixing.

        Args:
            z_k (`torch.Tensor`): Current latent iteration.
            t_curr, t_mid (`float`): Current and midpoint timestamps.
            use_aa (`bool`): Whether to use Anderson Acceleration or standard Picard iteration.
            aa_window_size (`int`): The fixed window size $m$ for history.
            aa_damping (`float`): The beta parameter for mixing.
            op_type (`str`): The type of operator ('H' or 'G').

        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`: The next iteration state and the current residual.
        """
        dt_half = t_mid - t_curr

        # 1. Operator Evaluation: Compute G(z_k)
        if op_type == "H":
            v_cond = self._get_velocity(z_k, t_curr, p_embeds, p_pooled, joint_kwargs)
            z_temp = z_k + v_cond * dt_half
            v_uncond = self._get_velocity(z_temp, t_curr, n_embeds, n_pooled, joint_kwargs)
            g_z_k = z_temp - v_uncond * dt_half
        else: # "G"
            v_uncond = self._get_velocity(z_k, t_curr, n_embeds, n_pooled, joint_kwargs)
            z_temp = z_k - v_uncond * dt_half
            v_cond = self._get_velocity(z_temp, t_curr, p_embeds, p_pooled, joint_kwargs)
            g_z_k = z_temp + v_cond * dt_half

        f_k = g_z_k - z_k # Residual

        # 2. Picard Route
        if not use_aa:
            return g_z_k, f_k

        # 3. Anderson Route
        z_history.append(z_k)
        f_history.append(f_k)

        if len(f_history) > (aa_window_size + 1):
            z_history.pop(0)
            f_history.pop(0)

        z_next = self.scheduler.solve_anderson_mixing(
            z_history, f_history, beta=aa_damping, ridge=aa_ridge
        )
        return z_next, f_k

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 4.5,
        use_aa: bool = True,
        max_aa_iter: int = 3,
        aa_window_size: int = 2,
        aa_tol: float = 1e-6,
        aa_damping: float = 1.0,
        aa_ridge: float = 1e-4,
        switching_threshold: float = 0.5, 
        **kwargs,
    ):
        """
        Inference call for image generation.
        """
        device = self._execution_device
        
        # 1. Encode prompt
        (p_embeds, n_embeds, p_pooled, n_pooled) = self.encode_prompt(
            prompt=prompt, prompt_2=prompt, prompt_3=prompt, device=device, do_classifier_free_guidance=True
        )

        # 2. Prepare latents
        latents = self.prepare_latents(
            len(prompt) if isinstance(prompt, list) else 1, 
            self.transformer.config.in_channels, kwargs.get("height", 1024), kwargs.get("width", 1024), 
            p_embeds.dtype, device, kwargs.get("generator"), None
        )
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        # 3. Sampling Loop 
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i in range(num_inference_steps):
                t_curr, t_mid, t_next = self.scheduler.get_fine_grained_times(i)
                dt_full = t_next - t_curr
                
                # 1. Manifold projection Phase 
                if t_curr <= switching_threshold:
                    z_star = latents
                else:
                    z_k = latents.clone()
                    z_history, f_history = [], []
                    op_type = "H" if t_curr >= 0.95 else "G"
                    
                    for k in range(max_aa_iter):
                        z_k, f_k = self.iterate_fixed_point(
                            z_k, t_curr, t_mid, p_embeds, p_pooled, n_embeds, n_pooled, 
                            kwargs.get("joint_attention_kwargs"), z_history, f_history,
                            use_aa=use_aa, aa_window_size=aa_window_size,
                            aa_damping=aa_damping, aa_ridge=aa_ridge, op_type=op_type
                        )
                        if f_k.abs().mean() < aa_tol:
                            break
                    z_star = z_k

                # 2. Sampling Phase 
                v_cond = self._get_velocity(z_star, t_curr, p_embeds, p_pooled, kwargs.get("joint_attention_kwargs"))
                v_uncond = self._get_velocity(z_star, t_curr, n_embeds, n_pooled, kwargs.get("joint_attention_kwargs"))
                v_final = v_uncond + guidance_scale * (v_cond - v_uncond)
                
                latents = self.scheduler.step(v_final, z_star, dt_full)
                progress_bar.update()

        # 4. Final Output
        return self.image_processor.postprocess(self.vae.decode(latents / self.vae.config.scaling_factor + self.vae.config.shift_factor,
                                                return_dict=False)[0],
                                                output_type=kwargs.get("output_type", "pil"))
