import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GTAlign(nn.Module):

    def __init__(self, model, temperature=0.07, beta_init=0.2,
                 lambda_init=1.0, mu_init=0.5, alpha=0.5,
                 momentum=0.9, filter_ratio_init=0.8, align_weight_init=1.0,
                 dataset_name=None, theta_init=0.5):
        super(GTAlign, self).__init__()
        self.model = model
        self.dataset_name = dataset_name
        self.temperature=temperature
        self.beta_init = beta_init
        self.lambda_init = lambda_init
        self.mu_init = mu_init
        self.align_weight_init = align_weight_init
        self.filter_ratio_init = filter_ratio_init  
        self.theta_init = theta_init 

        self.alpha = alpha
        self.momentum = momentum

        self.register_buffer('M0', torch.tensor(0.0))
        self.register_buffer('sigma', torch.tensor(1.0))
        self.register_buffer('n_updates', torch.tensor(0))

    def contrastive_loss(self, graph_embeddings, text_embeddings, precomputed_sim=None):
        if precomputed_sim is None:
            
            graph_embeddings = F.normalize(graph_embeddings, p=2, dim=1)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

            logits = torch.matmul(graph_embeddings, text_embeddings.t()) / self.temperature
        else:
            logits = precomputed_sim

        labels = torch.arange(logits.size(0), device=logits.device)

        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2

        return loss, logits

    def calculate_similarities(self, graph_embeddings, text_embeddings):

        graph_embeddings = F.normalize(graph_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

        sim_matrix = torch.matmul(graph_embeddings, text_embeddings.t()) / self.temperature

        batch_size = graph_embeddings.size(0)

        pos_sim = torch.diag(sim_matrix)

        all_neg_sim = []
        for i in range(batch_size):
            neg_indices = torch.ones(batch_size, dtype=torch.bool, device=sim_matrix.device)
            neg_indices[i] = False
            neg_sim = sim_matrix[i][neg_indices]
            all_neg_sim.append(neg_sim)

        return pos_sim, all_neg_sim, sim_matrix

    def calculate_reward_differences(self, pos_sim, all_neg_sim):
        Mi_values = []

        is_instagram = self.dataset_name == 'instagram'

        for i in range(len(all_neg_sim)):
            if is_instagram and (torch.isnan(all_neg_sim[i]).any() or torch.isinf(all_neg_sim[i]).any()):
                valid_neg_sim = all_neg_sim[i][~torch.isnan(all_neg_sim[i]) & ~torch.isinf(all_neg_sim[i])]
                if len(valid_neg_sim) > 0:
                    mean_neg_sim = torch.mean(valid_neg_sim)
                else:
                    mean_neg_sim = torch.tensor(0.0, device=pos_sim.device)
            else:
                mean_neg_sim = torch.mean(all_neg_sim[i])

            Mi = pos_sim[i] - mean_neg_sim  
            if is_instagram:
                Mi = torch.clamp(Mi, min=-5.0, max=5.0)

            Mi_values.append(Mi)

        Mi_tensor = torch.stack(Mi_values)

        if is_instagram:
            M_batch = torch.median(Mi_tensor)
        else:
            M_batch = torch.mean(Mi_tensor) 

        return Mi_tensor, M_batch

    def update_moving_averages(self, M_batch, Mi_tensor):
        if not self.training:
            return

        with torch.no_grad():
            try:
                self.n_updates += 1

                if self.n_updates == 1:
                    self.M0 = M_batch.detach()
                    std = torch.std(Mi_tensor.detach())
                    if torch.isnan(std) or std < 1e-8:
                        std = torch.tensor(1.0, device=Mi_tensor.device)
                    self.sigma = std
                else:
                    self.M0 = self.momentum * self.M0 + (1 - self.momentum) * M_batch.detach()
                    batch_var = torch.var(Mi_tensor.detach())
                    if torch.isnan(batch_var) or batch_var < 1e-8:
                        batch_std = torch.tensor(1.0, device=Mi_tensor.device)
                    else:
                        batch_std = torch.sqrt(batch_var)

                    self.sigma = self.momentum * self.sigma + (1 - self.momentum) * batch_std
            except Exception as e:
                print(f"{e}")

    def calculate_dynamic_weights(self, M_batch, fixed_theta=None, disable_dynamic=False):
        try:
            if torch.isnan(M_batch):
                return (
                    torch.tensor(self.beta_init, device=M_batch.device),
                    torch.tensor(0.0, device=M_batch.device), 
                    torch.tensor(self.mu_init, device=M_batch.device),
                    torch.tensor(self.align_weight_init, device=M_batch.device),
                    torch.tensor(self.filter_ratio_init, device=M_batch.device),
                    torch.tensor(0.0, device=M_batch.device),
                    torch.tensor(0.5, device=M_batch.device)
                )

            adjustment_factor = M_batch - self.M0 # Raw difference for potential logging

            if fixed_theta is not None:
                theta = torch.tensor(fixed_theta, device=M_batch.device)
                if not hasattr(self, '_printed_fixed_theta'):
                    self._printed_fixed_theta = True
            elif disable_dynamic:
                theta = torch.tensor(0.5, device=M_batch.device)
                if not hasattr(self, '_printed_disable_dynamic'):
                    self._printed_disable_dynamic = True
            else:
                theta_raw = self.theta_init + self.alpha * adjustment_factor 
                theta = torch.clamp(input=theta_raw, min=0.1, max=2.0) 

            beta = theta * self.beta_init  
            align_weight = theta * self.beta_init

            mu = 1 / (1 + theta) * self.mu_init 

            lambda_val = torch.tensor(0.0, device=M_batch.device)

            filter_ratio = self.filter_ratio_init + 0.5 * (theta - self.theta_init) 

            if not hasattr(self, '_printed_fixed_filter_ratio'):
                self._printed_fixed_filter_ratio = True

            beta = torch.clamp(input=beta, min=0.05, max=0.9)
            mu = torch.clamp(input=mu, min=0.1, max=0.9)
            align_weight = torch.clamp(input=align_weight, min=0.1, max=2.0)
            filter_ratio = torch.clamp(input=filter_ratio, min=0.3, max=0.99)

            if (torch.isnan(beta) or torch.isnan(lambda_val) or torch.isnan(mu) or
                torch.isnan(align_weight) or torch.isnan(filter_ratio) or torch.isnan(theta)):
                return (
                    torch.tensor(self.beta_init, device=M_batch.device),
                    torch.tensor(0.0, device=M_batch.device),
                    torch.tensor(self.mu_init, device=M_batch.device),
                    torch.tensor(self.align_weight_init, device=M_batch.device),
                    torch.tensor(self.filter_ratio_init, device=M_batch.device),
                    torch.tensor(0.0, device=M_batch.device),
                    torch.tensor(0.5, device=M_batch.device)
                )
        except Exception as e:
            return (
                torch.tensor(self.beta_init, device=M_batch.device),
                torch.tensor(0.0, device=M_batch.device), 
                torch.tensor(self.mu_init, device=M_batch.device),
                torch.tensor(self.align_weight_init, device=M_batch.device),
                torch.tensor(self.filter_ratio_init, device=M_batch.device),
                torch.tensor(0.0, device=M_batch.device),
                torch.tensor(0.5, device=M_batch.device)
            )
        return beta, lambda_val, mu, align_weight, filter_ratio, adjustment_factor, theta

    def calculate_sample_weights(self, Mi_tensor):
        try:
            sigma = self.sigma.clone()
            if torch.isnan(sigma) or sigma < 1e-8:
                sigma = torch.tensor(1.0, device=Mi_tensor.device)

            p_Mi = torch.exp(-0.5 * ((Mi_tensor - self.M0) / sigma)**2)
            p_Mi = p_Mi / (sigma * torch.sqrt(torch.tensor(2 * np.pi)))

            if torch.isnan(p_Mi).any():
                p_Mi = torch.ones_like(Mi_tensor) / len(Mi_tensor)
        except Exception as e:
            p_Mi = torch.ones_like(Mi_tensor) / len(Mi_tensor)

        return p_Mi

    def filter_samples(self, p_Mi, batch_size, filter_ratio):
        try:
            if torch.isnan(filter_ratio):
                ratio_value = 0.8
            else:
                ratio_value = filter_ratio.item()

            n_keep = max(1, int(batch_size * ratio_value))
        except Exception as e:
            n_keep = max(1, int(batch_size * 0.8))

        _, indices = torch.sort(p_Mi, descending=True)
        keep_indices = indices[:n_keep]

        mask = torch.zeros(batch_size, dtype=torch.bool, device=p_Mi.device)
        if len(keep_indices) > 0:  # Ensure keep_indices is not empty
            mask[keep_indices] = True
        return mask

    def compute_soft_targets(self, hard_labels, similarity_matrix, beta=None):
        if beta is None:
            beta = self.beta_init

        batch_size = similarity_matrix.size(0)

        one_hot_labels = F.one_hot(hard_labels, num_classes=batch_size).float()

        similarity_distribution = F.softmax(similarity_matrix, dim=1)

        soft_targets = (1 - beta) * one_hot_labels + beta * similarity_distribution

        return soft_targets

    def compute_decoupled_negatives(self, distribution):
        batch_size = distribution.size(0)

        eye_mask = ~torch.eye(batch_size, device=distribution.device, dtype=torch.bool)

        neg_distribution = distribution * eye_mask.float()

        row_sums = neg_distribution.sum(dim=1, keepdim=True)
        norm_neg_distribution = neg_distribution / (row_sums + 1e-8)

        return norm_neg_distribution

    def symmetric_kl_divergence(self, p, q):
        p = torch.clamp(p, min=1e-8)
        q = torch.clamp(q, min=1e-8)

        kl_p_q = torch.sum(p * torch.log(p / q), dim=1)
        kl_q_p = torch.sum(q * torch.log(q / p), dim=1)

        symmetric_kl = 0.5 * (kl_p_q + kl_q_p)

        return symmetric_kl.mean()

    def roi_tags_alignment_loss(self, roi_embeddings, tags_embeddings):
        roi_embeddings_norm = F.normalize(roi_embeddings, p=2, dim=1)
        tags_embeddings_norm = F.normalize(tags_embeddings, p=2, dim=1)

        sim_matrix = torch.matmul(roi_embeddings_norm, tags_embeddings_norm.t()) / self.temperature

        labels = torch.arange(roi_embeddings.size(0), device=roi_embeddings.device)

        r2t_loss = F.cross_entropy(sim_matrix, labels)

        t2r_loss = F.cross_entropy(sim_matrix.t(), labels)

        align_loss = 0.5 * (r2t_loss + t2r_loss)

        return align_loss

    def combined_loss(self, graph_embeddings, roi_embeddings, text_embeddings):
        pos_sim, all_neg_sim, sim_matrix = self.calculate_similarities(graph_embeddings, text_embeddings)
        Mi_tensor, M_batch = self.calculate_reward_differences(pos_sim, all_neg_sim)

        self.update_moving_averages(M_batch, Mi_tensor)

        beta, lambda_val, mu, align_weight, dynamic_filter_ratio, adjustment_factor, theta = self.calculate_dynamic_weights(
            M_batch,
            fixed_theta=self.fixed_theta if hasattr(self, 'fixed_theta') else None,
            disable_dynamic=self.disable_dynamic if hasattr(self, 'disable_dynamic') else False
        )

        p_Mi = self.calculate_sample_weights(Mi_tensor)
        mask = self.filter_samples(p_Mi, graph_embeddings.size(0), dynamic_filter_ratio)

        if mask.sum() == 0:
            mask = torch.ones_like(mask, dtype=torch.bool)
            current_filter_ratio = 1.0 
        else:
            current_filter_ratio = mask.float().mean().item() 

        filtered_graph_emb = graph_embeddings[mask]
        filtered_roi_emb = roi_embeddings[mask]
        filtered_text_emb = text_embeddings[mask]

        if filtered_graph_emb.size(0) == 0:
            zero_loss = torch.tensor(0.0, device=graph_embeddings.device, requires_grad=True)
            return {
                'total_loss': zero_loss,
                'soft_loss': zero_loss.detach(),
                'neg_enhanced_loss': zero_loss.detach(),
                'clip_loss': zero_loss.detach(),
                'align_loss': zero_loss.detach(),
                'beta': beta.detach().item(),
                'lambda': lambda_val.detach().item(), 
                'mu': mu.detach().item(),
                'align_weight': align_weight.detach().item(),
                'filter_ratio': dynamic_filter_ratio.detach().item(),
                'M': M_batch.detach().item(),
                'actual_filter_ratio': current_filter_ratio,
                'adjustment_factor': adjustment_factor.detach().item(),
                'samples_kept': 0,
                'theta': theta.detach().item()
            }

        graph_text_sim = torch.matmul(filtered_graph_emb, filtered_text_emb.t()) / self.temperature
        roi_roi_sim = torch.matmul(filtered_roi_emb, filtered_roi_emb.t()) / self.temperature
        tags_tags_sim = torch.matmul(filtered_text_emb, filtered_text_emb.t()) / self.temperature
        text_graph_sim = graph_text_sim.t()

        labels = torch.arange(filtered_graph_emb.size(0), device=graph_embeddings.device)
        roi_roi_soft = self.compute_soft_targets(labels, roi_roi_sim, beta)
        tags_tags_soft = self.compute_soft_targets(labels, tags_tags_sim, beta)

        g2t_soft_loss = self.symmetric_kl_divergence(roi_roi_soft, F.softmax(graph_text_sim, dim=1))
        t2g_soft_loss = self.symmetric_kl_divergence(tags_tags_soft, F.softmax(text_graph_sim, dim=1))
        soft_loss = 0.5 * (g2t_soft_loss + t2g_soft_loss)

        clip_loss = (F.cross_entropy(graph_text_sim, labels) + F.cross_entropy(text_graph_sim, labels)) / 2

        align_loss = self.roi_tags_alignment_loss(filtered_roi_emb, filtered_text_emb)

        total_loss = soft_loss + mu * clip_loss + align_weight * align_loss

        neg_enhanced_loss = torch.tensor(0.0, device=graph_embeddings.device)

        return {
            'total_loss': total_loss,
            'soft_loss': soft_loss.detach(),
            'neg_enhanced_loss': neg_enhanced_loss.detach(), 
            'clip_loss': clip_loss.detach(),
            'align_loss': align_loss.detach(),
            'beta': beta.detach().item(),
            'lambda': lambda_val.detach().item(),
            'mu': mu.detach().item(),
            'align_weight': align_weight.detach().item(),
            'filter_ratio': dynamic_filter_ratio.detach().item(),
            'M': M_batch.detach().item(),
            'actual_filter_ratio': current_filter_ratio,
            'adjustment_factor': adjustment_factor.detach().item(),
            'samples_kept': mask.sum().item(),
            'theta': theta.detach().item()
        }
