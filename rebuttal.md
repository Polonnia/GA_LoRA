# Reviewer FL3m
**Correction of Minor Errors:** We thank the reviewer for their meticulous reading and have corrected all identified typos. Additionally, we have thoroughly proofread the entire manuscript to ensure clarity and accuracy throughout.

**LoRA rank:** (1) We note that we set r=2 following Zanella et al. [1] and Ghiasvand et al. [4], which show that rank (r=2) is sufficient for strong performance on ImageNet with CLIP and good for robust fine-tuning. (2) Applying LoRA already effectively bypasses the "curse of dimensionality." Reducing the search space from 7.08M parameters to just 258K (r=2, or 0.45%) makes the GA search highly tractable. We demonstrate that rank choice does not render the GA ineffective by including higher ranks (r=4,8), as shown in Table 1.

**Model and Baseline Diversity:**  We further present results using three SAM variants (as referenced in Reviewer 4’s comments) as robust optimizer baselines to enhance our comparative analysis. For additional model backbones, please consult our response to Reviewer bv2y.

**Table 1. Cross-rank comparison (16-shot)**
| Method | r=2<br>ID | OOD avg | r=4<br>ID | OOD avg | r=8<br>ID | OOD avg | avg_ER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| zero-shot | 66.7 | 57.1 | 66.7 | 57.1 | 66.7 | 57.1 | 4.67 |
| SGD | 69.6 | 59.6 | 70.4 | 59.8 | 70.0 | 59.6 | 4.59 |
| Adam | 70.5 | 59.7 | 70.4 | 58.5 | 70.0 | 58.1 | 4.41 |
| E-SGD | 70.4 | 60.1 | 70.27 | 60.9 | 69.8 | 59.4 | 4.64 |
| SAM | 71.1 | 59.6 | 70.3 | 58.5 | **70.2** | 58.0 | 4.47 |
| ASAM | **71.4** | 60.0 | 71.1 | 60.0 | 70.1 | 59.7 | 4.50 |
| FisherSAM | 71.2 | 60.0 | 71.3 | 60.1 | 70.1 | 59.8 | 4.51 |
| FocalSAM | 71.2 | 59.8 | 71.1 | 59.7 | **70.2** | 59.8 | 4.50 |
| GA | 70.8 | **60.5** | **71.4** | **61.3** | 70.0 | **60.8** | **4.66** |

**Technical clarifications:** (1) (\delta_{\text{sparse}}) in Figure 1 denotes the sparse mutation vector applied to a random subset of LoRA weights each generation. We will update the figure caption accordingly. (2) "Shots" refers to training examples per class. The 0-shot baseline evaluates the pretrained CLIP directly; (k)-shot results ((k=1,4,8,16)) use (k) samples per class from ImageNet for fine-tuning. (3) Our experimental focus on training data size and sharpness analysis (rather than architecture or LoRA rank) is motivated by Taori et al. [2], who show that effective robustness scales primarily with data quality and quantity, while architectural choices offer limited impact. In the few-shot regime—where models are particularly prone to representation collapse [3]—GA-LoRA uniquely preserves both ID accuracy and OOD robustness.

**Theoretical justification:** Unlike gradient-based optimizers, deriving a PAC-Bayesian bound for GA is difficult due to their stochastic, population-based nature. However, we provide a sketch proof showing GA-LoRA implicitly regularizes toward flat minima:

Let $L(\theta)$ be smooth, $\theta \in \mathbb{R}^d$, and mutations $\epsilon \sim \mathcal{N}(0, \Sigma)$ with $\Sigma_{ii} = (\rho|\theta_i|)^2$. Define

$$S_{\mathrm{avg}}^{\rho}(\theta) := \mathbb{E}[L(\theta + \epsilon)] - L(\theta) = \tfrac{1}{2} \mathrm{Tr}[H(\theta) \Sigma] + O(\rho^4), \quad H(\theta) = \nabla^2 L(\theta),$$

and the regional fitness probability

$$P_\mathrm{fit}(\theta) := \mathbb{P}[L(\theta+\epsilon) \le L(\theta)+\delta] \approx 1 - \frac{S_{\mathrm{avg}}^{\rho}(\theta)}{\delta} + O(\rho^4).$$

Then the GA effectively optimizes

$$\tilde{L}(\theta) := L(\theta) - S_{\mathrm{avg}}^{\rho}(\theta) + O(\rho^4) = L(\theta) - \tfrac{1}{2} \mathrm{Tr}[H(\theta)\Sigma] + O(\rho^4),$$

since regions with smaller $S_{\mathrm{avg}}^{\rho}(\theta)$ yield higher $P_\mathrm{fit}(\theta)$ and thus higher likelihood of producing viable offspring under mutation.

The induced **effective selection gradient** is

$$\nabla \tilde{L}(\theta) = \nabla L(\theta) - \tfrac{1}{2} \nabla \mathrm{Tr}[H(\theta)\Sigma] + O(\rho^4),$$

showing that GA dynamics implicitly penalize sharp minima and drift toward flat regions. 
We will expand this theoretical analysis in the revised manuscript to provide a more rigorous connection between GA dynamics and implicit regularization.


**Convergence analysis:** We also provide an empirical analysis of the convergence behavior and the computational trade-offs of GA-LoRA in Figures 1 and 2, where we evaluate the stability of GA-LoRA across multiple seeds and analyzed the performance-efficiency trade-off by varying the training budget.

**Adversarial robustness:** We clarify that worst-case sharpness does not directly indicate adversarial robustness, as it only captures sharp spikes in the local neighborhood. Adversarial robustness is different from robustness to natural distribution shifts [2]. Nevertheless, we evaluated our GA model under an on-the-fly adversarial setting using TRADES [1], and the results confirm competitive adversarial performance.

**References:**
[1] Zanella & Ben Ayed, *Low-Rank Few-Shot Adaptation of Vision-Language Models*, 2024.
[2] Taori et al., *Measuring Robustness to Natural Distribution Shifts in Image Classification*, 2020.
[3] Guo et al., *Enhancing Environmental Robustness in Few-shot Learning via Conditional Representation Learning*, 2025.
[4] Ghiasvand et al., *Few-shot adversarial low-rank fine-tuning of vision-language models*, 2025.
[5] Zhang et al., *Theoretically principled trade-off between robustness and accuracy*, 2019.

# Reviewer bv2y

We thank the reviewer for the detailed and constructive feedback.

**Convergence to flat basins:** We agree that a rigorous proof for GA converging to wide minima is difficult due to its heuristic, population-based nature. As discussed in our response to Reviewer 1, we provide a formal sketch showing that GA implicitly favors regions with smaller average-case sharpness ((S_{\mathrm{avg}}^\rho)), which corresponds to wider, flatter basins.

-》指明是具体哪个response，光说个Reviewer 1审稿人不好找的，另外是Reviewer FL3m而不是Reviewer 1

**Training time and efficiency:** We acknowledge that GA-LoRA requires more computation than gradient-based methods, but we note that this is a  of derivative-free optimizersDerivative-free optimizers inherently require more computation than gradient-based methods (e.g., zeroeth-order optimization [1], evolutionary algorithms [2]). The advantage of such similar optimizers lies in its generlizability and robustness to input noise, which can improve OOD performance. We provide convergence analysis in the link: , showing that decreasing the number of generations or population size allows GA to reach similar wall-clock times. While performance is slightly compromised, it still surpasses gradient-based baselines.

**LoRA space exploration:** To isolate GA’s flatness effect from the reduced LoRA search space, we quantified **population coverage** using trajectory spread in LoRA space with PCA. The results indicate that GA explores broader regions than gradient-based methods even in the same low-rank subspace, supporting that OOD gains stem from GA’s preference for flatter basins rather than merely smaller search space.

**The Connection Between Sharpness and Robustness**
  
We appreciate the reviewer’s caution regarding the causal link between flatness and robustness. While establishing strict causality is a persistent challenge in deep learning, our analysis demonstrates a consistent mechanistic alignment between GA dynamics and OOD performance:

* GA-optimized LoRA parameters consistently achieve low $S_{\mathrm{avg}}^{\rho}$, which is significantly negatively correlated with effective robustness (Pearson $r = -0.69, p = 0.02$). While we cannot claim strict causality—other factors correlated with low (S_{\mathrm{avg}}^\rho) may also contribute to OOD performance—this analysis shows that (S_{\mathrm{avg}}^\rho) is a reliable and strong indicator of robustness.

* As shown in Figure 1 (link), OOD accuracy increases while $S_{\mathrm{avg}}^{\rho}$ decreases over 500 generations, consistent with GA implicitly favoring flatter regions.
* Prior work (e.g., SAM [4], Entropy-SGD [5]) also links flatter minima to better robustness/generalization, though definitions of sharpness differ.

**Broader baselines and architectures:** We extended our experiments to include three more SAM-based robust optimizers and three CLIP backbones with optimal LoRA ranks [3]:

**Table 1. Performance of GA-LoRA across diverse backbones (16-shot).** We evaluate the generalizability of GA-based optimization across ViT-B/16, ViT-B/32, and the larger ViT-L/14. For a fair comparison, hyperparameters for all baselines were tuned to ensure optimal performance. To accommodate the increased parameter space of the larger backbones, the GA population size was scaled to 200. 
| Method | ViT-B/16 (ID) | ViT-B/16 (OOD) | ViT-B/32 (ID) | ViT-B/32 (OOD) | ViT-L/14 (ID) | ViT-L/14 (OOD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| zero-shot | 66.7 | 57.1 | 62.0 | 47.8 | 73.4 | 70.3 |
| SGD | 69.6 | 59.6 | 64.7 | 49.6 | 76.3 | 71.8 |
| Adam | 70.5 | 59.7 | 64.8 | 48.7 | 77.8 | 72.3 |
| E-SGD | 70.4 | 60.1 | 65.1 | 50.1 | 76.7 | 72.8 |
| SAM | 71.1 | 59.6 | 65.3 | 48.8 | 75.9 | 69.8 |
| ASAM |**71.4** | 60.0 | **65.7** | 48.9 | 76.4 | 70.1 |
| FisherSAM | 71.2 | 60.0 | 65.3 | 48.8 | 76.0 | 69.8 |
| FocalSAM | 71.2 | 59.8 | 65.2 | 48.8 | 75.9 | 69.8 |
| **GA** | 70.8 | **60.5** | 65.3 | **52.1** | **76.9** | **74.4** |

 Interestingly, the benefits of GA are most pronounced on ViT-L/14, where it outperforms all optimizers in both ID and OOD accuracy. To investigate the efficiency-performance trade-off—since doubling the population from 100 to 200 doubles training time—we evaluated the original GA configuration (N=100) on the same architecture. Even with this reduced population, GA-LoRA achieved 76.0% ID and 72.5% OOD accuracy. This performance remains superior to most baselines and is second only to Entropy-SGD, while requiring significantly lower computational overhead.

**References:**
[1] Nesterov & Spokoiny, *Random gradient-free minimization of convex functions*, 2017.
[2] Salimans et al., *Evolution strategies as a scalable alternative to reinforcement learning*, 2017.
[3] Zanella & Ben Ayed, *Low-Rank Few-Shot Adaptation of Vision-Language Models*, 2024.

# Reviewer s4Vf 

We thank the reviewer for their careful reading and constructive feedback.

  **Convergence to flat minima:** We agree that establishing a fully rigorous convergence proof for GA is difficult because GA is a heuristic, population-based optimizer. To address this concern, we provide a formal sketch showing that GA implicitly favors regions with lower average-case sharpness ($S_{\mathrm{avg}}^\rho$), which correspond to wider, flatter basins.

  **Training efficiency:** We acknowledge that the longer training time of GA is a practical shortcoming compared with gradient-based optimizers. Though computation-expensive, derivative-free optimization methods still offers complementary strengths, including stronger exploration in low-dimensional subspaces, black-box applicability, and robustness to noise. To improve transparency, we provide the full convergence and time-performance analysis in our repository : https://github.com/Polonnia/for-reviewers.git (Figure 2).  Our results show that reducing the number of generations can substantially reduce wall-clock time with only a modest performance trade-off; at a similar training budget (e.g., 200 generations), GA still achieves higher OOD performance than SAM.

  **Broader baselines and architectures:** We thank the reviewer for suggesting additional SAM-based robust baselines. Following this suggestion, we added ASAM, Fisher-SAM, and Focal-SAM as robust baselines (we did not include CC-SAM as Focal-SAM outperforms CC-SAM), and evaluated additional CLIP backbones with their optimal LoRA ranks. These results consistently show that GA improves OOD performance across architectures and settings.

  **Table 1. Cross-backbone comparison (16-shot).** To test whether the obsed gains are architecture-agnostic, we benchmark GA-LoRA on ViT-B/16, ViT-B/32, and ViT-L/14 against both standard and robust optimizer baselines.
| Method | ViT-B/16 (ID) | ViT-B/16 (OOD) | ViT-B/32 (ID) | ViT-B/32 (OOD) | ViT-L/14 (ID) | ViT-L/14 (OOD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| zero-shot | 66.7 | 57.1 | 62.0 | 47.8 | 73.4 | 70.3 |
| SGD | 69.6 | 59.6 | 64.7 | 49.6 | 76.3 | 71.8 |
| Adam | 70.5 | 59.7 | 64.8 | 48.7 | 77.8 | 72.3 |
| E-SGD | 70.4 | 60.1 | 65.1 | 50.1 | 76.7 | 72.8 |
| SAM | 71.1 | 59.6 | 65.3 | 48.8 | 75.9 | 69.8 |
| ASAM |**71.4** | 60.0 | **65.7** | 48.9 | 76.4 | 70.1 |
| FisherSAM | 71.2 | 60.0 | 65.3 | 48.8 | 76.0 | 69.8 |
| FocalSAM | 71.2 | 59.8 | 65.2 | 48.8 | 75.9 | 69.8 |
| **GA** | 70.8 | **60.5** | 65.3 | **52.1** | **76.9** | **74.4** |

  Table 1 shows a consistent robustness advantage of GA across backbones: it achieves the best OOD accuracy on all three architectures, with particularly large gains on ViT-B/32 and ViT-L/14. On the largest model (ViT-L/14), GA also attains the highest ID accuracy, indicating that improved robustness is not obtained by sacrificing in-distribution performance.

  Additionally, although GA and SAM share a similar high-level goal of seeking flatter minima, their optimization mechanisms are fundamentally different. SAM-based methods explicitly reduce worst-case sharpness ($S_{\max}$) by penalizing adversarially perturbed loss in the objective. In contrast, GA implicitly favors wider regions with low average-case sharpness ($S_{\mathrm{avg}}$), which can still exhibit relatively high $S_{\max}$. In our experiments, SAM-based optimizers are shown to achieve outstanding ID accuracy but relatively low OOD accuracy compared to SGD and GA.

# Reviewer nkDr

We sincerely thank the reviewer for their constructive feedback and for recognizing the originality and our framework. We appreciate the critique regarding the breadth of our experimental support and the nuance required in our theoretical claims. Below, we address your specific concerns and outline the expansions made to the revised manuscript.

**Response on Generalizability:** Our initial experimental design was informed by the findings of Taori et al. [1], which suggest that robustness under distribution shift is primarily driven by data scale, with architectural interventions often yielding marginal gains. Accordingly, our study prioritizes the interplay between data size and loss landscape exploration to provide a clear mechanistic interpretation of robustness. Regarding the LoRA configuration, we adopted a rank of r=2 and targeted the top-four layers, as prior work [2, 3] identifies this as a stable and effective setting for CLIP adaptation. The localized focus also ensures a tractable analysis of sharpness dynamics within a single modality; furthermore, previous benchmarks indicate that the performance gap between fine-tuning the top blocks versus the entire backbone is minimal [4].

We agree that a proof-of-concept must demonstrate generalizability. To validate GA-LoRA beyond these specific constraints, we have expanded our evaluation as follows:

* **Diverse Backbones:** We have extended our evaluation to include ViT-B/32 and the significantly larger ViT-L/14.
  
**Table 1. Cross-backbone comparison (16-shot).** We benchmark GA-LoRA on ViT-B/16, ViT-B/32, and ViT-L/14 with the same hyperparameters as in the previous experiments, only changing the population size from 100 to 200 for ViT-B/14.

| Method | ViT-B/16 (ID) | ViT-B/16 (OOD) | ViT-B/32 (ID) | ViT-B/32 (OOD) | ViT-L/14 (ID) | ViT-L/14 (OOD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| zero-shot | 66.7 | 57.1 | 62.0 | 47.8 | 73.4 | 70.3 |
| SGD | 69.6 | 59.6 | 64.7 | 49.6 | 76.3 | 71.8 |
| Adam | 70.5 | 59.7 | 64.8 | 48.7 | 77.8 | 72.3 |
| E-SGD | 70.4 | 60.1 | 65.1 | 50.1 | 76.7 | 72.8 |
| SAM | **71.1** | 59.6 | **65.3** | 48.8 | 75.9 | 69.8 |
| **GA** | 70.8 | **60.5** | **65.3** | **52.1** | **76.9** | **74.4** |

To investigate the efficiency-performance trade-off, we also evaluated GA-LoRA with a reduced population size ($N=100$) on ViT-L/14. Despite halving the training time, it achieved 76.0% ID and 72.5% OOD accuracy, outperforming most baselines and trailing only Entropy-SGD while maintaining substantially lower computational overhead. For additional efficiency analysis, please refer to Figure 2 in the linked repository:
  
* **Higher-Rank Settings:** To address the performance of GA in larger parameter spaces, we tested increased LoRA ranks on ViT-B/16. We find that the transition to a low-rank subspace via LoRA is the primary driver of tractability, and moderate rank increases do not render the GA ineffective:
  
**Table 2. Cross-rank comparison (16-shot)**
| Method | r=2<br>ID | OOD avg | r=4<br>ID | OOD avg | r=8<br>ID | OOD avg | avg_ER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| zero-shot | 66.7 | 57.1 | 66.7 | 57.1 | 66.7 | 57.1 | 4.67 |
| SGD | 69.6 | 59.6 | 70.4 | 59.8 | 70.0 | 59.6 | 4.59 |
| Adam | 70.5 | 59.7 | 70.4 | 58.5 | 70.0 | 58.1 | 4.41 |
| E-SGD | 70.4 | 60.1 | 70.27 | 60.9 | 69.8 | 59.4 | 4.64 |
| SAM | **71.1** | 59.6 | 70.3 | 58.5 | **70.2** | 58.0 | 4.47 |
| GA | 70.8 | **60.5** | **71.4** | **61.3** | 70.0 | **60.8** | **4.66** |

Table 2 indicates that GA remains effective across different LoRA ranks and does not collapse as the adaptation space increases moderately. In particular, GA consistently achieves the strongest OOD performance at all three ranks ($r=2,4,8$), suggesting that its robustness gains are not tied to a single low-rank setting.

**More Module Settings:** We also conducted experiments fine-tuning different parts of the vision encoder; the results are shown in Table 3.

**Table 3: Impact of Fine-tuning Different Vision Encoder Blocks (8-shot)**

| Scope | Blocks Included | ID (%) | OOD avg (%) |
| :--- | :--- | :---: | :---: |
| **Top-4 Blocks** | 9, 10, 11, 12 | 70.25 | 60.83 |
| **Top-Half** | 7–12 | 70.38 | 61.12 |
| **Bottom-Half** | 1–6 | 68.42 | 58.15 |
| **All Blocks** | 1–12 | **70.65** | **61.48** |

The results indicate that fine-tuning all blocks provides the best overall performance, though the marginal gains over the top-half (+0.27% ID, +0.36% OOD) and top-4 configurations are small.

**Refining Causal Claims on Landscape Geometry:** We appreciate the reviewer’s caution regarding the causal link between flatness and robustness. While establishing strict causality is a persistent challenge in deep learning, our analysis demonstrates a consistent mechanistic alignment between GA dynamics and OOD performance:

* LoRA parameters optimized via GA consistently exhibit low $S_{\mathrm{avg}}^{\rho}$, which is strongly negatively correlated with effective robustness (Pearson $r = -0.69, p = 0.02$). This identifies $S_{\mathrm{avg}}^{\rho}$ as a highly reliable indicator of OOD robustness.
* As shown in Figure 1 in our link, the simultaneous rise in OOD accuracy and decline in $S_{\mathrm{avg}}^{\rho}$ over 500 generations indicate that GA-LoRA’s selection pressure implicitly favors individuals in flatter regions.
* The connection between flatter minima and improved robustness/generalization has been widely studied, both theoretically and empirically, in prior work (e.g., SAM [4] and Entropy-SGD [5]). The main difference across studies lies in how sharpness is defined. In our setting, we find that OOD robustness is strongly correlated with average-case sharpness.
We have clarified this distinction in the revised manuscript, framing $S_{\mathrm{avg}}^{\rho}$ as a strong indicator for the population’s concentration in generalizable basins rather than claiming a singular causal origin. We remain committed to exploring more granular explainable metrics in future work.

# Reviewer fErg

We thank the reviewer for the thoughtful and constructive comments. We agree that the motivation, statistical support, and efficiency discussion should be clearer, and we have revised the manuscript accordingly.

**Why gradient-free optimization for PEFT robustness-performance trade-offs:** We agree that many gradient-based methods also aim to find flatter minima. Our claim is not that gradient-free methods are universally necessary, but that they provide a complementary optimization bias that is particularly useful in PEFT. Our experiments have shown that while SAM explicitly minimizes worst-case sharpness, its OOD generalization is limited compared to standard optimizers like SGD though achieving good ID performance; and while entropy-SGD minimizes local entropy and achieves strong OOD accuracy for its , its requires significant training time for its SGLD loops(4x GA). GA selects for mutation-tolerant regions characterized by low average-case sharpness $S_{\mathrm{avg}}^{\rho}$, rather than directly minimizing a sharpness objective, suggesting that GA could be generalized to various scenarios as the robustness is a natural characteristic of the optimizer itself.

**Statistical significance and confidence intervals:** We have added multi-seed statistics (5 random seeds) and report confidence intervals in the revised results. We also reported ID, OOD and sharpness trends across 5 seeds in Figure 1 in the link: 

**Moderate gains vs. additional compute:** We agree that the OOD gains over SAM can appear moderate in some settings and that GA has a non-trivial time cost. At the same time, GA remains competitive under reduced budgets: when lowering the generation budget/population, wall-clock cost decreases substantially while preserving most robustness gains; at similar budgets (e.g., 200 generations in our analysis), GA still improves OOD performance over SAM, as shown in Figure 2 in our link.
Moreover, we have supplemented experiments on various CLIP backbones, showing that GA's advantage is consistent, and even widens on ViT-L/14 (428M parameters, about 5x ViT-B/16). We will explore why GA performs better on larger models in future work.

**Table 1. Cross-backbone comparison (16-shot).** 

| Method | ViT-B/16 (ID) | ViT-B/16 (OOD) | ViT-B/32 (ID) | ViT-B/32 (OOD) | ViT-L/14 (ID) | ViT-L/14 (OOD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| zero-shot | 66.7 | 57.1 | 62.0 | 47.8 | 73.4 | 70.3 |
| SGD | 69.6 | 59.6 | 64.7 | 49.6 | 76.3 | 71.8 |
| Adam | 70.5 | 59.7 | 64.8 | 48.7 | 77.8 | 72.3 |
| E-SGD | 70.4 | 60.1 | 65.1 | 50.1 | 76.7 | 72.8 |
| SAM | **71.1** | 59.6 | **65.3** | 48.8 | 75.9 | 69.8 |
| **GA** | 70.8 | **60.5** | **65.3** | **52.1** | **76.9** | **74.4** |

We acknowledge that GA's computational complexity is a major concern. We have updated the main text to report and discuss training-time comparisons, and we now highlight runtime as an explicit practical limitation alongside robustness gains.

**LoRA rank:** We appreciate the reviewer’s interest in the impact of LoRA rank on GA. In our implementation, most wall-clock cost comes from forward evaluation of the population. GA does not require backpropagation or gradient synchronization across devices, so increasing LoRA rank has only a limited effect on training time when the backbone is fixed. The main additional overhead comes from inter-GPU communication of LoRA parameters during parallel evaluation. Empirically, under the same experimental setup, the total training times for rank $r=2,4,8$ are 116, 121, and 126 minutes, respectively.
