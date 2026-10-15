# 📈 CARE

## Confidence-Adaptive Regime Experts for Portfolio Management

[**Paper**](./CARE_WWW.pdf)

**CARE** is a regime-adaptive portfolio management framework for non-stationary financial markets. Rather than forcing one global policy to reconcile the conflicting objectives of Bull, Neutral, and Bear markets, CARE learns three specialized reinforcement-learning experts and combines their candidate portfolios with a confidence-calibrated router.

The LLM is deliberately kept away from the executable trading layer. It converts causally available news and numerical indicators into an interpretable *soft prior* over market regimes; the trainable router then decides how much that prior should be trusted using its uncertainty, the current numerical market state, and each expert's realized historical reliability.

---

## 🚀 Framework

CARE consists of three jointly coordinated modules:

1. **LLM-based regime estimator**  
   Financial news and rolling market indicators are summarized separately and then consolidated into repeated Bull/Neutral/Bear probability estimates. Their average forms the semantic regime prior $\mathbf{q}^{\mathrm{LLM}}_t$. Its confidence is measured by normalized entropy.

2. **Confidence-calibrated mixture-of-experts router**  
   The router combines the LLM prior with causal numerical features and an exponentially weighted measure of each expert's recent risk-adjusted performance. A learned confidence gate attenuates ambiguous semantic guidance while allowing market evidence and expert feedback to correct an unreliable prior.

3. **Regime-specialized portfolio experts**  
   Bull, Neutral, and Bear experts output simplex-constrained candidate portfolios through stochastic Dirichlet policies. Their weak inductive biases encourage momentum exposure, balanced diversification, and downside protection, respectively. The final executable allocation is a convex mixture that includes an explicit cash position.

The router and experts are optimized jointly with PPO from realized, transaction-cost-adjusted portfolio returns. Load-balancing and regime-aligned specialization objectives discourage expert collapse without requiring regime labels during policy learning.


---

## 🛠️ Environment Setup

### 1. Create the Python Environment

```bash
conda create -n CARE python=3.13.5
conda activate CARE
```

Install the required Python packages:

```bash
pip install torch==2.5.1
pip install numpy==2.2.5
pip install requests
pip install pandas==2.3.0
```

If your machine requires a specific CUDA-compatible PyTorch build, please install the matching PyTorch wheel before installing the remaining packages.

---

## 🧩 LLM Backend Setup

CARE uses local LLMs served by [Ollama](https://ollama.com/).

Please install Ollama and prepare the required LLM backbones before running the experiments.

Supported LLM backbones include:

- 🦙 Llama-3-8B
- 🌪️ Mistral
- 💎 Gemma4-E2B
- 💎 Gemma4-E4B
- 🔮 GLM4

After installing Ollama, make sure the selected model can be called locally.

---

## ▶️ Running Experiments

### 1. Start the LLM Services

Open a terminal and start the local LLM services:

```bash
nohup python run_llama/run_llama1.py &
nohup python run_llama/run_llama2.py &
```

### 2. Run CARE

Open another terminal and run:
```bash
bash run/run_build_prior_cache.sh
```

```bash
bash run/run_regime_moe_experiments.sh
```
