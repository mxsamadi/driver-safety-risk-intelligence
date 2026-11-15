## Project Overview

This project demonstrates an end-to-end machine learning workflow for predicting driver safety incidents using synthetic data.

The objective is to explore how historical driver characteristics, violations, and compliance records can support proactive risk identification and safety review prioritization.

### Explore the Analysis

**Main Notebook:** [Driver Safety Risk Analysis](driver_safety_risk_analysis.ipynb)

The notebook includes:

- Synthetic data generation and exploratory data analysis
- Feature engineering and preprocessing
- Logistic Regression and Gradient Boosting model comparison
- Model evaluation using ROC-AUC, Average Precision, and Brier Score
- Risk-based driver prioritization using Recall@20%
- Calibration analysis and business recommendations

### Business Objective

Identify the highest-risk 20% of drivers to help safety teams prioritize reviews and evaluate how effectively predictive modeling concentrates future incidents within limited review capacity.

### Data Disclaimer

This project uses entirely synthetic data generated for educational and portfolio purposes. The findings do not represent actual driver safety patterns or the performance of a real-world safety prediction system.


# Driver Safety Risk Intelligence

**Independent portfolio case study | Synthetic data only**

## Executive question
Can pre-existing driver records help prioritize preventive safety outreach before incidents occur?

## Decision
Prioritize a limited-capacity outreach pilot for drivers with the highest predicted 12-month incident risk. **This is a hypothetical analysis, not a validated safety product.**

## Dataset and target
6,500 fully synthetic driver records generated using an explicitly programmed risk mechanism. The target `future_12m_incident` is a simulated event occurring after all features are measured. No real driver records or employer data are used. Because the outcome is generated from these features, model performance is illustrative and does **not** establish real-world predictive validity.

## Research hypotheses
1. Prior violations and incidents carry signal about subsequent incidents.
2. Exposure (annual miles) matters when estimating risk.
3. A calibrated risk ranking can concentrate simulated incidents in a reviewable subset.

## Method
- Stratified 60/20/20 train/validation/test split (a demonstration split; real deployments need out-of-time and entity-level testing).
- Compare logistic regression with histogram gradient boosting.
- Select by validation PR-AUC, then report untouched test performance.
- Assess ROC-AUC, PR-AUC, Brier score, calibration, and top-20% risk capture.
- Features are all defined as prior-period variables; no post-outcome fields are included.

## Illustrative holdout results
- Selected model: **Logistic regression**
- Test ROC-AUC: **0.698**; PR-AUC: **0.143** (baseline prevalence ≈ 0.0418)
- Brier score: **0.038**
- Top 20% risk segment captures **46%** of simulated incidents; precision **10%**

![Calibration](outputs/calibration.png)
![Risk by prior violations](outputs/risk_by_violations.png)

## Business decision framework
Let `p` = estimated incident probability, `r` = assumed relative reduction in incidents from outreach, `C` = estimated avoidable cost per incident, and `K` = outreach cost per driver. Then expected net value per outreach is `p × r × C − K`. For illustration only, with r=10%, C=$8,000 and K=$50, outreach is economically positive when p > 6.25%. **No causal effect, costs, or benefit are established by this synthetic dataset.** A randomized or carefully designed quasi-experimental pilot is required.

## Risks and limitations
- Synthetic generation encodes associations by design; it cannot validate real-world generalization.
- Region and fleet size may proxy for protected characteristics or operational differences; fairness assessment requires suitable, legally permissible data and governance.
- Exposure, reporting practices, and missingness may differ by fleet; check measurement bias.
- Avoid punitive or automated employment decisions; use as a reviewed safety-outreach decision support prototype.
- Evaluate calibration, drift, subgroup performance, and actual intervention effectiveness before any deployment.

## Recommendation to leadership
**Proceed only to a data feasibility and pilot-design stage.** Audit label definitions and timestamps, assess leakage, validate across time and fleets, and run a measured outreach pilot with predefined outcome and guardrail metrics.

## Reproduce
```bash
pip install -r requirements.txt
python analysis.py
```

`analysis.py` regenerates the synthetic dataset, models, metrics, and plots.
