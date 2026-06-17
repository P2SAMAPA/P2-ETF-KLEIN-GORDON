# Klein-Gordon Relativistic Quantum Field for ETFs

Applies the Klein-Gordon equation (relativistic quantum field theory) to model ETF returns as a scalar field. The field evolves according to the wave equation with a macro‑dependent mass term. The per‑ETF score is the field amplitude at the last time step – a measure of coherent trend/momentum.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Klein-Gordon equation with finite difference discretisation
- Macro‑dependent mass: m = mass_base + mass_range * macro_factor
- Score = field amplitude at the last time step
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-klein-gordon-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High field amplitude → coherent propagating trend (momentum).
- Low amplitude → weak or decaying field.

## Requirements

See `requirements.txt`.
