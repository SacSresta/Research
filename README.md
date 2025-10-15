# Research

This repository provides a collection of Python scripts, notebooks, and resources for advanced data analysis, machine learning experimentation, and research automation. The project includes work on time-series forecasting, sentiment analysis, and risk modeling, with implementations of neural architectures such as LSTM and GRU.

## Features

- **Time-Series Analysis:** Scripts and notebooks for forecasting and modeling, including LSTM and GRU-based models (`LSTM_main.py`, `gru_main.py`).
- **Sentiment Analysis:** Categorical sentiment analysis modules for conference and research data (`Conferance_sentiment_categorical`, `sentiment_categorical`).
- **Risk Modeling:** Combined risk assessment tools and output visualizations.
- **Research Automation:** Utilities for data fetching, preprocessing, and model evaluation.
- **Visualization:** Output images and analysis plots.
- **Utilities and Demos:** Tools for various Python tasks, including color editing (`pynche`), file processing, and algorithm demos.

## Directory Structure

- `ANALYSIS`, `MERGED`, `notebook`, `research/`, `data_pipeline/`: Project modules, data, and Jupyter notebooks.
- `models/`, `saved_models/`, `saved_params/`: Machine learning models and parameters.
- `Conferance_Data/`, `master_combined_risk/`: Source datasets and results.
- `requirements.txt`, `setup.py`, `.gitignore`: Project dependencies and setup.
- `output.png`, `returns.png`, `returnswithoutrisk.png`, `scatter.png`: Output plots and figures.
- `artifact/`, `dist/`: Build artifacts and distributions.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SacSresta/Research.git
   cd Research
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run analysis scripts:**
   - Example: `python main.py`
   - See notebooks and scripts for additional functionality.

## Important Notes

- Sensitive files such as `.env` have been removed for public safety.
- Please review CSVs, ZIPs, and model directories for any residual private or proprietary data before publishing.
- For questions or contributions, open an issue or pull request.

## License

Specify your license here (e.g., MIT, GPL, etc.).

---

**For more details, see the full file list and code comments.**
