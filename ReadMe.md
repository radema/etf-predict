# Predict ETF

This repository provides a pipeline for predicting Exchange-Traded Funds (ETFs) using historical market data and Facebook's Prophet forecasting model. The project fetches ETF data from Yahoo Finance, processes it, trains a time-series forecasting model, and generates predictions.

## Repository Structure

```
.
├── .github/workflows/project.yml   # GitHub Actions workflow for CI/CD
├── config/requirements.txt         # List of required Python packages
├── notebook/utils.py               # Utility functions for data extraction and transformation (notebook usage)
├── src/main.py                     # Main script to run the forecasting pipeline
├── src/utils.py                     # Utility functions for data extraction, transformation, and modeling
├── .gitignore                       # Files and folders to be ignored by Git
└── ReadMe.md                        # Documentation for the project
```

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/predict-etf.git
   cd predict-etf
   ```

2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```sh
   pip install -r config/requirements.txt
   ```

## Usage

To run the pipeline and generate ETF predictions, execute:
```sh
python src/main.py
```
By default, it will fetch historical ETF data, train a forecasting model using Prophet, and generate plots for future predictions.

### Optional Arguments:
- `--show`: Displays the generated forecast plots.
- `--save_fig`: Saves the forecast figures in the `data/report/` directory.

Example:
```sh
python src/main.py --show True --save_fig True
```

## Features
- Fetches ETF data from Yahoo Finance.
- Cleans and processes the data for time-series forecasting.
- Uses Facebook Prophet to train models for each ETF.
- Generates and saves forecasts.
- Evaluates model performance with cross-validation.

## Notes
- Ensure you have internet access to fetch ETF data.
- Modify `src/utils.py` to update the list of ETFs being analyzed.
- The output files (model performance reports, plots) are saved in the `data/` directory.

## Roadmap
- [ ] Improve data validation and error handling.
- [ ] Add more ETFs to the dataset.
- [ ] Create a web-based dashboard for visualization.

## License
This project is licensed under the MIT License.

---

If you have any questions or suggestions, feel free to contribute or raise an issue. 🚀

