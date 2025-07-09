# Physical Restraint Use in US Intensive Care Units: A MIMIC-IV Analysis (2008-2022)

## Overview

This repository contains the complete analysis code for our study examining racial and ethnic disparities in physical restraint use among intensive care unit patients. Using the MIMIC-IV database, we analyzed 51,838 adult ICU patients over a 14 year period to investigate patterns of restraint utilization and the impact of policy interventions.

## Repository Contents

### Data Extraction
- `sql-data-extraction.sql` - Primary SQL queries for extracting patient data, clinical variables, and restraint documentation from MIMIC-IV
- `alternative-definition-restraint-extraction-code.sql` - Alternative restraint identification using conservative keyword criteria for sensitivity analysis

### Analysis Scripts
- `preprocessing-and-table1.py` - Data cleaning, variable creation, and descriptive statistics (Table 1)
- `primary-stats-model-and-diagnostics.py` - Main binomial GLM fitting, interaction testing, and model diagnostics
- `secondary-analysis.py` - Binary restraint models, ICU stratification, and propensity score matching
- `sensitivity-analysis-alternative-definition.py` - Robustness testing using conservative restraint definition
- `temporal-change-analysis.py` - Time trend analysis and policy impact evaluation

## Data Requirements

This analysis uses the [MIMIC-IV database](https://mimic.mit.edu/), which requires:
- Completion of CITI training for human subjects research
- Signed data use agreement with PhysioNet
- Access to MIMIC-IV v3.1 or later

**Note**: Raw patient data cannot be shared publicly due to privacy restrictions. Researchers must obtain independent access to MIMIC-IV.

## Technical Requirements

```python
# Primary dependencies
pandas >= 1.3.0
numpy >= 1.21.0
statsmodels >= 0.13.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
scipy >= 1.7.0
```

## Usage Instructions

1. **Obtain MIMIC-IV access** through PhysioNet
2. **Extract data** using the provided SQL scripts
3. **Run preprocessing** to create analysis variables
4. **Execute analysis scripts** in the following order:
   - Data preprocessing and Table 1 generation
   - Primary statistical modeling and diagnostics
   - Secondary analyses and sensitivity testing
   - Temporal trend analysis

## Methodology

Our analysis employed:
- **Binomial generalized linear models** with logit link functions
- **Forward stepwise selection** followed by backward elimination
- **Interaction testing** for race × clinical factor effects
- **Propensity score matching** for bias control
- **Multiple sensitivity analyses** for robustness testing

## Clinical Impact

This research provides support for the claim that:
- Current restraint practices exhibit systematic ethnic bias
- Federal reporting policies have not reduced restraint use
- Healthcare organizations need targeted interventions addressing both overall restraint reduction and demographic equity
- Real time clinical decision support systems could help identify and prevent biased restraint decisions

## Citation

```
[Paper details to be added upon publication]
```

## Data Availability

- **Source code**: Available in this repository under MIT license
- **MIMIC-IV data**: Available through PhysioNet after completing required training
- **Processed datasets**: Cannot be shared due to data use agreements


## Ethics and Approvals

This research was conducted under the original MIMIC-IV IRB approvals from MIT and Beth Israel Deaconess Medical Center, with waiver of informed consent for retrospective analysis of deidentified data.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions about the analysis or code, please open an issue in this repository or contact the corresponding authors.

---

**Disclaimer**: This code is provided for research purposes. Clinical applications should be validated independently and implemented with appropriate clinical oversight.
