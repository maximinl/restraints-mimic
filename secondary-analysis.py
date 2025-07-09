import pandas as pd
import numpy as np
import statsmodels.api as sm

# --- Assume X_final_primary and model_df_aligned_for_diagnostics exist ---
# X_final_primary: Predictor DataFrame for the final primary model (includes interactions)
# model_df_aligned_for_diagnostics: Aligned DataFrame with observed outcomes and original variables
# Assume 'physically_restrained' and 'death_within_24h' columns exist in model_df_aligned_for_diagnostics

# IMPORTANT: Ensure these variables are available from prior execution AND
# that the 'physically_restrained' and 'death_within_24h' columns exist
# in model_df_aligned_for_diagnostics.
required_vars = ['X_final_primary', 'model_df_aligned_for_diagnostics']
required_outcome_cols = ['physically_restrained', 'death_within_24h']

if not all(v in locals() or v in globals() for v in required_vars):
    raise NameError(f"Missing required variables: {', '.join([v for v in required_vars if v not in locals() and v not in globals()])}. Please run the previous code blocks first.")

if not all(col in model_df_aligned_for_diagnostics.columns for col in required_outcome_cols):
    missing = [col for col in required_outcome_cols if col not in model_df_aligned_for_diagnostics.columns]
    raise ValueError(f"Missing required outcome columns for secondary analysis: {missing}. Ensure these columns exist in model_df_aligned_for_diagnostics.")


print("\n--- Secondary Analyses ---")

# Use the aligned DataFrame which should contain the outcome variables
analysis_df = model_df_aligned_for_diagnostics.copy() # Work with a copy

# Ensure X_final_primary index matches analysis_df index
# X_final_primary should already be aligned based on previous steps, but double-check
if not X_final_primary.index.equals(analysis_df.index):
    # Re-align X_final_primary if its index somehow drifted
    X_final_primary_aligned = X_final_primary.loc[analysis_df.index].copy()
    print("Info: Re-aligned X_final_primary to match analysis_df index for secondary analyses.")
else:
    X_final_primary_aligned = X_final_primary.copy() # Use as is if aligned


# --- Secondary Analysis 1: Binary Restraint Use (Yes/No) ---
print("\n--- Secondary Analysis 1: Binary Restraint Use (Yes/No) ---")

y_binary_restraint = analysis_df['physically_restrained'].astype(int)

# Fit Logistic Regression Model
print("\nFitting Logistic Regression for Binary Restraint Use...")
# Logistic Regression is a type of GLM with a Binomial family and Logit link (which is default)
binary_restraint_model = sm.GLM(y_binary_restraint, X_final_primary_aligned, family=sm.families.Binomial())

# Check if the outcome has variation (not all 0s or all 1s)
if y_binary_restraint.nunique() < 2:
     print(f"Warning: Outcome variable 'physically_restrained' has no variation (all {y_binary_restraint.iloc[0]}). Model cannot be fitted.")
else:
    try:
        binary_restraint_result = binary_restraint_model.fit()

        print("\n--- Binary Restraint Use Model Summary ---")
        print(binary_restraint_result.summary(xname=X_final_primary_aligned.columns.tolist())) # Use aligned column names

        print("\n--- Binary Restraint Use Model Odds Ratios ---")
        try:
            # Calculate Odds Ratios and 95% Confidence Intervals
            odds_ratios = np.exp(binary_restraint_result.params)
            conf_int = np.exp(binary_restraint_result.conf_int())
            p_values = binary_restraint_result.pvalues

            # Create a DataFrame for easier viewing
            or_df = pd.DataFrame({
                'Var': odds_ratios.index,
                'OR': odds_ratios.values,
                'CIL': conf_int.iloc[:, 0].values,
                'CIU': conf_int.iloc[:, 1].values,
                'p': p_values.values
            })
            or_df['OR (95% CI)'] = or_df.apply(lambda r: f"{r['OR']:.2f} ({r['CIL']:.2f}-{r['CIU']:.2f})" if pd.notna(r['OR']) else "N/A", axis=1)

            # Print the relevant columns
            print(or_df[['Var', 'OR (95% CI)', 'p']])

        except Exception as e:
            print(f"Error calculating ORs for binary restraint model: {e}")


    except Exception as e:
        print(f"\nError fitting the binary restraint model: {e}")


print("\n--- Secondary Analysis 1 Complete ---")


# --- Secondary Analysis 2: Death within 24h of Restraint (Yes/No) ---
print("\n\n--- Secondary Analysis 2: Death within 24h of Restraint (Yes/No) ---")

y_binary_death_restraint = analysis_df['death_within_24h'].astype(int)

# Fit Logistic Regression Model
print("\nFitting Logistic Regression for Death within 24h of Restraint...")
# Logistic Regression (Binomial family, Logit link)
death_restraint_model = sm.GLM(y_binary_death_restraint, X_final_primary_aligned, family=sm.families.Binomial())

# Check if the outcome has variation (not all 0s or all 1s)
if y_binary_death_restraint.nunique() < 2:
     print(f"Warning: Outcome variable 'death_within_24h' has no variation (all {y_binary_death_restraint.iloc[0]}). Model cannot be fitted.")
else:
    try:
        death_restraint_result = death_restraint_model.fit()

        print(f"\n--- Death within 24h of Restraint Model Summary ---")
        print(death_restraint_result.summary(xname=X_final_primary_aligned.columns.tolist()))

        print(f"\n--- Death within 24h of Restraint Model Odds Ratios ---")
        try:
            # Calculate Odds Ratios and 95% Confidence Intervals
            odds_ratios = np.exp(death_restraint_result.params)
            conf_int = np.exp(death_restraint_result.conf_int())
            p_values = death_restraint_result.pvalues

            # Create a DataFrame for easier viewing
            or_df = pd.DataFrame({
                'Var': odds_ratios.index,
                'OR': odds_ratios.values,
                'CIL': conf_int.iloc[:, 0].values,
                'CIU': conf_int.iloc[:, 1].values,
                'p': p_values.values
            })
            or_df['OR (95% CI)'] = or_df.apply(lambda r: f"{r['OR']:.2f} ({r['CIL']:.2f}-{r['CIU']:.2f})" if pd.notna(r['OR']) else "N/A", axis=1)

            # Print the relevant columns
            print(or_df[['Var', 'OR (95% CI)', 'p']])

        except Exception as e:
            print(f"Error calculating ORs for death within 24h model: {e}")


    except Exception as e:
        print(f"\nError fitting the death within 24h model: {e}")


print(f"\n--- Secondary Analyses Complete ---")

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings

# --- Enhanced ICU Stratification Analysis ---
# Building on your methodologically sound approach

# Validate required variables
required_vars = ['X_final_primary', 'y_binomial', 'model_df_aligned_for_diagnostics']
if not all(v in locals() or v in globals() for v in required_vars):
    raise NameError(f"Missing required variables: {', '.join([v for v in required_vars if v not in locals() and v not in globals()])}. Please run the previous code blocks first.")

# Ensure alignment
if not X_final_primary.index.equals(model_df_aligned_for_diagnostics.index):
    raise ValueError("Index mismatch between X_final_primary and model_df_aligned_for_diagnostics required for stratification.")

print("=" * 80)
print("ENHANCED ICU STRATIFICATION ANALYSIS")
print("=" * 80)
print("Analyzing demographic disparities in restraint use across ICU types")
print("Using validated model specification from primary analysis\n")

# Initialize results storage
stratification_results = {
    'icu_summary': [],
    'demographic_effects': [],
    'model_diagnostics': [],
    'interaction_findings': []
}

# Define key variables for focused analysis
demographic_vars = ['race_Asian', 'race_Hispanic_Latino', 'race_Unknown_Declined_Other', 'gender_M']
clinical_vars = ['deeply_sedated_day1', 'positive_cam_day1', 'first_day_sofa', 'chemical_restraint_day1']
interaction_vars = [col for col in X_final_primary.columns if '_x_' in col and any(demo in col for demo in ['Asian', 'Hispanic_Latino', 'Unknown_Declined_Other'])]

print(f"Focus variables identified:")
print(f"  Demographic: {demographic_vars}")
print(f"  Key Clinical: {clinical_vars}")
print(f"  Demographic Interactions: {len(interaction_vars)} interaction terms")

# Get ICU types and basic statistics
unique_icu_types = model_df_aligned_for_diagnostics['icu_type'].dropna().unique()
print(f"\nICU types for analysis: {len(unique_icu_types)}")

# Overall ICU distribution and restraint rates
print("\n" + "-" * 60)
print("ICU TYPE OVERVIEW")
print("-" * 60)

icu_overview = []
for icu_type in sorted(unique_icu_types):
    icu_indices = model_df_aligned_for_diagnostics.index[model_df_aligned_for_diagnostics['icu_type'] == icu_type]
    y_subset = y_binomial[model_df_aligned_for_diagnostics.index.get_indexer(icu_indices)]

    n_patients = len(icu_indices)
    n_restrained = y_subset.sum(axis=0)[0]
    n_total_days = y_subset.sum()
    restraint_rate = n_restrained / n_total_days if n_total_days > 0 else 0

    icu_overview.append({
        'ICU_Type': icu_type,
        'N_Patients': n_patients,
        'N_Restrained_Days': n_restrained,
        'Total_Days': n_total_days,
        'Restraint_Rate': restraint_rate,
        'Analyzable': n_patients >= 100 and n_restrained >= 10
    })

    print(f"{icu_type[:25].ljust(25)} | {n_patients:>6,} pts | {restraint_rate:>6.1%} restraint rate | {'✓' if n_patients >= 100 and n_restrained >= 10 else '✗'}")

icu_overview_df = pd.DataFrame(icu_overview)
analyzable_icus = icu_overview_df[icu_overview_df['Analyzable']]['ICU_Type'].tolist()

print(f"\nICU types meeting analysis criteria (≥100 patients, ≥10 restraint days): {len(analyzable_icus)}")
print(f"Analyzable ICUs: {analyzable_icus}")

# --- Detailed Analysis by ICU Type ---
print("\n" + "=" * 80)
print("DETAILED ICU STRATIFIED ANALYSIS")
print("=" * 80)

for icu_type in analyzable_icus:
    print(f"\n{'='*20} {icu_type} {'='*20}")

    # Get subset indices and data
    icu_indices = model_df_aligned_for_diagnostics.index[model_df_aligned_for_diagnostics['icu_type'] == icu_type]
    X_subset = X_final_primary.loc[icu_indices].copy()
    y_subset_binomial = y_binomial[model_df_aligned_for_diagnostics.index.get_indexer(icu_indices)]

    # Calculate basic statistics
    n_patients = len(X_subset)
    n_restrained_days = y_subset_binomial.sum(axis=0)[0]
    n_non_restrained_days = y_subset_binomial.sum(axis=0)[1]
    total_days = n_restrained_days + n_non_restrained_days
    restraint_rate = n_restrained_days / total_days

    print(f"Sample: {n_patients:,} patients, {total_days:,} ICU days")
    print(f"Restraint rate: {restraint_rate:.1%} ({n_restrained_days:,} restrained days)")

    # Demographic composition
    demo_data = model_df_aligned_for_diagnostics.loc[icu_indices]
    print(f"\nDemographic composition:")

    # Calculate demographic percentages
    total_subset = len(demo_data)
    if 'race_Asian' in X_subset.columns:
        asian_pct = X_subset['race_Asian'].sum() / total_subset * 100
        print(f"  Asian: {asian_pct:.1f}%")
    if 'race_Hispanic_Latino' in X_subset.columns:
        hispanic_pct = X_subset['race_Hispanic_Latino'].sum() / total_subset * 100
        print(f"  Hispanic/Latino: {hispanic_pct:.1f}%")
    if 'race_Unknown_Declined_Other' in X_subset.columns:
        unknown_pct = X_subset['race_Unknown_Declined_Other'].sum() / total_subset * 100
        print(f"  Other/Unknown: {unknown_pct:.1f}%")
    if 'gender_M' in X_subset.columns:
        male_pct = X_subset['gender_M'].sum() / total_subset * 100
        print(f"  Male: {male_pct:.1f}%")

    # Remove low variance columns for this subset
    X_subset_check = X_subset.drop(columns='const', errors='ignore')
    subset_variances = X_subset_check.var()
    subset_low_var_cols = subset_variances[subset_variances < 1e-9].index.tolist()

    if subset_low_var_cols:
        print(f"\nRemoving low-variance variables: {subset_low_var_cols}")
        X_subset = X_subset.drop(columns=subset_low_var_cols)

    # Check for adequate model complexity
    remaining_vars = X_subset.shape[1]
    if remaining_vars <= 2:  # Only const + 1 other variable
        print(f"Warning: Too few variables ({remaining_vars}) after preprocessing. Skipping detailed analysis.")
        continue

    # Fit the model
    print(f"\nFitting logistic regression model...")
    try:
        X_subset.columns = X_subset.columns.astype(str)
        icu_model = sm.GLM(y_subset_binomial, X_subset, family=sm.families.Binomial())
        icu_result = icu_model.fit()

        if not icu_result.converged:
            print(f"⚠️  Warning: Model convergence issues for {icu_type}")

        # Calculate odds ratios
        odds_ratios = np.exp(icu_result.params)
        conf_int = np.exp(icu_result.conf_int())
        p_values = icu_result.pvalues

        # Create results dataframe
        or_df = pd.DataFrame({
            'Variable': odds_ratios.index,
            'OR': odds_ratios.values,
            'CI_Lower': conf_int.iloc[:, 0].values,
            'CI_Upper': conf_int.iloc[:, 1].values,
            'P_Value': p_values.values,
            'Significant': p_values.values < 0.05
        })

        or_df['OR_CI_Display'] = or_df.apply(
            lambda r: f"{r['OR']:.2f} ({r['CI_Lower']:.2f}-{r['CI_Upper']:.2f})"
            if pd.notna(r['OR']) else "N/A", axis=1
        )

        # Store model diagnostics
        stratification_results['model_diagnostics'].append({
            'ICU_Type': icu_type,
            'N_Patients': n_patients,
            'N_Parameters': len(icu_result.params),
            'Converged': icu_result.converged,
            'Log_Likelihood': icu_result.llf,
            'AIC': icu_result.aic,
            'Pseudo_R2': 1 - (icu_result.deviance / icu_result.null_deviance)
        })

        # Focus on demographic findings
        print(f"\n🎯 KEY DEMOGRAPHIC FINDINGS:")
        demographic_results = or_df[or_df['Variable'].isin(demographic_vars)].copy()

        if not demographic_results.empty:
            print(f"{'Variable':<25} {'OR (95% CI)':<20} {'P-Value':<10} {'Significant'}")
            print("-" * 65)

            for _, row in demographic_results.iterrows():
                var_clean = row['Variable'].replace('race_', '').replace('_', ' ')
                sig_marker = "***" if row['P_Value'] < 0.001 else "**" if row['P_Value'] < 0.01 else "*" if row['P_Value'] < 0.05 else ""

                print(f"{var_clean:<25} {row['OR_CI_Display']:<20} {row['P_Value']:<10.3f} {sig_marker}")

                # Store demographic effects
                stratification_results['demographic_effects'].append({
                    'ICU_Type': icu_type,
                    'Variable': row['Variable'],
                    'OR': row['OR'],
                    'CI_Lower': row['CI_Lower'],
                    'CI_Upper': row['CI_Upper'],
                    'P_Value': row['P_Value'],
                    'Significant': row['Significant'],
                    'N_Patients': n_patients
                })
        else:
            print("No demographic variables retained in final model")

        # Check for significant interactions
        interaction_results = or_df[or_df['Variable'].isin(interaction_vars) & (or_df['P_Value'] < 0.05)].copy()

        if not interaction_results.empty:
            print(f"\n🔗 SIGNIFICANT DEMOGRAPHIC INTERACTIONS:")
            for _, row in interaction_results.iterrows():
                interaction_clean = row['Variable'].replace('_x_', ' × ').replace('_', ' ')
                print(f"  {interaction_clean}: OR = {row['OR_CI_Display']}, p = {row['P_Value']:.3f}")

                stratification_results['interaction_findings'].append({
                    'ICU_Type': icu_type,
                    'Interaction': row['Variable'],
                    'OR': row['OR'],
                    'P_Value': row['P_Value'],
                    'N_Patients': n_patients
                })

        # Store ICU summary
        stratification_results['icu_summary'].append({
            'ICU_Type': icu_type,
            'N_Patients': n_patients,
            'Restraint_Rate': restraint_rate,
            'Significant_Demographics': sum(demographic_results['Significant']) if not demographic_results.empty else 0,
            'Significant_Interactions': len(interaction_results),
            'Model_Converged': icu_result.converged
        })

    except Exception as e:
        print(f"❌ Model fitting failed for {icu_type}: {str(e)}")
        if "Perfect separation" in str(e) or "Singular matrix" in str(e):
            print("   This suggests perfect separation or multicollinearity in this ICU subset")
        continue

# --- Summary Analysis Across ICU Types ---
print("\n" + "=" * 80)
print("CROSS-ICU COMPARISON SUMMARY")
print("=" * 80)

if stratification_results['demographic_effects']:
    demo_effects_df = pd.DataFrame(stratification_results['demographic_effects'])

    print("\n📊 DEMOGRAPHIC EFFECTS COMPARISON ACROSS ICU TYPES:")

    for demo_var in demographic_vars:
        var_results = demo_effects_df[demo_effects_df['Variable'] == demo_var].copy()

        if not var_results.empty:
            var_clean = demo_var.replace('race_', '').replace('_', ' ').title()
            print(f"\n{var_clean}:")
            print(f"{'ICU Type':<25} {'OR (95% CI)':<20} {'P-Value':<10} {'N Patients'}")
            print("-" * 70)

            for _, row in var_results.iterrows():
                icu_short = row['ICU_Type'][:22]
                or_display = f"{row['OR']:.2f} ({row['CI_Lower']:.2f}-{row['CI_Upper']:.2f})"
                sig_marker = "***" if row['P_Value'] < 0.001 else "**" if row['P_Value'] < 0.01 else "*" if row['P_Value'] < 0.05 else ""

                print(f"{icu_short:<25} {or_display:<20} {row['P_Value']:<10.3f} {row['N_Patients']:>8,} {sig_marker}")

            # Calculate consistency metrics
            consistent_direction = (var_results['OR'] > 1).all() or (var_results['OR'] < 1).all()
            any_significant = var_results['Significant'].any()

            print(f"  → Consistent direction: {'Yes' if consistent_direction else 'No'}")
            print(f"  → Any significant: {'Yes' if any_significant else 'No'}")

# --- Key Research Insights ---
print("\n" + "=" * 80)
print("KEY RESEARCH INSIGHTS")
print("=" * 80)

if stratification_results['icu_summary']:
    summary_df = pd.DataFrame(stratification_results['icu_summary'])

    # ICU types with highest/lowest restraint rates
    highest_restraint = summary_df.loc[summary_df['Restraint_Rate'].idxmax()]
    lowest_restraint = summary_df.loc[summary_df['Restraint_Rate'].idxmin()]

    print(f"\n🏥 ICU PRACTICE VARIATION:")
    print(f"  Highest restraint rate: {highest_restraint['ICU_Type']} ({highest_restraint['Restraint_Rate']:.1%})")
    print(f"  Lowest restraint rate: {lowest_restraint['ICU_Type']} ({lowest_restraint['Restraint_Rate']:.1%})")
    print(f"  Fold difference: {highest_restraint['Restraint_Rate']/lowest_restraint['Restraint_Rate']:.1f}x")

    # Demographic disparity consistency
    total_icus_analyzed = len(summary_df)
    icus_with_demo_effects = sum(summary_df['Significant_Demographics'] > 0)

    print(f"\n📈 DEMOGRAPHIC DISPARITY PATTERNS:")
    print(f"  ICU types analyzed: {total_icus_analyzed}")
    print(f"  ICUs with significant demographic effects: {icus_with_demo_effects}")
    print(f"  Consistency rate: {icus_with_demo_effects/total_icus_analyzed:.1%}")

# Store results for further analysis
print(f"\n💾 RESULTS STORED:")
print(f"  ICU summaries: {len(stratification_results['icu_summary'])}")
print(f"  Demographic effects: {len(stratification_results['demographic_effects'])}")
print(f"  Interaction findings: {len(stratification_results['interaction_findings'])}")
print(f"  Model diagnostics: {len(stratification_results['model_diagnostics'])}")

print("\n" + "=" * 80)
print("ICU STRATIFICATION ANALYSIS COMPLETE")
print("=" * 80)
print("✅ Results available in 'stratification_results' dictionary")
print("✅ Ready for manuscript preparation and visualization")

# Create summary for manuscript
manuscript_summary = {
    'total_icus_analyzed': len(stratification_results['icu_summary']),
    'total_patients': sum([r['N_Patients'] for r in stratification_results['icu_summary']]),
    'restraint_rate_range': (
        min([r['Restraint_Rate'] for r in stratification_results['icu_summary']]),
        max([r['Restraint_Rate'] for r in stratification_results['icu_summary']])
    ),
    'icus_with_demographic_effects': sum([r['Significant_Demographics'] > 0 for r in stratification_results['icu_summary']]),
    'demographic_effects_by_variable': {}
}

# Count effects by demographic variable
if stratification_results['demographic_effects']:
    demo_df = pd.DataFrame(stratification_results['demographic_effects'])
    for var in demographic_vars:
        var_effects = demo_df[demo_df['Variable'] == var]
        manuscript_summary['demographic_effects_by_variable'][var] = {
            'total_icus': len(var_effects),
            'significant_icus': sum(var_effects['Significant']),
            'consistent_direction': len(set(var_effects['OR'] > 1)) == 1
        }

print(f"\n📋 MANUSCRIPT SUMMARY:")
print(f"  Analyzed {manuscript_summary['total_icus_analyzed']} ICU types")
print(f"  Total patients: {manuscript_summary['total_patients']:,}")
print(f"  Restraint rate range: {manuscript_summary['restraint_rate_range'][0]:.1%} - {manuscript_summary['restraint_rate_range'][1]:.1%}")
print(f"  ICUs with demographic effects: {manuscript_summary['icus_with_demographic_effects']}")

# Examine the specific demographic effects by ICU
demo_effects_df = pd.DataFrame(stratification_results['demographic_effects'])

# Show which demographics are consistently affected
for demo_var in ['race_Asian', 'race_Hispanic_Latino', 'race_Unknown_Declined_Other']:
    var_results = demo_effects_df[demo_effects_df['Variable'] == demo_var]
    print(f"\n{demo_var} effects across ICUs:")
    print(var_results[['ICU_Type', 'OR', 'P_Value', 'Significant']])


# Examine the 52 interaction findings
interactions_df = pd.DataFrame(stratification_results['interaction_findings'])
print("Most significant interactions by ICU type:")
print(interactions_df.sort_values('P_Value').head(10))

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PROPENSITY SCORE MATCHING ANALYSIS")
print("=" * 80)
print("Validating demographic disparities through matched cohort analysis")
print("Adapting protocol to focus on Asian and Hispanic/Latino patients\n")

# Validate required variables
required_vars = ['X_final_primary', 'y_binomial', 'model_df_aligned_for_diagnostics']
if not all(v in locals() or v in globals() for v in required_vars):
    raise NameError(f"Missing required variables: {', '.join([v for v in required_vars if v not in locals() and v not in globals()])}")

# Create working dataset
print("📋 PREPARING PROPENSITY SCORE MATCHING DATASET")
print("-" * 60)

# Combine all necessary data
matching_df = model_df_aligned_for_diagnostics.copy()

# Add predictor variables
for col in X_final_primary.columns:
    if col not in matching_df.columns:
        matching_df[col] = X_final_primary[col]

# Add outcome information
restraint_outcomes = []
for i in range(len(y_binomial)):
    total_days = y_binomial[i].sum()
    restrained_days = y_binomial[i][0]
    restraint_proportion = restrained_days / total_days if total_days > 0 else 0
    restraint_binary = 1 if restrained_days > 0 else 0

    restraint_outcomes.append({
        'restrained_days': restrained_days,
        'total_days': total_days,
        'restraint_proportion': restraint_proportion,
        'restraint_binary': restraint_binary
    })

restraint_df = pd.DataFrame(restraint_outcomes, index=matching_df.index)
matching_df = pd.concat([matching_df, restraint_df], axis=1)

print(f"Total dataset: {len(matching_df):,} patients")
print(f"Overall restraint rate: {matching_df['restraint_binary'].mean():.1%}")

# Define demographic groups for matching
demographic_groups = {
    'Asian': 'race_Asian',
    'Hispanic_Latino': 'race_Hispanic_Latino',
    'Unknown_Other': 'race_Unknown_Declined_Other'
}

# Caucasian patients (reference group - all others are 0)
matching_df['race_Caucasian'] = 1 - (
    matching_df.get('race_Asian', 0) +
    matching_df.get('race_Hispanic_Latino', 0) +
    matching_df.get('race_Unknown_Declined_Other', 0) +
    matching_df.get('race_Black_African_American', 0)  # In case it exists
)

print(f"\nDemographic composition:")
for group_name, group_var in demographic_groups.items():
    if group_var in matching_df.columns:
        count = matching_df[group_var].sum()
        pct = count / len(matching_df) * 100
        print(f"  {group_name}: {count:,} ({pct:.1f}%)")

caucasian_count = matching_df['race_Caucasian'].sum()
caucasian_pct = caucasian_count / len(matching_df) * 100
print(f"  Caucasian: {caucasian_count:,} ({caucasian_pct:.1f}%)")

def estimate_propensity_scores(df, treatment_var, covariates):
    """
    Estimate propensity scores for treatment assignment
    """
    print(f"\n🎯 ESTIMATING PROPENSITY SCORES FOR {treatment_var.replace('race_', '').upper()}")
    print("-" * 50)

    # Prepare covariates (exclude demographic variables to avoid post-treatment bias)
    clinical_covariates = [var for var in covariates if not var.startswith('race_') and var != treatment_var]
    available_covariates = [var for var in clinical_covariates if var in df.columns]

    print(f"Covariates used: {available_covariates}")

    # Prepare data
    X_ps = df[available_covariates].fillna(0)
    y_ps = df[treatment_var]

    # Check treatment prevalence
    treatment_prev = y_ps.mean()
    print(f"Treatment prevalence: {treatment_prev:.1%}")

    if treatment_prev < 0.01 or treatment_prev > 0.99:
        print(f"⚠️  Warning: Extreme treatment prevalence ({treatment_prev:.1%})")
        return None, None

    # Fit propensity score model
    try:
        ps_model = LogisticRegression(random_state=42, max_iter=1000)
        ps_model.fit(X_ps, y_ps)

        # Get propensity scores
        propensity_scores = ps_model.predict_proba(X_ps)[:, 1]

        print(f"Propensity score range: {propensity_scores.min():.3f} - {propensity_scores.max():.3f}")
        print(f"Mean propensity score: {propensity_scores.mean():.3f}")

        return propensity_scores, ps_model

    except Exception as e:
        print(f"❌ Propensity score estimation failed: {e}")
        return None, None

def perform_nearest_neighbor_matching(df, treatment_var, propensity_scores, caliper=0.1):
    """
    Perform 1:1 nearest neighbor matching with caliper
    """
    print(f"\n🔗 PERFORMING 1:1 NEAREST NEIGHBOR MATCHING")
    print("-" * 45)

    if propensity_scores is None:
        return None

    # Create dataframe with propensity scores for easier handling
    df_with_ps = df.copy()
    df_with_ps['propensity_score'] = propensity_scores

    # Separate treatment and control groups
    treatment_group = df_with_ps[df_with_ps[treatment_var] == 1].copy()
    control_group = df_with_ps[df_with_ps[treatment_var] == 0].copy()

    print(f"Treatment group size: {len(treatment_group):,}")
    print(f"Control pool size: {len(control_group):,}")

    # Perform matching
    matched_pairs = []
    used_control_indices = set()

    # Sort treatment group by propensity score for stable matching
    treatment_group_sorted = treatment_group.sort_values('propensity_score')

    for treat_idx, treat_row in treatment_group_sorted.iterrows():
        treat_ps = treat_row['propensity_score']

        # Find eligible controls (within caliper and not yet used)
        eligible_controls = control_group[
            (~control_group.index.isin(used_control_indices)) &
            (abs(control_group['propensity_score'] - treat_ps) <= caliper)
        ].copy()

        if len(eligible_controls) > 0:
            # Select closest match
            eligible_controls['ps_distance'] = abs(eligible_controls['propensity_score'] - treat_ps)
            best_match = eligible_controls.loc[eligible_controls['ps_distance'].idxmin()]

            matched_pairs.append({
                'treatment_idx': treat_idx,
                'control_idx': best_match.name,
                'treatment_ps': treat_ps,
                'control_ps': best_match['propensity_score'],
                'ps_difference': best_match['ps_distance']
            })

            used_control_indices.add(best_match.name)

    print(f"Successful matches: {len(matched_pairs):,}")
    print(f"Match rate: {len(matched_pairs)/len(treatment_group):.1%}")

    if matched_pairs:
        ps_diffs = [pair['ps_difference'] for pair in matched_pairs]
        print(f"Mean PS difference: {np.mean(ps_diffs):.4f}")
        print(f"Max PS difference: {np.max(ps_diffs):.4f}")

        return pd.DataFrame(matched_pairs)
    else:
        return None

def assess_balance_after_matching(df, matched_pairs_df, treatment_var, covariates):
    """
    Assess covariate balance after matching
    """
    print(f"\n⚖️  ASSESSING COVARIATE BALANCE AFTER MATCHING")
    print("-" * 50)

    if matched_pairs_df is None:
        print("No matched pairs to assess")
        return None

    # Create matched dataset
    treatment_indices = matched_pairs_df['treatment_idx'].tolist()
    control_indices = matched_pairs_df['control_idx'].tolist()
    matched_treatment = df.loc[treatment_indices]
    matched_control = df.loc[control_indices]

    balance_results = []

    # Check balance for key covariates
    key_covariates = [var for var in covariates if var in df.columns and not var.startswith('race_')]

    print(f"{'Variable':<25} {'Treatment Mean':<15} {'Control Mean':<15} {'Std Diff':<10} {'P-Value'}")
    print("-" * 80)

    for var in key_covariates:
        if var in df.columns:
            treat_mean = matched_treatment[var].mean()
            ctrl_mean = matched_control[var].mean()

            # Calculate standardized difference
            pooled_std = np.sqrt(
                (matched_treatment[var].var() + matched_control[var].var()) / 2
            )
            std_diff = abs(treat_mean - ctrl_mean) / pooled_std if pooled_std > 0 else 0

            # T-test for difference
            try:
                _, p_value = ttest_ind(matched_treatment[var].dropna(),
                                     matched_control[var].dropna())
            except:
                p_value = np.nan

            balance_results.append({
                'variable': var,
                'treatment_mean': treat_mean,
                'control_mean': ctrl_mean,
                'std_diff': std_diff,
                'p_value': p_value
            })

            balance_flag = "✓" if std_diff < 0.1 else "⚠️" if std_diff < 0.25 else "❌"
            print(f"{var[:24]:<25} {treat_mean:<15.3f} {ctrl_mean:<15.3f} {std_diff:<10.3f} {p_value:<10.3f} {balance_flag}")

    # Summary
    good_balance = sum(1 for r in balance_results if r['std_diff'] < 0.1)
    total_vars = len(balance_results)

    print(f"\nBalance Summary:")
    print(f"  Variables with good balance (std diff < 0.1): {good_balance}/{total_vars}")
    print(f"  Overall balance quality: {good_balance/total_vars:.1%}")

    return pd.DataFrame(balance_results)

def run_matched_analysis(df, matched_pairs_df, treatment_var, X_final_cols, y_binomial_original):
    """
    Run the primary analysis on matched cohort
    """
    print(f"\n📊 RUNNING MATCHED COHORT ANALYSIS")
    print("-" * 40)

    if matched_pairs_df is None:
        print("No matched pairs available for analysis")
        return None

    # Create matched dataset indices
    treatment_indices = matched_pairs_df['treatment_idx'].tolist()
    control_indices = matched_pairs_df['control_idx'].tolist()
    matched_indices = treatment_indices + control_indices

    print(f"Matched cohort size: {len(matched_indices):,} patients")
    print(f"  Treatment group: {len(treatment_indices):,}")
    print(f"  Control group: {len(control_indices):,}")

    # Get matched data
    X_matched = X_final_primary.loc[matched_indices].copy()

    # Get corresponding y_binomial data
    original_index_map = {idx: pos for pos, idx in enumerate(model_df_aligned_for_diagnostics.index)}
    matched_positions = [original_index_map[idx] for idx in matched_indices]
    y_matched = y_binomial_original[matched_positions]

    # Remove variables with no variation in matched cohort
    X_matched_check = X_matched.drop(columns='const', errors='ignore')
    variances = X_matched_check.var()
    low_var_cols = variances[variances < 1e-9].index.tolist()

    if low_var_cols:
        print(f"Removing low-variance variables in matched cohort: {low_var_cols}")
        X_matched = X_matched.drop(columns=low_var_cols)

    # Fit model on matched cohort
    print(f"\nFitting primary model on matched cohort...")

    try:
        matched_model = sm.GLM(y_matched, X_matched, family=sm.families.Binomial())
        matched_result = matched_model.fit()

        if not matched_result.converged:
            print("⚠️  Warning: Model convergence issues in matched analysis")

        # Calculate odds ratios
        odds_ratios = np.exp(matched_result.params)
        conf_int = np.exp(matched_result.conf_int())
        p_values = matched_result.pvalues

        # Create results dataframe
        matched_or_df = pd.DataFrame({
            'Variable': odds_ratios.index,
            'OR': odds_ratios.values,
            'CI_Lower': conf_int.iloc[:, 0].values,
            'CI_Upper': conf_int.iloc[:, 1].values,
            'P_Value': p_values.values,
            'Significant': p_values.values < 0.05
        })

        matched_or_df['OR_CI_Display'] = matched_or_df.apply(
            lambda r: f"{r['OR']:.2f} ({r['CI_Lower']:.2f}-{r['CI_Upper']:.2f})"
            if pd.notna(r['OR']) else "N/A", axis=1
        )

        # Focus on demographic findings
        demographic_vars = ['race_Asian', 'race_Hispanic_Latino', 'race_Unknown_Declined_Other', 'gender_M']
        demo_results = matched_or_df[matched_or_df['Variable'].isin(demographic_vars)].copy()

        if not demo_results.empty:
            print(f"\n🎯 DEMOGRAPHIC EFFECTS IN MATCHED COHORT:")
            print(f"{'Variable':<25} {'OR (95% CI)':<20} {'P-Value':<10} {'Significant'}")
            print("-" * 65)

            for _, row in demo_results.iterrows():
                var_clean = row['Variable'].replace('race_', '').replace('_', ' ')
                sig_marker = "***" if row['P_Value'] < 0.001 else "**" if row['P_Value'] < 0.01 else "*" if row['P_Value'] < 0.05 else ""

                print(f"{var_clean:<25} {row['OR_CI_Display']:<20} {row['P_Value']:<10.3f} {sig_marker}")

        return {
            'model_result': matched_result,
            'odds_ratios_df': matched_or_df,
            'demographic_results': demo_results,
            'matched_indices': matched_indices,
            'n_matched': len(matched_indices)
        }

    except Exception as e:
        print(f"❌ Matched analysis failed: {e}")
        return None

# Main propensity score matching execution
propensity_results = {}

# Define covariates for propensity score estimation
ps_covariates = [
    'const', 'gender_M', 'deeply_sedated_day1', 'positive_cam_day1',
    'first_day_sofa', 'first_day_oasis', 'first_day_sapsii',
    'chemical_restraint_day1', 'ventilated_day1', 'evd_or_ecmo',
    'has_psychosis_admission_dx', 'has_mania_bipolar_admission_dx',
    'has_substance_use_admission_dx', 'has_phys_condition_mental_disorder_admission_dx'
] + [col for col in X_final_primary.columns if col.startswith('icu_type_')]

# Run propensity score matching for each demographic group
for group_name, group_var in demographic_groups.items():
    if group_var in matching_df.columns and matching_df[group_var].sum() >= 100:

        print(f"\n{'='*20} {group_name.upper()} vs CAUCASIAN MATCHING {'='*20}")

        # Create binary dataset for this comparison
        comparison_df = matching_df[
            (matching_df[group_var] == 1) | (matching_df['race_Caucasian'] == 1)
        ].copy()

        if len(comparison_df) < 200:
            print(f"Insufficient sample size for {group_name} matching: {len(comparison_df)}")
            continue

        # Estimate propensity scores
        ps_scores, ps_model = estimate_propensity_scores(
            comparison_df, group_var, ps_covariates
        )

        if ps_scores is not None:
            # Perform matching
            matched_pairs = perform_nearest_neighbor_matching(
                comparison_df, group_var, ps_scores, caliper=0.1
            )

            if matched_pairs is not None and len(matched_pairs) > 0:
                # Assess balance
                balance_df = assess_balance_after_matching(
                    comparison_df, matched_pairs, group_var, ps_covariates
                )

                # Run matched analysis
                matched_analysis = run_matched_analysis(
                    comparison_df, matched_pairs, group_var,
                    X_final_primary.columns, y_binomial
                )

                # Store results
                propensity_results[group_name] = {
                    'propensity_scores': ps_scores,
                    'propensity_model': ps_model,
                    'matched_pairs': matched_pairs,
                    'balance_assessment': balance_df,
                    'matched_analysis': matched_analysis,
                    'comparison_df': comparison_df
                }
            else:
                print(f"❌ No successful matches for {group_name}")
    else:
        min_group_size = matching_df[group_var].sum() if group_var in matching_df.columns else 0
        print(f"Skipping {group_name}: insufficient sample size ({min_group_size})")

# Summary of propensity score matching results
print(f"\n{'='*80}")
print("PROPENSITY SCORE MATCHING SUMMARY")
print("="*80)

successful_matches = 0
total_demographic_effects = 0

for group_name, results in propensity_results.items():
    if results['matched_analysis'] is not None:
        successful_matches += 1
        n_matched = results['matched_analysis']['n_matched']
        demo_results = results['matched_analysis']['demographic_results']
        n_sig_demo = sum(demo_results['Significant']) if not demo_results.empty else 0

        print(f"\n{group_name}:")
        print(f"  ✅ Successful matching: {n_matched:,} patients")
        print(f"  📊 Significant demographic effects after matching: {n_sig_demo}")

        if not demo_results.empty:
            target_var = f"race_{group_name}" if group_name != 'Hispanic_Latino' else 'race_Hispanic_Latino'
            target_result = demo_results[demo_results['Variable'] == target_var]
            if not target_result.empty:
                or_val = target_result.iloc[0]['OR']
                p_val = target_result.iloc[0]['P_Value']
                print(f"  🎯 {group_name} effect: OR = {or_val:.2f}, p = {p_val:.3f}")
                total_demographic_effects += 1 if p_val < 0.05 else 0

print(f"\n📋 OVERALL MATCHING SUMMARY:")
print(f"  Groups successfully matched: {successful_matches}/{len(demographic_groups)}")
print(f"  Demographic effects persisting after matching: {total_demographic_effects}")
print(f"  Robustness validation: {'✅ Confirmed' if total_demographic_effects > 0 else '❌ Not confirmed'}")

print(f"\n✅ PROPENSITY SCORE MATCHING ANALYSIS COMPLETE")
print("📊 Results stored in 'propensity_results' dictionary")
print("🔄 Ready for temporal trend analysis (next protocol step)")

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("COMPLETE CASE ANALYSIS")
print("=" * 80)
print("Assessing robustness of findings to missing data patterns")
print("Protocol sensitivity analysis: patients with complete data only\n")

# Validate required variables
required_vars = ['X_final_primary', 'y_binomial', 'model_df_aligned_for_diagnostics']
if not all(v in locals() or v in globals() for v in required_vars):
    raise NameError(f"Missing required variables: {', '.join([v for v in required_vars if v not in locals() and v not in globals()])}")

print("📋 MISSING DATA ASSESSMENT")
print("-" * 50)

# Step 1: Assess missing data patterns in key variables
print(f"Original dataset: {len(model_df_aligned_for_diagnostics):,} patients")

# Combine predictor variables with outcome and key demographic variables
analysis_vars = list(X_final_primary.columns)

# Add outcome data
outcome_data = []
for i in range(len(y_binomial)):
    total_days = y_binomial[i].sum()
    restrained_days = y_binomial[i][0]
    outcome_data.append({
        'restrained_days': restrained_days,
        'total_days': total_days,
        'has_outcome_data': total_days > 0
    })

outcome_df = pd.DataFrame(outcome_data, index=model_df_aligned_for_diagnostics.index)

# Create comprehensive dataset for missing data assessment
complete_data_df = model_df_aligned_for_diagnostics.copy()

# Add predictor variables
for col in X_final_primary.columns:
    if col not in complete_data_df.columns:
        complete_data_df[col] = X_final_primary[col]

# Add outcome data
for col in outcome_df.columns:
    complete_data_df[col] = outcome_df[col]

print(f"\nMissing data assessment across {len(analysis_vars)} key variables:")

# Check missing patterns for each variable
missing_summary = []
for var in analysis_vars:
    if var in complete_data_df.columns:
        missing_count = complete_data_df[var].isnull().sum()
        missing_pct = (missing_count / len(complete_data_df)) * 100

        missing_summary.append({
            'Variable': var,
            'Missing_Count': missing_count,
            'Missing_Percent': missing_pct,
            'Complete_Count': len(complete_data_df) - missing_count
        })

missing_df = pd.DataFrame(missing_summary)

# Display missing data summary
print(f"\n{'Variable':<35} {'Missing':<10} {'%':<8} {'Complete':<10}")
print("-" * 70)

variables_with_missing = 0
total_missing = 0

for _, row in missing_df.iterrows():
    var_name = row['Variable'][:32]  # Truncate long names
    missing_count = int(row['Missing_Count'])
    missing_pct = row['Missing_Percent']
    complete_count = int(row['Complete_Count'])

    if missing_count > 0:
        variables_with_missing += 1
        total_missing += missing_count
        missing_flag = "⚠️"
    else:
        missing_flag = "✓"

    print(f"{var_name:<35} {missing_count:<10} {missing_pct:<8.1f} {complete_count:<10} {missing_flag}")

print(f"\nMissing Data Summary:")
print(f"  Variables with any missing data: {variables_with_missing}/{len(analysis_vars)}")
print(f"  Overall missing data rate: {total_missing/(len(complete_data_df)*len(analysis_vars))*100:.2f}%")

# Step 2: Identify complete cases
print(f"\n🔍 IDENTIFYING COMPLETE CASES")
print("-" * 40)

# Check which patients have complete data on ALL key variables
complete_case_mask = pd.Series(True, index=complete_data_df.index)

for var in analysis_vars:
    if var in complete_data_df.columns:
        complete_case_mask = complete_case_mask & complete_data_df[var].notna()

# Also require valid outcome data
complete_case_mask = complete_case_mask & complete_data_df['has_outcome_data']

complete_case_indices = complete_data_df.index[complete_case_mask]
n_complete_cases = len(complete_case_indices)
complete_case_rate = (n_complete_cases / len(complete_data_df)) * 100

print(f"Patients with complete data: {n_complete_cases:,} ({complete_case_rate:.1f}%)")
print(f"Patients excluded due to missing data: {len(complete_data_df) - n_complete_cases:,}")

if n_complete_cases < 1000:
    print(f"⚠️  Warning: Small complete case sample may limit statistical power")
elif complete_case_rate < 50:
    print(f"⚠️  Warning: <50% complete cases may indicate substantial bias risk")
else:
    print(f"✅ Adequate complete case sample for robust analysis")

# Step 3: Compare characteristics between complete and incomplete cases
print(f"\n📊 COMPARING COMPLETE vs INCOMPLETE CASES")
print("-" * 50)

incomplete_case_indices = complete_data_df.index[~complete_case_mask]
n_incomplete_cases = len(incomplete_case_indices)

if n_incomplete_cases > 0:
    # Compare key demographic characteristics
    demographic_vars = ['physically_restrained', 'race_Asian', 'race_Hispanic_Latino', 'race_Unknown_Declined_Other', 'gender_M']
    available_demo_vars = [var for var in demographic_vars if var in complete_data_df.columns]

    print(f"{'Characteristic':<25} {'Complete Cases':<15} {'Incomplete Cases':<15} {'P-Value'}")
    print("-" * 70)

    comparison_results = []

    for var in available_demo_vars:
        if var in complete_data_df.columns:
            complete_vals = complete_data_df.loc[complete_case_indices, var]
            incomplete_vals = complete_data_df.loc[incomplete_case_indices, var]

            # Remove any remaining missing values for comparison
            complete_vals = complete_vals.dropna()
            incomplete_vals = incomplete_vals.dropna()

            if len(complete_vals) > 0 and len(incomplete_vals) > 0:
                complete_mean = complete_vals.mean()
                incomplete_mean = incomplete_vals.mean()

                # Chi-square test for independence (if binary)
                if set(complete_vals.unique()).issubset({0, 1}) and set(incomplete_vals.unique()).issubset({0, 1}):
                    try:
                        # Create contingency table
                        complete_yes = complete_vals.sum()
                        complete_no = len(complete_vals) - complete_yes
                        incomplete_yes = incomplete_vals.sum()
                        incomplete_no = len(incomplete_vals) - incomplete_yes

                        contingency = np.array([[complete_yes, complete_no],
                                              [incomplete_yes, incomplete_no]])

                        chi2, p_value, _, _ = chi2_contingency(contingency)

                    except:
                        p_value = np.nan
                else:
                    # T-test for continuous variables
                    try:
                        from scipy.stats import ttest_ind
                        _, p_value = ttest_ind(complete_vals, incomplete_vals)
                    except:
                        p_value = np.nan

                var_clean = var.replace('_', ' ').replace('race ', '').title()
                sig_flag = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

                print(f"{var_clean:<25} {complete_mean:<15.3f} {incomplete_mean:<15.3f} {p_value:<10.3f} {sig_flag}")

                comparison_results.append({
                    'variable': var,
                    'complete_mean': complete_mean,
                    'incomplete_mean': incomplete_mean,
                    'p_value': p_value,
                    'significant_diff': p_value < 0.05 if not np.isnan(p_value) else False
                })

    # Summary of differences
    if comparison_results:
        significant_diffs = sum(1 for r in comparison_results if r['significant_diff'])
        print(f"\nCharacteristic differences: {significant_diffs}/{len(comparison_results)} significant")

        if significant_diffs > 0:
            print("⚠️  Systematic differences detected - complete case analysis especially important")
        else:
            print("✅ No major systematic differences between complete/incomplete cases")

# Step 4: Run complete case analysis
print(f"\n📈 RUNNING COMPLETE CASE ANALYSIS")
print("-" * 40)

if n_complete_cases >= 100:  # Minimum threshold for analysis

    # Subset data to complete cases only
    X_complete = X_final_primary.loc[complete_case_indices].copy()

    # Get corresponding y_binomial data
    original_index_map = {idx: pos for pos, idx in enumerate(model_df_aligned_for_diagnostics.index)}
    complete_positions = [original_index_map[idx] for idx in complete_case_indices]
    y_complete = y_binomial[complete_positions]

    print(f"Complete case sample: {len(X_complete):,} patients")

    # Calculate restraint rate in complete cases
    complete_restraint_days = sum(y_complete[:, 0])
    complete_total_days = sum(y_complete.sum(axis=1))
    complete_restraint_rate = complete_restraint_days / complete_total_days

    print(f"Complete case restraint rate: {complete_restraint_rate:.1%}")

    # Check for variables with no variation in complete cases
    X_complete_check = X_complete.drop(columns='const', errors='ignore')
    complete_variances = X_complete_check.var()
    complete_low_var_cols = complete_variances[complete_variances < 1e-9].index.tolist()

    if complete_low_var_cols:
        print(f"Removing zero-variance variables in complete cases: {complete_low_var_cols}")
        X_complete = X_complete.drop(columns=complete_low_var_cols)

    # Fit primary model on complete cases
    print(f"\nFitting primary model on complete cases...")

    try:
        complete_model = sm.GLM(y_complete, X_complete, family=sm.families.Binomial())
        complete_result = complete_model.fit()

        if not complete_result.converged:
            print("⚠️  Warning: Model convergence issues in complete case analysis")

        print(f"✅ Complete case model fitted successfully")
        print(f"   - Converged: {complete_result.converged}")
        print(f"   - Log-likelihood: {complete_result.llf:.1f}")
        print(f"   - AIC: {complete_result.aic:.1f}")

        # Calculate odds ratios for complete case analysis
        complete_odds_ratios = np.exp(complete_result.params)
        complete_conf_int = np.exp(complete_result.conf_int())
        complete_p_values = complete_result.pvalues

        # Create results dataframe
        complete_or_df = pd.DataFrame({
            'Variable': complete_odds_ratios.index,
            'OR_Complete': complete_odds_ratios.values,
            'CI_Lower_Complete': complete_conf_int.iloc[:, 0].values,
            'CI_Upper_Complete': complete_conf_int.iloc[:, 1].values,
            'P_Value_Complete': complete_p_values.values,
            'Significant_Complete': complete_p_values.values < 0.05
        })

        complete_or_df['OR_CI_Complete'] = complete_or_df.apply(
            lambda r: f"{r['OR_Complete']:.2f} ({r['CI_Lower_Complete']:.2f}-{r['CI_Upper_Complete']:.2f})"
            if pd.notna(r['OR_Complete']) else "N/A", axis=1
        )

        # Step 5: Compare complete case results to original analysis
        print(f"\n🔄 COMPARING COMPLETE CASE vs ORIGINAL ANALYSIS")
        print("-" * 55)

        # Focus on key demographic variables
        demographic_vars = ['race_Asian', 'race_Hispanic_Latino', 'race_Unknown_Declined_Other', 'gender_M']
        key_clinical_vars = ['deeply_sedated_day1', 'positive_cam_day1', 'first_day_sofa', 'chemical_restraint_day1']

        comparison_vars = demographic_vars + key_clinical_vars
        available_comparison_vars = [var for var in comparison_vars if var in complete_or_df['Variable'].values]

        if available_comparison_vars:
            print(f"🎯 KEY VARIABLE COMPARISON:")
            print(f"{'Variable':<25} {'Original OR':<15} {'Complete OR':<15} {'Difference':<12} {'Robust'}")
            print("-" * 80)

            robustness_summary = []

            for var in available_comparison_vars:
                complete_row = complete_or_df[complete_or_df['Variable'] == var]

                if not complete_row.empty:
                    complete_or = complete_row.iloc[0]['OR_Complete']
                    complete_sig = complete_row.iloc[0]['Significant_Complete']

                    # Note: You'd need to store original results to compare
                    # For now, we'll note this limitation
                    var_clean = var.replace('race_', '').replace('_', ' ')

                    # Placeholder for original OR (would need to be stored from main analysis)
                    print(f"{var_clean[:24]:<25} {'[See main]':<15} {complete_or:<15.2f} {'[Compare]':<12} {'TBD'}")

                    robustness_summary.append({
                        'variable': var,
                        'complete_or': complete_or,
                        'complete_significant': complete_sig
                    })

            print(f"\nComplete Case Analysis Results:")
            demographic_complete_results = complete_or_df[complete_or_df['Variable'].isin(demographic_vars)]

            if not demographic_complete_results.empty:
                print(f"\n🎯 DEMOGRAPHIC EFFECTS IN COMPLETE CASES:")
                for _, row in demographic_complete_results.iterrows():
                    var_clean = row['Variable'].replace('race_', '').replace('_', ' ')
                    sig_marker = "***" if row['P_Value_Complete'] < 0.001 else "**" if row['P_Value_Complete'] < 0.01 else "*" if row['P_Value_Complete'] < 0.05 else ""

                    print(f"  {var_clean}: {row['OR_CI_Complete']}, p = {row['P_Value_Complete']:.3f} {sig_marker}")

        # Store complete case results
        complete_case_results = {
            'model_result': complete_result,
            'odds_ratios_df': complete_or_df,
            'n_complete_cases': n_complete_cases,
            'complete_case_rate': complete_case_rate,
            'complete_case_indices': complete_case_indices,
            'demographic_results': demographic_complete_results if 'demographic_complete_results' in locals() else None,
            'robustness_assessment': robustness_summary if 'robustness_summary' in locals() else None
        }

    except Exception as e:
        print(f"❌ Complete case analysis failed: {e}")
        complete_case_results = None

else:
    print(f"❌ Insufficient complete cases ({n_complete_cases}) for reliable analysis")
    print("   Minimum threshold: 100 patients")
    complete_case_results = None

# Step 6: Sensitivity Analysis Summary
print(f"\n{'='*80}")
print("COMPLETE CASE ANALYSIS SUMMARY")
print("="*80)

if complete_case_results is not None:
    print(f"✅ SUCCESSFUL COMPLETE CASE ANALYSIS")
    print(f"   Sample size: {n_complete_cases:,} patients ({complete_case_rate:.1f}% of original)")
    print(f"   Model convergence: {complete_case_results['model_result'].converged}")

    if complete_case_results['demographic_results'] is not None:
        n_demo_effects = sum(complete_case_results['demographic_results']['Significant_Complete'])
        print(f"   Significant demographic effects: {n_demo_effects}")

    print(f"\n📋 INTERPRETATION GUIDANCE:")
    print(f"   • If effects similar to main analysis → Robust to missing data")
    print(f"   • If effects differ substantially → Missing data may bias results")
    print(f"   • If effects disappear → Findings may be driven by missing data patterns")

    print(f"\n📊 MANUSCRIPT REPORTING:")
    print(f"   'Complete case analysis on {n_complete_cases:,} patients ({complete_case_rate:.1f}%)")
    print(f"    with no missing data yielded [similar/different] results to the main analysis,'")
    print(f"    [supporting/questioning] the robustness of our findings to missing data patterns.'")

else:
    print(f"❌ COMPLETE CASE ANALYSIS NOT FEASIBLE")
    print(f"   Reason: Insufficient complete cases ({n_complete_cases})")
    print(f"   Recommendation: Document as limitation in manuscript")

print(f"\n✅ COMPLETE CASE ANALYSIS COMPLETE")
print("📊 Results stored in 'complete_case_results' (if successful)")
print("🔄 Ready for temporal trend analysis (final protocol step)")

# Optional: Brief missing data pattern visualization
if variables_with_missing > 0:
    print(f"\n📈 MISSING DATA PATTERN NOTE:")
    print(f"   Consider reporting missing data patterns in supplementary material")
    print(f"   Variables with missing data: {variables_with_missing}/{len(analysis_vars)}")
