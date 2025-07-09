import pandas as pd

correct_file_path = '/content/drive/MyDrive/bq-results-20250505-135342-1746453237283/bq-results-20250505-135342-1746453237283.csv'

try:
    df = pd.read_csv(correct_file_path)
    print("Successfully loaded the DataFrame!")
    print(df.head()) # Display the first few rows to confirm
except FileNotFoundError:
    print(f"Error: File not found at path: {correct_file_path}")
    print("Please verify the path and filename again using the directory listing from Step 2.")
except Exception as e:
    print(f"An error occurred while reading the CSV: {e}")

icu_counts = df['icu_type'].value_counts()
rare_icus = icu_counts[icu_counts < 100].index
df['icu_type'] = df['icu_type'].apply(lambda x: 'other' if x in rare_icus else x)
print("\nICU type frequencies after filtering:")
print(df['icu_type'].value_counts())


if 'length_of_stay_calendar_days' in df.columns:
    df = df.rename(columns={'length_of_stay_calendar_days': 'length_of_stay'})
    print("Column renamed successfully!")
    print(df.head())
else:
    print("Warning: Column 'cohort_year' not found in the DataFrame.")
    print(df.head())

if 'cohort_year' in df.columns:
    df = df.rename(columns={'cohort_year': 'time_period'})
    print("Column renamed successfully!")
    print(df.head())
else:
    print("Warning: Column 'cohort_year' not found in the DataFrame.")
    print(df.head())

import numpy as np # Import numpy for np.where, often useful with pandas

# --- Apply Language Grouping ---
print("\nGrouping 'language' categories into 'English' and 'Other'...")

df['language'] = np.where(df['language'].str.contains('ENGL', case=False, na=False),
                          'English',
                          'Other')

print("\nValue counts for final updated 'language' column:")
print(df['language'].value_counts())
# --- End Language Grouping Block ---

# --- Apply Race Grouping ONCE ---
print("\nGrouping 'race' categories...")

# First, let's examine what values are actually in the dataset
print("\nSample of unique race values in the original dataset:")
unique_races = df['race'].unique()
print(unique_races[:20])  # Print first 20 unique values to see the format

# Check if there are any formatting issues (whitespace, case, etc.)
print("\nChecking for potential formatting issues...")
print("Count of values with leading/trailing spaces:",
      sum(df['race'].astype(str).str.strip() != df['race'].astype(str)))

mapping_dict = {
    # White variations
    'WHITE': 'White', 'WHITE - RUSSIAN': 'White', 'WHITE - OTHER EUROPEAN': 'White',
    'WHITE - EASTERN EUROPEAN': 'White', 'PORTUGUESE': 'White',
    # Black variations
    'BLACK/AFRICAN AMERICAN': 'Black/African American', 'BLACK/CARIBBEAN ISLAND': 'Black/African American',
    'BLACK/AFRICAN': 'Black/African American', 'BLACK/CAPE VERDEAN': 'Black/African American',
    # Hispanic/Latino variations
    'HISPANIC OR LATINO': 'Hispanic/Latino', 'HISPANIC/LATINO - CUBAN': 'Hispanic/Latino',
    'HISPANIC/LATINO - PUERTO RICAN': 'Hispanic/Latino', 'HISPANIC/LATINO - DOMINICAN': 'Hispanic/Latino',
    'HISPANIC/LATINO - CENTRAL AMERICAN': 'Hispanic/Latino', 'HISPANIC/LATINO - GUATEMALAN': 'Hispanic/Latino',
    'HISPANIC/LATINO - MEXICAN': 'Hispanic/Latino', 'HISPANIC/LATINO - COLUMBIAN': 'Hispanic/Latino',
    'HISPANIC/LATINO - HONDURAN': 'Hispanic/Latino', 'HISPANIC/LATINO - SALVADORAN': 'Hispanic/Latino',
    'SOUTH AMERICAN': 'Hispanic/Latino', 'WHITE - BRAZILIAN': 'Hispanic/Latino',
    # Asian variations
    'ASIAN': 'Asian', 'ASIAN - CHINESE': 'Asian', 'ASIAN - ASIAN INDIAN': 'Asian',
    'ASIAN - KOREAN': 'Asian', 'ASIAN - SOUTH EAST ASIAN': 'Asian',
    # Moving these to Unknown/Declined/Other as requested
    'AMERICAN INDIAN/ALASKA NATIVE': 'Unknown/Declined/Other',
    'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'Unknown/Declined/Other',
    'MULTIPLE RACE/ETHNICITY': 'Unknown/Declined/Other',
    # Unknown / Other / Declined
    'UNKNOWN': 'Unknown/Declined/Other', 'OTHER': 'Unknown/Declined/Other',
    'PATIENT DECLINED TO ANSWER': 'Unknown/Declined/Other', 'UNABLE TO OBTAIN': 'Unknown/Declined/Other',
}

# A more robust approach - store the original values
df['original_race'] = df['race'].copy()

# Apply the mapping with additional safeguards
# Convert keys to uppercase for case insensitivity
case_insensitive_dict = {k.upper().strip(): v for k, v in mapping_dict.items()}

# Apply mapping with case normalization
df['race'] = df['race'].astype(str).str.upper().str.strip().map(case_insensitive_dict).fillna('Unknown/Declined/Other')

print("\nValue counts for final updated 'race' column:")
print(df['race'].value_counts())

# Verify which original values were not in the mapping dictionary
if 'Unknown/Declined/Other' in df['race'].unique():
    unknown_count = (df['race'] == 'Unknown/Declined/Other').sum()
    if unknown_count > 0:
        print(f"\n{unknown_count} values were mapped to 'Unknown/Declined/Other'")

        # Find the top original values that got mapped to unknown
        print("\nTop original values that got mapped to 'Unknown/Declined/Other':")
        unknown_origins = df[df['race'] == 'Unknown/Declined/Other']['original_race'].value_counts().head(10)
        print(unknown_origins)
# --- End Race Grouping Block ---



# Add blank row after Language
import pandas as pd
import numpy as np

# Assume df is your DataFrame, loaded elsewhere

# --- Configuration ---
strata_variable = 'physically_restrained'
strata_labels = {0: 'Not Restrained', 1: 'Restrained'}

continuous_vars_all = [
    'age', 'first_day_sofa', 'first_day_oasis', 'first_day_sapsii',
    'max_nrs_pain_day1', 'max_cpot_day1', 'max_ciwa_day1'
]

severity_scores_continuous = [
    'first_day_sofa', 'first_day_oasis', 'first_day_sapsii',
    'max_nrs_pain_day1', 'max_cpot_day1', 'max_ciwa_day1'
]

categorical_vars_all = ['gender', 'race', 'icu_type', 'admission_source', 'time_period', 'language']
los_variable = 'length_of_stay'
los_cat_variable = 'los_category'

binary_vars = [
    'ventilated_day1', 'niv_day1', 'has_cline_day1', 'has_aline_day1',
    'has_dline_day1', 'has_evd_day1', 'has_ecmo_day1', 'agitated_day1',
    'deeply_sedated_day1', 'admitted_with_suicide_attempt_dx',
    'chemical_restraint_day1', 'nrs_recorded_day1', 'cpot_recorded_day1',
    'ciwa_recorded_day1', 'has_psychosis_admission_dx',
    'has_mania_bipolar_admission_dx', 'has_substance_use_admission_dx',
    'has_phys_condition_mental_disorder_admission_dx', 'death_within_24h',
    'positive_cam_day1'
]
categorical_vars_all.extend(binary_vars)

# Define variable groups for subheadings
ventilation_vars = ['ventilated_day1', 'niv_day1']
lines_vars = ['has_cline_day1', 'has_aline_day1', 'has_dline_day1']
life_support_vars = ['has_evd_day1', 'has_ecmo_day1']
monitoring_vars = ['agitated_day1', 'deeply_sedated_day1', 'chemical_restraint_day1', 'death_within_24h']
clinical_assessment_vars = ['nrs_recorded_day1', 'cpot_recorded_day1', 'ciwa_recorded_day1']
severity_score_vars = ['first_day_sofa', 'first_day_oasis', 'first_day_sapsii',
                      'max_nrs_pain_day1', 'max_cpot_day1', 'max_ciwa_day1']

# Define mental health related variables for grouped presentation
mental_health_vars = [
    'admitted_with_suicide_attempt_dx',
    'has_psychosis_admission_dx',
    'has_mania_bipolar_admission_dx',
    'has_substance_use_admission_dx',
    'has_phys_condition_mental_disorder_admission_dx',
    'positive_cam_day1'
]

display_names = {
    'age': 'Age',
    'age_group': 'Age Group',
    'gender': 'Gender', 'race': 'Race', 'language': 'Language', 'icu_type': 'ICU Type',
    'admission_source': 'Admission Source', 'time_period': 'Time Period',
    los_cat_variable: 'Length of Stay (days)',
    'ventilated_day1': 'Invasive Ventilation', 'niv_day1': 'Non-Invasive Ventilation',
    'has_cline_day1': 'Central Line', 'has_aline_day1': 'Arterial Line',
    'has_dline_day1': 'Dialysis Line', 'has_evd_day1': 'EVD', 'has_ecmo_day1': 'ECMO',
    'first_day_sofa': 'SOFA Score', 'first_day_oasis': 'OASIS Score', 'first_day_sapsii': 'SAPS-II Score',
    'agitated_day1': 'Agitated', 'deeply_sedated_day1': 'Deeply Sedated',
    'admitted_with_suicide_attempt_dx': 'Suicide Attempt',
    'chemical_restraint_day1': 'Chemical Restraint',
    'nrs_recorded_day1': 'NRS Recorded', 'max_nrs_pain_day1': 'Max NRS Score',
    'cpot_recorded_day1': 'CPOT Recorded', 'max_cpot_day1': 'MAX CPOT Score',
    'ciwa_recorded_day1': 'CIWA Recorded', 'max_ciwa_day1': 'Max CIWA Score',
    'has_psychosis_admission_dx': 'Psychosis',
    'has_mania_bipolar_admission_dx': 'Bipolar/Mania',
    'has_substance_use_admission_dx': 'Substance Use',
    'has_phys_condition_mental_disorder_admission_dx': 'Mental Disorder due to Phys Condition',
    'death_within_24h': 'Death within 24h of first restraint',
    'positive_cam_day1': 'Delirious (Positive CAM Score)'
}

# --- Pre-processing ---
df_analysis = df.dropna(subset=[strata_variable]).copy()
if pd.api.types.is_numeric_dtype(df_analysis[strata_variable]):
    df_analysis[strata_variable] = df_analysis[strata_variable].astype(int)

# LOS binning
if los_variable in df_analysis.columns:
    bins = [0, 3, 6, 10, 50, np.inf]
    labels = ['1-3 days', '4-6 days', '7-10 days', '11-50 days', '>50 days']
    df_analysis[los_cat_variable] = pd.cut(df_analysis[los_variable], bins=bins, labels=labels)
    if los_cat_variable not in categorical_vars_all:
        categorical_vars_all.append(los_cat_variable)
    display_names[los_cat_variable] = 'Length of Stay (days)'

# Age binning
if 'age' in df_analysis.columns:
    age_bins = [17, 30, 40, 60, 75, np.inf]
    age_labels = ['18–30', '31–40', '41–60', '61–75', '76+']
    df_analysis['age_group'] = pd.cut(df_analysis['age'], bins=age_bins, labels=age_labels)
    if 'age_group' not in categorical_vars_all:
        categorical_vars_all.append('age_group')

# --- Table 1 generation ---
table = []

# Total N row
table.append({"Characteristic": "N", "Overall": len(df_analysis),
              **{label: (df_analysis[strata_variable] == val).sum() for val, label in strata_labels.items()}})

# Add TWO blank rows after N
table.append({"Characteristic": "", "Overall": "", **{label: "" for label in strata_labels.values()}})
table.append({"Characteristic": "", "Overall": "", **{label: "" for label in strata_labels.values()}})

# ----- AGE SECTION -----
# Add Age (continuous variable)
if 'age' in df_analysis.columns:
    name = display_names.get('age', 'Age')
    row = {"Characteristic": f"{name}, mean (SD)"}
    row["Overall"] = f"{df_analysis['age'].mean():.1f} ({df_analysis['age'].std():.1f})"
    for val, label in strata_labels.items():
        subset = df_analysis[df_analysis[strata_variable] == val]['age'].dropna()
        row[label] = f"{subset.mean():.1f} ({subset.std():.1f})" if not subset.empty else ""
    table.append(row)

# Add blank row after Age
table.append({"Characteristic": "", "Overall": "", **{label: "" for label in strata_labels.values()}})

# ----- AGE GROUP SECTION -----
# Add Age Group
if 'age_group' in df_analysis.columns:
    name = display_names.get('age_group', 'Age Group')
    table.append({"Characteristic": f"{name}", "Overall": "", **{label: "" for label in strata_labels.values()}})
    ctab = pd.crosstab(df_analysis['age_group'], df_analysis[strata_variable])
    for level in ctab.index:
        overall = ctab.loc[level].sum()
        row = {"Characteristic": f"  {str(level)}", "Overall": f"{overall}"}
        for val, label in strata_labels.items():
            count = ctab.loc[level, val] if val in ctab.columns else 0
            pct = (count / overall * 100) if overall > 0 else 0
            row[label] = f"{count} ({pct:.1f}%)"
        table.append(row)

# Add blank row after Age Group
table.append({"Characteristic": "", "Overall": "", **{label: "" for label in strata_labels.values()}})

# ----- GENDER SECTION -----
# Add Gender
if 'gender' in df_analysis.columns:
    name = display_names.get('gender', 'Gender')
    table.append({"Characteristic": f"{name}", "Overall": "", **{label: "" for label in strata_labels.values()}})
    ctab = pd.crosstab(df_analysis['gender'], df_analysis[strata_variable])
    for level in ctab.index:
        overall = ctab.loc[level].sum()
        row = {"Characteristic": f"  {str(level)}", "Overall": f"{overall}"}
        for val, label in strata_labels.items():
            count = ctab.loc[level, val] if val in ctab.columns else 0
            pct = (count / overall * 100) if overall > 0 else 0
            row[label] = f"{count} ({pct:.1f}%)"
        table.append(row)

# Add blank row after Gender
table.append({"Characteristic": "", "Overall": "", **{label: "" for label in strata_labels.values()}})

# ----- LENGTH OF STAY SECTION -----
# Add Length of Stay
if los_cat_variable in df_analysis.columns:
    name = display_names.get(los_cat_variable, 'Length of Stay (days)')
    table.append({"Characteristic": f"{name}", "Overall": "", **{label: "" for label in strata_labels.values()}})
    ctab = pd.crosstab(df_analysis[los_cat_variable], df_analysis[strata_variable])
    for level in ctab.index:
        overall = ctab.loc[level].sum()
        row = {"Characteristic": f"  {str(level)}", "Overall": f"{overall}"}
        for val, label in strata_labels.items():
            count = ctab.loc[level, val] if val in ctab.columns else 0
            pct = (count / overall * 100) if overall > 0 else 0
            row[label] = f"{count} ({pct:.1f}%)"
        table.append(row)

# Add blank row after Length of Stay
table.append({"Characteristic": "", "Overall": "", **{label: "" for label in strata_labels.values()}})

# ----- RACE SECTION -----
# Add Race
if 'race' in df_analysis.columns:
    name = display_names.get('race', 'Race')
    table.append({"Characteristic": f"{name}", "Overall": "", **{label: "" for label in strata_labels.values()}})
    ctab = pd.crosstab(df_analysis['race'], df_analysis[strata_variable])
    for level in ctab.index:
        overall = ctab.loc[level].sum()
        row = {"Characteristic": f"  {str(level)}", "Overall": f"{overall}"}
        for val, label in strata_labels.items():
            count = ctab.loc[level, val] if val in ctab.columns else 0
            pct = (count / overall * 100) if overall > 0 else 0
            row[label] = f"{count} ({pct:.1f}%)"
        table.append(row)

# Add blank row after Race
table.append({"Characteristic": "", "Overall": "", **{label: "" for label in strata_labels.values()}})

# ----- LANGUAGE SECTION -----
# Add Language
if 'language' in df_analysis.columns:
    name = display_names.get('language', 'Language')
    table.append({"Characteristic": f"{name}", "Overall": "", **{label: "" for label in strata_labels.values()}})
    ctab = pd.crosstab(df_analysis['language'], df_analysis[strata_variable])
    for level in ctab.index:
        overall = ctab.loc[level].sum()
        row = {"Characteristic": f"  {str(level)}", "Overall": f"{overall}"}
        for val, label in strata_labels.items():
            count = ctab.loc[level, val] if val in ctab.columns else 0
            pct = (count / overall * 100) if overall > 0 else 0
            row[label] = f"{count} ({pct:.1f}%)"
        table.append(row)

# Add blank row after Language
table.append({"Characteristic": "", "Overall": "", **{label: "" for label in strata_labels.values()}})

# ----- ICU TYPE SECTION -----
# Add ICU Type
if 'icu_type' in df_analysis.columns:
    name = display_names.get('icu_type', 'ICU Type')
    table.append({"Characteristic": f"{name}", "Overall": "", **{label: "" for label in strata_labels.values()}})
    ctab = pd.crosstab(df_analysis['icu_type'], df_analysis[strata_variable])
    for level in ctab.index:
        overall = ctab.loc[level].sum()
        row = {"Characteristic": f"  {str(level)}", "Overall": f"{overall}"}
        for val, label in strata_labels.items():
            count = ctab.loc[level, val] if val in ctab.columns else 0
            pct = (count / overall * 100) if overall > 0 else 0
            row[label] = f"{count} ({pct:.1f}%)"
        table.append(row)

# Add blank row after ICU Type
table.append({"Characteristic": "", "Overall": "", **{label: "" for label in strata_labels.values()}})

# ----- ADMISSION SOURCE SECTION -----
# Add Admission Source
if 'admission_source' in df_analysis.columns:
    name = display_names.get('admission_source', 'Admission Source')
    table.append({"Characteristic": f"{name}", "Overall": "", **{label: "" for label in strata_labels.values()}})
    ctab = pd.crosstab(df_analysis['admission_source'], df_analysis[strata_variable])
    for level in ctab.index:
        overall = ctab.loc[level].sum()
        row = {"Characteristic": f"  {str(level)}", "Overall": f"{overall}"}
        for val, label in strata_labels.items():
            count = ctab.loc[level, val] if val in ctab.columns else 0
            pct = (count / overall * 100) if overall > 0 else 0
            row[label] = f"{count} ({pct:.1f}%)"
        table.append(row)

# Add blank row after Admission Source
table.append({"Characteristic": "", "Overall": "", **{label: "" for label in strata_labels.values()}})

# ----- TIME PERIOD SECTION -----
# Add Time Period
if 'time_period' in df_analysis.columns:
    name = display_names.get('time_period', 'Time Period')
    table.append({"Characteristic": f"{name}", "Overall": "", **{label: "" for label in strata_labels.values()}})
    ctab = pd.crosstab(df_analysis['time_period'], df_analysis[strata_variable])
    for level in ctab.index:
        overall = ctab.loc[level].sum()
        row = {"Characteristic": f"  {str(level)}", "Overall": f"{overall}"}
        for val, label in strata_labels.items():
            count = ctab.loc[level, val] if val in ctab.columns else 0
            pct = (count / overall * 100) if overall > 0 else 0
            row[label] = f"{count} ({pct:.1f}%)"
        table.append(row)

# Add blank row after Time Period
table.append({"Characteristic": "", "Overall": "", **{label: "" for label in strata_labels.values()}})

# Add TWO blank rows before the next section
table.append({"Characteristic": "", "Overall": "", **{label: "" for label in strata_labels.values()}})

# Add Ventilation section
table.append({"Characteristic": "Ventilation", "Overall": "", **{label: "" for label in strata_labels.values()}})
for var in ventilation_vars:
    if var in df_analysis.columns:
        name = display_names.get(var, var.replace('_', ' ').capitalize())
        ctab = pd.crosstab(df_analysis[var], df_analysis[strata_variable])
        if 1 in ctab.index:
            overall_yes = ctab.loc[1].sum()
            overall_total = ctab.sum().sum()
            overall_pct = (overall_yes / overall_total * 100) if overall_total > 0 else 0

            row = {"Characteristic": f"  {name}", "Overall": f"{overall_yes} ({overall_pct:.1f}%)"}

            for val, label in strata_labels.items():
                yes_count = ctab.loc[1, val] if val in ctab.columns and 1 in ctab.index else 0
                strata_total = (df_analysis[strata_variable] == val).sum()
                strata_pct = (yes_count / strata_total * 100) if strata_total > 0 else 0
                row[label] = f"{yes_count} ({strata_pct:.1f}%)"

            table.append(row)

# Add TWO blank rows before the next section
table.append({"Characteristic": "", "Overall": "", **{label: "" for label in strata_labels.values()}})
table.append({"Characteristic": "", "Overall": "", **{label: "" for label in strata_labels.values()}})

# Add Lines on the Body section
table.append({"Characteristic": "Lines on the Body", "Overall": "", **{label: "" for label in strata_labels.values()}})
for var in lines_vars:
    if var in df_analysis.columns:
        name = display_names.get(var, var.replace('_', ' ').capitalize())
        ctab = pd.crosstab(df_analysis[var], df_analysis[strata_variable])
        if 1 in ctab.index:
            overall_yes = ctab.loc[1].sum()
            overall_total = ctab.sum().sum()
            overall_pct = (overall_yes / overall_total * 100) if overall_total > 0 else 0

            row = {"Characteristic": f"  {name}", "Overall": f"{overall_yes} ({overall_pct:.1f}%)"}

            for val, label in strata_labels.items():
                yes_count = ctab.loc[1, val] if val in ctab.columns and 1 in ctab.index else 0
                strata_total = (df_analysis[strata_variable] == val).sum()
                strata_pct = (yes_count / strata_total * 100) if strata_total > 0 else 0
                row[label] = f"{yes_count} ({strata_pct:.1f}%)"

            table.append(row)

# Add TWO blank rows before the next section
table.append({"Characteristic": "", "Overall": "", **{label: "" for label in strata_labels.values()}})
table.append({"Characteristic": "", "Overall": "", **{label: "" for label in strata_labels.values()}})

# Add Life Support section
table.append({"Characteristic": "Life Support", "Overall": "", **{label: "" for label in strata_labels.values()}})
for var in life_support_vars:
    if var in df_analysis.columns:
        name = display_names.get(var, var.replace('_', ' ').capitalize())
        ctab = pd.crosstab(df_analysis[var], df_analysis[strata_variable])
        if 1 in ctab.index:
            overall_yes = ctab.loc[1].sum()
            overall_total = ctab.sum().sum()
            overall_pct = (overall_yes / overall_total * 100) if overall_total > 0 else 0

            row = {"Characteristic": f"  {name}", "Overall": f"{overall_yes} ({overall_pct:.1f}%)"}

            for val, label in strata_labels.items():
                yes_count = ctab.loc[1, val] if val in ctab.columns and 1 in ctab.index else 0
                strata_total = (df_analysis[strata_variable] == val).sum()
                strata_pct = (yes_count / strata_total * 100) if strata_total > 0 else 0
                row[label] = f"{yes_count} ({strata_pct:.1f}%)"

            table.append(row)

# Add TWO blank rows before the next section
table.append({"Characteristic": "", "Overall": "", **{label: "" for label in strata_labels.values()}})
table.append({"Characteristic": "", "Overall": "", **{label: "" for label in strata_labels.values()}})

# Add Monitoring and Status section
table.append({"Characteristic": "Monitoring and Status", "Overall": "", **{label: "" for label in strata_labels.values()}})
for var in monitoring_vars:
    if var in df_analysis.columns:
        name = display_names.get(var, var.replace('_', ' ').capitalize())
        ctab = pd.crosstab(df_analysis[var], df_analysis[strata_variable])
        if 1 in ctab.index:
            overall_yes = ctab.loc[1].sum()
            overall_total = ctab.sum().sum()
            overall_pct = (overall_yes / overall_total * 100) if overall_total > 0 else 0

            row = {"Characteristic": f"  {name}", "Overall": f"{overall_yes} ({overall_pct:.1f}%)"}

            for val, label in strata_labels.items():
                yes_count = ctab.loc[1, val] if val in ctab.columns and 1 in ctab.index else 0
                strata_total = (df_analysis[strata_variable] == val).sum()
                strata_pct = (yes_count / strata_total * 100) if strata_total > 0 else 0
                row[label] = f"{yes_count} ({strata_pct:.1f}%)"

            table.append(row)

# Add TWO blank rows before the next section
table.append({"Characteristic": "", "Overall": "", **{label: "" for label in strata_labels.values()}})
table.append({"Characteristic": "", "Overall": "", **{label: "" for label in strata_labels.values()}})

# Add Clinical Assessment section
table.append({"Characteristic": "Clinical Assessment and Severity Scores", "Overall": "", **{label: "" for label in strata_labels.values()}})

# First add the binary recording variables
for var in clinical_assessment_vars:
    if var in df_analysis.columns:
        name = display_names.get(var, var.replace('_', ' ').capitalize())
        ctab = pd.crosstab(df_analysis[var], df_analysis[strata_variable])
        if 1 in ctab.index:
            overall_yes = ctab.loc[1].sum()
            overall_total = ctab.sum().sum()
            overall_pct = (overall_yes / overall_total * 100) if overall_total > 0 else 0

            row = {"Characteristic": f"  {name}", "Overall": f"{overall_yes} ({overall_pct:.1f}%)"}

            for val, label in strata_labels.items():
                yes_count = ctab.loc[1, val] if val in ctab.columns and 1 in ctab.index else 0
                strata_total = (df_analysis[strata_variable] == val).sum()
                strata_pct = (yes_count / strata_total * 100) if strata_total > 0 else 0
                row[label] = f"{yes_count} ({strata_pct:.1f}%)"

            table.append(row)

# Then add the continuous severity scores
for var in severity_score_vars:
    if var in df_analysis.columns:
        name = display_names.get(var, var.replace('_', ' ').capitalize())
        row = {"Characteristic": f"  {name}, mean (SD)"}
        row["Overall"] = f"{df_analysis[var].mean():.1f} ({df_analysis[var].std():.1f})"
        for val, label in strata_labels.items():
            subset = df_analysis[df_analysis[strata_variable] == val][var].dropna()
            row[label] = f"{subset.mean():.1f} ({subset.std():.1f})" if not subset.empty else ""
        table.append(row)

# Add TWO blank rows before the next section
table.append({"Characteristic": "", "Overall": "", **{label: "" for label in strata_labels.values()}})
table.append({"Characteristic": "", "Overall": "", **{label: "" for label in strata_labels.values()}})

# Add the severe mental health subheading and its variables
table.append({"Characteristic": "Severe Mental Health", "Overall": "", **{label: "" for label in strata_labels.values()}})

# Process mental health variables
for var in mental_health_vars:
    if var in df_analysis.columns:
        name = display_names.get(var, var.replace('_', ' ').capitalize())
        if var in binary_vars:
            # For binary variables, only show the "Yes" row
            ctab = pd.crosstab(df_analysis[var], df_analysis[strata_variable])
            # Check if 1 is in the index (representing "Yes")
            if 1 in ctab.index:
                overall_yes = ctab.loc[1].sum()
                overall_total = ctab.sum().sum()
                overall_pct = (overall_yes / overall_total * 100) if overall_total > 0 else 0

                row = {"Characteristic": f"  {name}", "Overall": f"{overall_yes} ({overall_pct:.1f}%)"}

                for val, label in strata_labels.items():
                    yes_count = ctab.loc[1, val] if val in ctab.columns and 1 in ctab.index else 0
                    strata_total = (df_analysis[strata_variable] == val).sum()
                    strata_pct = (yes_count / strata_total * 100) if strata_total > 0 else 0
                    row[label] = f"{yes_count} ({strata_pct:.1f}%)"

                table.append(row)
        else:
            # For non-binary categorical variables, show all levels
            table.append({"Characteristic": f"  {name}", "Overall": "", **{label: "" for label in strata_labels.values()}})
            ctab = pd.crosstab(df_analysis[var], df_analysis[strata_variable])
            for level in ctab.index:
                overall = ctab.loc[level].sum()
                row = {"Characteristic": f"    {str(level)}", "Overall": f"{overall}"}
                for val, label in strata_labels.items():
                    count = ctab.loc[level, val] if val in ctab.columns else 0
                    pct = (count / overall * 100) if overall > 0 else 0
                    row[label] = f"{count} ({pct:.1f}%)"
                table.append(row)

# --- Output ---
table1_df = pd.DataFrame(table)
print("Table 1: Summary Characteristics Stratified by Physical Restraint Use")
print(table1_df.to_string(index=False))
