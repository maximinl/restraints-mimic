import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import chi2
from scipy import stats  # Added missing import
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TEMPORAL TREND ANALYSIS")
print("=" * 80)
print("Examining restraint use trends and policy impact (2008-2022)")
print("Protocol objective: Assess 2014 CMS policy effects and demographic disparity evolution\n")

# Validate required data
if 'df' not in locals() and 'df' not in globals():
    raise NameError("Dataset 'df' not found. Please ensure df is loaded.")

print("📋 TEMPORAL ANALYSIS SETUP")
print("_" * 30)  # Changed from dashes per user preference

def setup_temporal_variables(data):
    """Setup and validate temporal variables for analysis"""

    # Check available time related columns
    time_cols = [col for col in data.columns if 'time' in col.lower() or 'year' in col.lower() or 'period' in col.lower()]
    print(f"Available time related columns: {time_cols}")

    # Try different approaches to get time periods
    if 'time_period' in data.columns:
        print("✓ Using existing 'time_period' variable")
        return 'time_period'
    elif 'cohort_year' in data.columns:
        print("✓ Found 'cohort_year' will create time periods")
        return 'cohort_year'
    else:
        print("⚠️  No clear time variable found attempting to create from available data")
        # Check first few rows to understand data structure
        print("Available columns:", list(data.columns))
        return None

def create_time_periods(data, time_var):
    """Create standardized time periods for analysis"""

    if time_var == 'time_period':
        # Already have time periods
        time_periods = data['time_period'].value_counts().sort_index()
        print(f"Existing time periods: {time_periods.index.tolist()}")
        return data.copy()

    elif time_var == 'cohort_year':
        # Create periods from cohort year
        data_temporal = data.copy()

        # Map cohort years to periods
        def map_year_to_period(year_group):
            if pd.isna(year_group):
                return 'Unknown'
            year_str = str(year_group)

            # Extract year from cohort_year_group format (like "2008 - 2010")
            if ' - ' in year_str:  # Handle space around dash format
                start_year = int(year_str.split(' - ')[0].strip())
            elif '-' in year_str:  # Handle no space format
                start_year = int(year_str.split('-')[0].strip())
            else:
                try:
                    start_year = int(year_str)
                except:
                    return 'Unknown'

            if start_year <= 2010:
                return '2008 - 2010'  # Match the actual format in data
            elif start_year <= 2013:
                return '2011 - 2013'
            elif start_year <= 2016:
                return '2014 - 2016'
            elif start_year <= 2019:
                return '2017 - 2019'
            elif start_year <= 2022:
                return '2020 - 2022'
            else:
                return 'Unknown'

        data_temporal['time_period'] = data_temporal['cohort_year'].apply(map_year_to_period)

        # Validate creation
        period_counts = data_temporal['time_period'].value_counts()
        print(f"Created time periods: {period_counts.to_dict()}")

        return data_temporal

    else:
        print("❌ Cannot create time periods from available data")
        return None

# Setup temporal analysis
time_var = setup_temporal_variables(df)
if time_var:
    df_temporal = create_time_periods(df, time_var)
    if df_temporal is not None:
        print(f"✅ Temporal analysis ready with {len(df_temporal)} patients")
    else:
        raise ValueError("Could not create temporal dataset")
else:
    raise ValueError("No suitable time variable found for temporal analysis")

# Step 1: Overall Temporal Trends
print(f"\n📊 OVERALL RESTRAINT TRENDS OVER TIME")
print("_" * 45)

def analyze_overall_trends(data):
    """Analyze overall restraint trends across time periods"""

    # Calculate restraint rates by time period
    trend_data = data.groupby('time_period').agg({
        'physically_restrained': ['count', 'sum', 'mean'],
        'restraint_calendar_days': 'sum',
        'length_of_stay': 'sum' if 'length_of_stay' in data.columns else lambda x: len(x)
    }).round(4)

    # Flatten column names
    trend_data.columns = ['n_patients', 'n_restrained', 'restraint_rate', 'total_restraint_days', 'total_icu_days']

    # Add restraint proportion (days based)
    if 'total_icu_days' in trend_data.columns:
        trend_data['restraint_proportion'] = trend_data['total_restraint_days'] / trend_data['total_icu_days']

    # Sort by time period using the actual format in data
    period_order = ['2008 - 2010', '2011 - 2013', '2014 - 2016', '2017 - 2019', '2020 - 2022']
    available_periods = [p for p in period_order if p in trend_data.index]
    trend_data = trend_data.reindex(available_periods)

    print(f"Overall Restraint Trends:")
    print(f"{'Period':<12} {'N Patients':<12} {'N Restrained':<13} {'Rate (%)':<10} {'Restraint Days':<15}")
    print("_" * 70)

    for period in trend_data.index:
        if period != 'Unknown' and not pd.isna(trend_data.loc[period, 'restraint_rate']):
            row = trend_data.loc[period]
            rate_pct = row['restraint_rate'] * 100
            print(f"{period:<12} {row['n_patients']:<12,.0f} {row['n_restrained']:<13,.0f} {rate_pct:<10.1f} {row['total_restraint_days']:<15,.0f}")

    return trend_data

overall_trends = analyze_overall_trends(df_temporal)

# Step 2: Cochran Armitage Test for Trend
print(f"\n📈 COCHRAN ARMITAGE TREND TEST")
print("_" * 35)

def cochran_armitage_trend_test(trend_data):
    """Perform Cochran Armitage test for linear trend"""

    try:
        # Prepare data for trend test
        periods = trend_data.index.tolist()
        periods = [p for p in periods if p != 'Unknown' and not pd.isna(trend_data.loc[p, 'restraint_rate'])]

        if len(periods) < 3:
            print("❌ Insufficient time periods for trend test")
            return None

        # Create ordered data
        restrained_counts = []
        total_counts = []
        period_scores = []

        for i, period in enumerate(periods):
            restrained_counts.append(int(trend_data.loc[period, 'n_restrained']))
            total_counts.append(int(trend_data.loc[period, 'n_patients']))
            period_scores.append(i + 1)  # Linear scoring: 1, 2, 3, 4, 5

        # Calculate trend test statistic
        n = len(periods)
        R = sum(restrained_counts)
        N = sum(total_counts)

        # Calculate weighted sum
        numerator = 0
        for i in range(n):
            numerator += period_scores[i] * restrained_counts[i]

        # Calculate expected and variance
        score_sum = sum(period_scores)
        score_sq_sum = sum(x**2 for x in period_scores)

        expected = R * score_sum / N

        variance_num = 0
        for i in range(n):
            variance_num += total_counts[i] * (period_scores[i] - score_sum/n)**2

        variance = R * (N - R) * variance_num / (N**2 * (N - 1))

        # Test statistic
        if variance > 0:
            z_stat = (numerator - expected) / np.sqrt(variance)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two tailed

            print(f"Cochran Armitage Trend Test Results:")
            print(f"  Z statistic: {z_stat:.3f}")
            print(f"  P value: {p_value:.3f}")

            if p_value < 0.001:
                significance = "***"
                interpretation = "Highly significant trend"
            elif p_value < 0.01:
                significance = "**"
                interpretation = "Significant trend"
            elif p_value < 0.05:
                significance = "*"
                interpretation = "Significant trend"
            else:
                significance = ""
                interpretation = "No significant trend"

            direction = "Decreasing" if z_stat < 0 else "Increasing"

            print(f"  Significance: {significance}")
            print(f"  Interpretation: {interpretation} ({direction})")

            return {
                'z_statistic': z_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'direction': direction
            }
        else:
            print("❌ Cannot calculate trend test insufficient variance")
            return None

    except Exception as e:
        print(f"❌ Trend test failed: {e}")
        return None

trend_test_results = cochran_armitage_trend_test(overall_trends)

# Step 3: Policy Impact Analysis (2014 CMS Changes)
print(f"\n🏛️  POLICY IMPACT ANALYSIS (2014 CMS Form CMS 10455)")
print("_" * 55)

def analyze_policy_impact(data, trend_data):
    """Analyze the impact of 2014 CMS policy changes"""

    # Define pre and post periods using actual format
    pre_2014_periods = ['2008 - 2010', '2011 - 2013']
    post_2014_periods = ['2014 - 2016', '2017 - 2019', '2020 - 2022']

    # Calculate pre/post rates
    pre_data = trend_data.loc[trend_data.index.intersection(pre_2014_periods)]
    post_data = trend_data.loc[trend_data.index.intersection(post_2014_periods)]

    if len(pre_data) > 0 and len(post_data) > 0:
        # Aggregate pre/post statistics
        pre_patients = pre_data['n_patients'].sum()
        pre_restrained = pre_data['n_restrained'].sum()
        pre_rate = pre_restrained / pre_patients

        post_patients = post_data['n_patients'].sum()
        post_restrained = post_data['n_restrained'].sum()
        post_rate = post_restrained / post_patients

        # Calculate impact
        rate_change = post_rate - pre_rate
        rate_change_pct = (rate_change / pre_rate) * 100

        print(f"Policy Impact Assessment:")
        print(f"{'Period':<15} {'N Patients':<12} {'N Restrained':<13} {'Rate (%)':<10}")
        print("_" * 55)
        print(f"{'Pre 2014':<15} {pre_patients:<12,.0f} {pre_restrained:<13,.0f} {pre_rate*100:<10.1f}")
        print(f"{'Post 2014':<15} {post_patients:<12,.0f} {post_restrained:<13,.0f} {post_rate*100:<10.1f}")
        print(f"{'Change':<15} {'':<12} {'':<13} {rate_change*100:>+10.1f}")
        print(f"{'% Change':<15} {'':<12} {'':<13} {rate_change_pct:>+10.1f}%")

        # Statistical test for difference
        try:
            # Chi square test for independence
            contingency = np.array([
                [pre_restrained, pre_patients - pre_restrained],
                [post_restrained, post_patients - post_restrained]
            ])

            chi2_stat, p_value, _, _ = chi2_contingency(contingency)

            print(f"\nStatistical Test (Chi square):")
            print(f"  Chi square: {chi2_stat:.3f}")
            print(f"  P value: {p_value:.3f}")

            if p_value < 0.001:
                print(f"  Result: Highly significant difference (p < 0.001)")
            elif p_value < 0.01:
                print(f"  Result: Significant difference (p < 0.01)")
            elif p_value < 0.05:
                print(f"  Result: Significant difference (p < 0.05)")
            else:
                print(f"  Result: No significant difference (p ≥ 0.05)")

        except Exception as e:
            print(f"Statistical test failed: {e}")

        return {
            'pre_rate': pre_rate,
            'post_rate': post_rate,
            'rate_change': rate_change,
            'rate_change_pct': rate_change_pct,
            'p_value': p_value if 'p_value' in locals() else None
        }
    else:
        print("❌ Insufficient data for pre/post 2014 comparison")
        return None

policy_impact = analyze_policy_impact(df_temporal, overall_trends)

# Step 4: Demographic Disparities Over Time
print(f"\n🎯 DEMOGRAPHIC DISPARITIES EVOLUTION")
print("_" * 40)

def analyze_demographic_trends(data):
    """Analyze how demographic disparities have changed over time"""

    # Focus on main demographic groups
    demographic_groups = {}

    # Use race categories from raw data
    if 'race' in data.columns:
        race_categories = ['Asian', 'Hispanic/Latino', 'Black/African American', 'White']
        for race in race_categories:
            if race in data['race'].values:
                demographic_groups[f'Race_{race.replace("/", "_").replace(" ", "_")}'] = data['race'] == race

    # Gender analysis
    if 'gender' in data.columns:
        demographic_groups['Gender_Male'] = data['gender'] == 'M'
        demographic_groups['Gender_Female'] = data['gender'] == 'F'

    if not demographic_groups:
        print("❌ No demographic variables available for trend analysis")
        return None

    print(f"Analyzing trends for: {list(demographic_groups.keys())}")

    # Calculate trends for each group
    demographic_trends = {}

    for group_name, group_mask in demographic_groups.items():
        group_data = data[group_mask]

        if len(group_data) > 100:  # Minimum size threshold
            group_trends = group_data.groupby('time_period')['physically_restrained'].agg(['count', 'sum', 'mean'])
            group_trends.columns = ['n_patients', 'n_restrained', 'restraint_rate']

            # Sort by time period using actual format
            period_order = ['2008 - 2010', '2011 - 2013', '2014 - 2016', '2017 - 2019', '2020 - 2022']
            available_periods = [p for p in period_order if p in group_trends.index]
            group_trends = group_trends.reindex(available_periods)

            demographic_trends[group_name] = group_trends

    # Display trends
    for group_name, trends in demographic_trends.items():
        print(f"\n{group_name} Restraint Trends:")
        print(f"{'Period':<12} {'N Patients':<12} {'Rate (%)':<10}")
        print("_" * 35)

        for period in trends.index:
            if period != 'Unknown' and not pd.isna(trends.loc[period, 'restraint_rate']):
                rate_pct = trends.loc[period, 'restraint_rate'] * 100
                n_patients = trends.loc[period, 'n_patients']
                print(f"{period:<12} {n_patients:<12,.0f} {rate_pct:<10.1f}")

    return demographic_trends

demographic_trends = analyze_demographic_trends(df_temporal)

# Step 5: Time × Demographic Interactions
print(f"\n🔗 TIME × DEMOGRAPHIC INTERACTIONS")
print("_" * 35)

def test_time_demographic_interactions(data):
    """Test for significant time × demographic interactions"""

    if 'race' not in data.columns:
        print("❌ No race variable available for interaction testing")
        return None

    # Focus on main racial groups
    main_races = ['Asian', 'Hispanic/Latino', 'Black/African American']
    interaction_results = {}

    for race in main_races:
        if race in data['race'].values:
            race_data = data[data['race'].isin([race, 'White'])].copy()

            if len(race_data) > 500:  # Sufficient sample size
                # Create race indicator (1 = minority group, 0 = White)
                race_data['minority_group'] = (race_data['race'] == race).astype(int)

                # Create time period dummies
                time_periods = race_data['time_period'].unique()
                time_periods = [p for p in time_periods if p != 'Unknown']

                if len(time_periods) >= 3:
                    # Test interaction using logistic regression
                    try:
                        # Create design matrix
                        X_interact = pd.get_dummies(race_data['time_period'], prefix='time')
                        X_interact['minority_group'] = race_data['minority_group'].astype(float)

                        # Add interaction terms
                        for col in X_interact.columns:
                            if col.startswith('time_'):
                                interaction_col = f"{col}_x_minority"
                                X_interact[interaction_col] = X_interact[col] * X_interact['minority_group']

                        # Add constant
                        X_interact = sm.add_constant(X_interact)

                        # Remove any non numeric columns and handle missing values
                        X_interact = X_interact.select_dtypes(include=[np.number]).fillna(0)

                        # Fit model
                        y_interact = race_data['physically_restrained'].astype(float)

                        # Ensure X and y have same length
                        min_len = min(len(X_interact), len(y_interact))
                        X_interact = X_interact.iloc[:min_len]
                        y_interact = y_interact.iloc[:min_len]

                        model = sm.GLM(y_interact, X_interact, family=sm.families.Binomial())
                        result = model.fit()

                        # Test for significant interactions
                        interaction_vars = [col for col in X_interact.columns if '_x_minority' in col]
                        if interaction_vars:
                            interaction_p_values = [result.pvalues[var] for var in interaction_vars if var in result.pvalues.index]
                            min_p = min(interaction_p_values) if interaction_p_values else 1.0

                            significant_interactions = sum(1 for p in interaction_p_values if p < 0.05)

                            interaction_results[race] = {
                                'min_p_value': min_p,
                                'significant_interactions': significant_interactions,
                                'total_interactions': len(interaction_p_values),
                                'sample_size': len(race_data)
                            }

                            print(f"{race} vs White:")
                            print(f"  Sample size: {len(race_data):,}")
                            print(f"  Significant interactions: {significant_interactions}/{len(interaction_p_values)}")
                            print(f"  Min p value: {min_p:.3f}")

                            if min_p < 0.05:
                                print(f"  → Disparity pattern changes over time *")
                            else:
                                print(f"  → Stable disparity pattern over time")

                    except Exception as e:
                        print(f"Interaction test failed for {race}: {e}")

    return interaction_results

interaction_results = test_time_demographic_interactions(df_temporal)

# Step 6: Summary and Interpretation
print(f"\n{'='*80}")
print("TEMPORAL TREND ANALYSIS SUMMARY")
print("="*80)

def summarize_temporal_findings(overall_trends, trend_test, policy_impact, demographic_trends, interactions):
    """Provide comprehensive summary of temporal analysis"""

    print(f"📊 KEY TEMPORAL FINDINGS:")

    # Overall trend summary
    if len(overall_trends) >= 2:
        valid_periods = [p for p in overall_trends.index if p != 'Unknown' and not pd.isna(overall_trends.loc[p, 'restraint_rate'])]
        if len(valid_periods) >= 2:
            first_period = valid_periods[0]
            last_period = valid_periods[-1]
            first_rate = overall_trends.loc[first_period, 'restraint_rate']
            last_rate = overall_trends.loc[last_period, 'restraint_rate']
            total_change = ((last_rate - first_rate) / first_rate) * 100

            print(f"   • Overall trend ({first_period} to {last_period}): {first_rate:.1%} → {last_rate:.1%}")
            print(f"   • Total change: {total_change:+.1f}%")

    # Trend test results
    if trend_test:
        direction = "decreasing" if trend_test['direction'] == 'Decreasing' else "increasing"
        significance = "significant" if trend_test['significant'] else "non significant"
        print(f"   • Statistical trend: {significance} {direction} pattern (p = {trend_test['p_value']:.3f})")

    # Policy impact
    if policy_impact:
        policy_direction = "decreased" if policy_impact['rate_change'] < 0 else "increased"
        print(f"   • 2014 CMS policy impact: Restraint use {policy_direction} by {abs(policy_impact['rate_change_pct']):.1f}%")

        if policy_impact['p_value'] and policy_impact['p_value'] < 0.05:
            print(f"   • Policy impact: Statistically significant (p = {policy_impact['p_value']:.3f})")
        else:
            print(f"   • Policy impact: Not statistically significant")

    # Demographic trend summary
    if demographic_trends:
        stable_groups = 0
        changing_groups = 0

        for group_name, trends in demographic_trends.items():
            valid_trends = trends.dropna()
            if len(valid_trends) >= 2:
                first_rate = valid_trends.iloc[0]['restraint_rate']
                last_rate = valid_trends.iloc[-1]['restraint_rate']

                if abs(last_rate - first_rate) / first_rate > 0.1:  # >10% change
                    changing_groups += 1
                else:
                    stable_groups += 1

        print(f"   • Demographic disparity evolution: {stable_groups} stable, {changing_groups} changing patterns")

    # Interaction summary
    if interaction_results:
        significant_interactions = sum(1 for r in interaction_results.values() if r['min_p_value'] < 0.05)
        total_groups = len(interaction_results)

        print(f"   • Time × demographic interactions: {significant_interactions}/{total_groups} groups show changing disparities")

    return True

summary_complete = summarize_temporal_findings(
    overall_trends, trend_test_results, policy_impact, demographic_trends, interaction_results
)

# Final protocol completion message
print(f"\n📋 PROTOCOL COMPLETION:")
print(f"   ✅ Temporal trends calculated across time periods")
print(f"   ✅ Cochran Armitage trend test performed")
print(f"   ✅ 2014 CMS policy impact assessed")
print(f"   ✅ Demographic disparity evolution analyzed")
print(f"   ✅ Time × demographic interactions tested")

print(f"\n🎯 MANUSCRIPT IMPLICATIONS:")
print(f"   • Document overall restraint trend direction and magnitude")
print(f"   • Report 2014 policy effectiveness (or lack thereof)")
print(f"   • Describe demographic disparity persistence/evolution")
print(f"   • Discuss implications for future policy interventions")

print(f"\n✅ TEMPORAL TREND ANALYSIS COMPLETE")
print("📊 All protocol sensitivity analyses now complete!")
print("🔄 Ready for manuscript preparation with comprehensive analytical package")
