import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Chart style setup
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load data
print("Loading data for visualization...")
df = pd.read_csv('simple_interview_events.csv')

# Funnel analysis
onboarding_events = ['onboarding_start', 'profile_start', 'email_submit', 'paywall_show', 'payment_done']

def analyze_funnel(df):
    user_events = df.groupby('user_id').apply(
        lambda x: x.sort_values('event_time')[['event_type', 'event_time']].to_dict('records')
    ).reset_index()
    
    funnel_stages = {
        'onboarding_start': 1,
        'profile_start': 2, 
        'email_submit': 3,
        'paywall_show': 4,
        'payment_done': 5
    }
    
    user_max_stages = []
    for _, row in user_events.iterrows():
        events = row[0]
        max_stage = 0
        for event in events:
            if event['event_type'] in funnel_stages:
                max_stage = max(max_stage, funnel_stages[event['event_type']])
        user_max_stages.append(max_stage)
    
    user_events['max_stage'] = user_max_stages
    
    total_users = len(user_events)
    funnel_conversion = {}
    
    for stage_num, stage_name in enumerate(onboarding_events, 1):
        users_reached = len(user_events[user_events['max_stage'] >= stage_num])
        conversion_rate = users_reached / total_users * 100
        funnel_conversion[stage_name] = {
            'users_reached': users_reached,
            'conversion_rate': conversion_rate
        }
    
    return funnel_conversion, user_events

funnel_results, user_events = analyze_funnel(df)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Simple App Onboarding Funnel Analysis', fontsize=16, fontweight='bold')

# 1. Conversion funnel
stages = list(funnel_results.keys())
conversion_rates = [funnel_results[stage]['conversion_rate'] for stage in stages]
users_reached = [funnel_results[stage]['users_reached'] for stage in stages]

# Create funnel
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B2635']
bars = axes[0, 0].bar(stages, conversion_rates, color=colors, alpha=0.8)
axes[0, 0].set_title('Conversion by Funnel Stages', fontweight='bold')
axes[0, 0].set_ylabel('Conversion (%)')
axes[0, 0].tick_params(axis='x', rotation=45)

# Add values on bars
for bar, rate, users in zip(bars, conversion_rates, users_reached):
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%\n({users:,})', ha='center', va='bottom', fontweight='bold')

# 2. Losses between stages
losses = []
loss_labels = []
for i in range(len(stages) - 1):
    current_stage = stages[i]
    next_stage = stages[i + 1]
    current_users = funnel_results[current_stage]['users_reached']
    next_users = funnel_results[next_stage]['users_reached']
    lost_users = current_users - next_users
    loss_rate = lost_users / current_users * 100
    losses.append(loss_rate)
    loss_labels.append(f'{current_stage}\n→ {next_stage}')

colors_loss = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars_loss = axes[0, 1].bar(loss_labels, losses, color=colors_loss, alpha=0.8)
axes[0, 1].set_title('Losses Between Stages', fontweight='bold')
axes[0, 1].set_ylabel('Loss (%)')
axes[0, 1].tick_params(axis='x', rotation=45)

for bar, loss in zip(bars_loss, losses):
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{loss:.1f}%', ha='center', va='bottom', fontweight='bold')

# 3. Analysis by funnel types
funnel_type_events = df[df['event_type'] == 'onboarding_start'].copy()
funnel_type_events['funnel_params'] = funnel_type_events['event_params'].apply(
    lambda x: json.loads(x) if pd.notna(x) and x != '{}' else {}
)
funnel_type_events['funnel_type'] = funnel_type_events['funnel_params'].apply(
    lambda x: x.get('funnel_type', 'unknown') if isinstance(x, dict) else 'unknown'
)

funnel_type_counts = funnel_type_events['funnel_type'].value_counts()
colors_funnel = ['#FF9999', '#66B2FF', '#99FF99']
wedges, texts, autotexts = axes[1, 0].pie(funnel_type_counts.values, labels=funnel_type_counts.index, 
                                         autopct='%1.1f%%', colors=colors_funnel, startangle=90)
axes[1, 0].set_title('Distribution by Funnel Types', fontweight='bold')

# --- EXPERIMENT ANALYSIS (ENGLISH, BAR CHARTS, 4 SUBPLOTS, TRANSLATED LABELS) ---
print("\nCreating experiment analysis chart (EN, bar charts, identical to original)...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Simple App Experiments Analysis', fontsize=16, fontweight='bold')

# 1. Overall conversion by experiment
exp_names = [exp for exp, _ in sorted_experiments[:8]]
conversions = [data['overall_conversion'] for _, data in sorted_experiments[:8]]

bars = axes[0, 0].bar(exp_names, conversions, color='skyblue', alpha=0.8)
axes[0, 0].set_title('Overall Conversion by Experiment', fontweight='bold')
axes[0, 0].set_ylabel('Conversion (%)')
axes[0, 0].tick_params(axis='x', rotation=45)

for bar, conv in zip(bars, conversions):
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{conv:.1f}%', ha='center', va='bottom', fontweight='bold')

# 2. Lift compared to control
lifts = [data['max_lift'] for _, data in sorted_experiments[:8]]
colors = ['green' if lift > 0 else 'red' for lift in lifts]

bars_lift = axes[0, 1].bar(exp_names, lifts, color=colors, alpha=0.8)
axes[0, 1].set_title('Lift Compared to Control Group', fontweight='bold')
axes[0, 1].set_ylabel('Lift (%)')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)

for bar, lift in zip(bars_lift, lifts):
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + (0.5 if lift > 0 else -1),
                    f'{lift:+.1f}%', ha='center', va='bottom' if lift > 0 else 'top', 
                    fontweight='bold', color='green' if lift > 0 else 'red')

# 3. Sample size
users = [data['total_users'] for _, data in sorted_experiments[:8]]

bars_users = axes[1, 0].bar(exp_names, users, color='orange', alpha=0.8)
axes[1, 0].set_title('Sample Size by Experiment', fontweight='bold')
axes[1, 0].set_ylabel('Number of Users')
axes[1, 0].tick_params(axis='x', rotation=45)

for bar, user_count in zip(bars_users, users):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{user_count:,}', ha='center', va='bottom', fontweight='bold')

# 4. Detailed comparison for top-3 experiments
top_3_exp = sorted_experiments[:3]

if top_3_exp:
    exp_names_detailed = []
    control_rates = []
    treatment_rates = []
    
    for exp_name, _ in top_3_exp:
        if exp_name in experiment_results and 'control' in experiment_results[exp_name]:
            control_rate = experiment_results[exp_name]['control']['conversion_rate']
            
            # Find the best treatment group
            best_treatment_rate = 0
            for group, data in experiment_results[exp_name].items():
                if group != 'control':
                    best_treatment_rate = max(best_treatment_rate, data['conversion_rate'])
            
            if best_treatment_rate > 0:
                exp_names_detailed.append(exp_name)
                control_rates.append(control_rate)
                treatment_rates.append(best_treatment_rate)
    
    if exp_names_detailed:
        x = np.arange(len(exp_names_detailed))
        width = 0.35
        
        bars_control = axes[1, 1].bar(x - width/2, control_rates, width, label='Control', color='lightcoral', alpha=0.8)
        bars_treatment = axes[1, 1].bar(x + width/2, treatment_rates, width, label='Test', color='lightgreen', alpha=0.8)
        
        axes[1, 1].set_title('Control vs Test Comparison (Top-3)', fontweight='bold')
        axes[1, 1].set_ylabel('Conversion (%)')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(exp_names_detailed)
        axes[1, 1].legend()
        
        for bar, rate in zip(bars_control, control_rates):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
        
        for bar, rate in zip(bars_treatment, treatment_rates):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('experiment_analysis_EN.png', dpi=300, bbox_inches='tight')
print("\nExperiment analysis chart saved as 'experiment_analysis_EN.png'")

# --- EXPERIMENT LIFT ANALYSIS (ENGLISH, MATCHES ORIGINAL) ---
print("\nCreating experiment lift chart (EN)...")
experiment_events = df[df['event_type'] == 'experiment_exposure'].copy()

if len(experiment_events) > 0:
    experiment_events['experiment_params'] = experiment_events['event_params'].apply(
        lambda x: json.loads(x) if pd.notna(x) and x != '{}' else {}
    )
    experiment_events['experiment_name'] = experiment_events['experiment_params'].apply(
        lambda x: x.get('experiment_name', 'unknown') if isinstance(x, dict) else 'unknown'
    )
    experiment_events['experiment_group'] = experiment_events['experiment_params'].apply(
        lambda x: x.get('experiment_group', 'unknown') if isinstance(x, dict) else 'unknown'
    )
    
    experiments = experiment_events['experiment_name'].unique()
    plot_data = []
    for exp_name in experiments:
        if exp_name == 'unknown':
            continue
        exp_data = experiment_events[experiment_events['experiment_name'] == exp_name]
        groups = exp_data['experiment_group'].unique()
        group_conv = {}
        group_n = {}
        group_payments = {}
        for group in groups:
            if group == 'unknown':
                continue
            group_users = set(exp_data[exp_data['experiment_group'] == group]['user_id'])
            group_payments[group] = df[(df['user_id'].isin(group_users)) & (df['event_type'] == 'payment_done')]['user_id'].nunique()
            group_n[group] = len(group_users)
            group_conv[group] = group_payments[group] / group_n[group] * 100 if group_n[group] > 0 else 0
        # If both control and test exist
        if 'control' in group_conv and len(group_conv) > 1:
            for group in groups:
                if group != 'control' and group != 'unknown':
                    lift = group_conv[group] - group_conv['control']
                    lift_pct = (lift / group_conv['control'] * 100) if group_conv['control'] > 0 else 0
                    # t-test
                    control_data = [1]*group_payments['control'] + [0]*(group_n['control']-group_payments['control'])
                    treat_data = [1]*group_payments[group] + [0]*(group_n[group]-group_payments[group])
                    t_stat, p_value = stats.ttest_ind(control_data, treat_data)
                    plot_data.append({
                        'experiment': exp_name,
                        'group': group,
                        'control_rate': group_conv['control'],
                        'test_rate': group_conv[group],
                        'lift': lift_pct,
                        'p_value': p_value
                    })
    # To DataFrame
    plot_df = pd.DataFrame(plot_data)
    plot_df = plot_df.sort_values('lift', ascending=False)
    # Visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(plot_df['experiment'], plot_df['lift'], color=plot_df['lift'].apply(lambda x: 'green' if x > 0 else 'red'), alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel('Lift test vs control (%)')
    ax.set_title('Experiment Lift Analysis')
    for bar, pval, lift, test, control in zip(bars, plot_df['p_value'], plot_df['lift'], plot_df['test_rate'], plot_df['control_rate']):
        height = bar.get_height()
        signif = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        if pval < 0.001:
            p_str = 'p < 0.001'
        else:
            p_str = f'p={pval:.3f}'
        ax.text(bar.get_x() + bar.get_width()/2., height + (2 if lift > 0 else -4),
                f"{lift:+.1f}%\n{p_str} {signif}\nT: {test:.1f}%\nC: {control:.1f}%",
                ha='center', va='bottom' if lift > 0 else 'top', fontsize=10, fontweight='bold', color='black')
    plt.tight_layout()
    plt.savefig('experiment_lift_EN.png', dpi=300, bbox_inches='tight')
    print("Experiment lift chart saved as 'experiment_lift_EN.png'")
else:
    print('No experiment data for lift visualization.')

# --- EXPERIMENT PERFORMANCE COMPARISON (ENGLISH, MATCHES ORIGINAL) ---
print("\nCreating experiment performance comparison chart (EN)...")
if len(experiment_events) > 0:
    experiments = experiment_events['experiment_name'].unique()
    group_colors = sns.color_palette('Set2', len(experiments))
    plot_data = {}
    for exp_name, color in zip(experiments, group_colors):
        if exp_name == 'unknown':
            continue
        exp_data = experiment_events[experiment_events['experiment_name'] == exp_name]
        test_users = set(exp_data[exp_data['experiment_group'] == 'test']['user_id'])
        control_users = set(exp_data[exp_data['experiment_group'] == 'control']['user_id'])
        for group, users, style in [('test', test_users, '-'), ('control', control_users, '--')]:
            if not users:
                continue
            group_conv = []
            for stage in onboarding_events:
                stage_num = onboarding_events.index(stage) + 1
                users_reached = user_events[(user_events['user_id'].isin(users)) & (user_events['max_stage'] >= stage_num)]['user_id'].nunique()
                group_conv.append(users_reached / len(users) * 100 if len(users) > 0 else 0)
            plot_data[(exp_name, group)] = group_conv
    # Plot
    fig, ax = plt.subplots(figsize=(16, 8))
    for (exp_name, group), conv in plot_data.items():
        label = f'{exp_name} ({group})'
        ax.plot(onboarding_events, conv, label=label, linewidth=2 if group=='test' else 1, linestyle='-' if group=='test' else '--')
    ax.set_xlabel('Funnel Stages')
    ax.set_ylabel('Conversion (%)')
    ax.set_title('Experiment Performance Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiment_analysis_EN.png', dpi=300, bbox_inches='tight')
    print("Experiment performance chart saved as 'experiment_analysis_EN.png'")
else:
    print('No experiment data for performance comparison.')

print("All English charts have been created successfully!") 