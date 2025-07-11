import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Загружаем данные
print("Загружаем данные для визуализации...")
df = pd.read_csv('data/simple_interview_events.csv')

# Анализ воронки
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

# Создаем визуализации
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Анализ воронки онбординга Simple App', fontsize=16, fontweight='bold')

# 1. Воронка конверсии
stages = list(funnel_results.keys())
conversion_rates = [funnel_results[stage]['conversion_rate'] for stage in stages]
users_reached = [funnel_results[stage]['users_reached'] for stage in stages]

# Создаем воронку
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B2635']
bars = axes[0, 0].bar(stages, conversion_rates, color=colors, alpha=0.8)
axes[0, 0].set_title('Конверсия по этапам воронки', fontweight='bold')
axes[0, 0].set_ylabel('Конверсия (%)')
axes[0, 0].tick_params(axis='x', rotation=45)

# Добавляем значения на столбцы
for bar, rate, users in zip(bars, conversion_rates, users_reached):
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%\n({users:,})', ha='center', va='bottom', fontweight='bold')

# 2. Потери между этапами
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
axes[0, 1].set_title('Потери между этапами', fontweight='bold')
axes[0, 1].set_ylabel('Потери (%)')
axes[0, 1].tick_params(axis='x', rotation=45)

for bar, loss in zip(bars_loss, losses):
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{loss:.1f}%', ha='center', va='bottom', fontweight='bold')

# 3. Анализ по типам воронок
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
axes[1, 0].set_title('Распределение по типам воронок', fontweight='bold')

# 4. Анализ экспериментов
experiment_events = df[df['event_type'] == 'experiment_exposure'].copy()
if len(experiment_events) > 0:
    experiment_events['experiment_params'] = experiment_events['event_params'].apply(
        lambda x: json.loads(x) if pd.notna(x) and x != '{}' else {}
    )
    experiment_events['experiment_name'] = experiment_events['experiment_params'].apply(
        lambda x: x.get('experiment_name', 'unknown') if isinstance(x, dict) else 'unknown'
    )
    
    experiment_stats = experiment_events.groupby('experiment_name').agg({
        'user_id': 'nunique',
        'event_time': 'count'
    }).rename(columns={'user_id': 'unique_users', 'event_time': 'total_exposures'})
    
    experiment_stats = experiment_stats.sort_values('unique_users', ascending=False).head(8)
    
    # Анализируем конверсию в экспериментах
    conversion_rates_exp = []
    for exp_name in experiment_stats.index:
        exp_users = set(experiment_events[experiment_events['experiment_name'] == exp_name]['user_id'])
        exp_payments = df[
            (df['user_id'].isin(exp_users)) & 
            (df['event_type'] == 'payment_done')
        ]['user_id'].nunique()
        conversion_rate = exp_payments / len(exp_users) * 100
        conversion_rates_exp.append(conversion_rate)
    
    bars_exp = axes[1, 1].bar(experiment_stats.index, conversion_rates_exp, 
                             color='#FFD93D', alpha=0.8)
    axes[1, 1].set_title('Конверсия в покупку по экспериментам', fontweight='bold')
    axes[1, 1].set_ylabel('Конверсия (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Добавляем линию среднего значения
    avg_conversion = 5.4  # из общего анализа
    axes[1, 1].axhline(y=avg_conversion, color='red', linestyle='--', 
                       label=f'Среднее: {avg_conversion}%')
    axes[1, 1].legend()
    
    for bar, rate in zip(bars_exp, conversion_rates_exp):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('funnel_analysis.png', dpi=300, bbox_inches='tight')
print("График сохранен как 'funnel_analysis.png'")

# Создаем дополнительный график для детального анализа по типам воронок
fig2, ax = plt.subplots(figsize=(12, 8))

# Подготавливаем данные для сравнения воронок
funnel_types = ['female', 'male', 'main']
funnel_data = {}

for funnel_type in funnel_types:
    funnel_users = set(funnel_type_events[funnel_type_events['funnel_type'] == funnel_type]['user_id'])
    funnel_user_events = user_events[user_events['user_id'].isin(funnel_users)]
    total_funnel_users = len(funnel_user_events)
    
    funnel_conversion_rates = []
    for stage in onboarding_events:
        stage_num = onboarding_events.index(stage) + 1
        users_reached = len(funnel_user_events[funnel_user_events['max_stage'] >= stage_num])
        conversion_rate = users_reached / total_funnel_users * 100
        funnel_conversion_rates.append(conversion_rate)
    
    funnel_data[funnel_type] = funnel_conversion_rates

# Создаем график
x = np.arange(len(onboarding_events))
width = 0.25

colors_funnel_detailed = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for i, (funnel_type, rates) in enumerate(funnel_data.items()):
    ax.bar(x + i*width, rates, width, label=funnel_type.capitalize(), 
           color=colors_funnel_detailed[i], alpha=0.8)

ax.set_xlabel('Этапы воронки')
ax.set_ylabel('Конверсия (%)')
ax.set_title('Сравнение конверсии по типам воронок', fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(onboarding_events, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)

# Добавляем значения на столбцы
for i, (funnel_type, rates) in enumerate(funnel_data.items()):
    for j, rate in enumerate(rates):
        ax.text(j + i*width, rate + 1, f'{rate:.1f}%', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('funnel_comparison.png', dpi=300, bbox_inches='tight')
print("График сравнения воронок сохранен как 'funnel_comparison.png'")

print("\nВизуализация завершена!") 

# --- НОВАЯ ВИЗУАЛИЗАЦИЯ ПО ЭКСПЕРИМЕНТАМ ---
print("\nСоздаю график сравнения control/test по экспериментам...")
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
        # Если есть и control, и test
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
    # В датафрейм
    plot_df = pd.DataFrame(plot_data)
    # Сортируем по lift
    plot_df = plot_df.sort_values('lift', ascending=False)
    # Визуализация
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(plot_df['experiment'], plot_df['lift'], color=plot_df['lift'].apply(lambda x: 'green' if x > 0 else 'red'), alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel('Lift test vs control (%)')
    ax.set_title('Lift по конверсии в покупку (test vs control)\nи статистическая значимость (p-value)')
    for bar, pval, lift, test, control in zip(bars, plot_df['p_value'], plot_df['lift'], plot_df['test_rate'], plot_df['control_rate']):
        height = bar.get_height()
        signif = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        if pval < 0.001:
            p_str = 'p < 0.001'
        else:
            p_str = f'p={pval:.3f}'
        ax.text(bar.get_x() + bar.get_width()/2., height + (2 if lift > 0 else -4),
                f"{lift:+.1f}%\n{p_str} {signif}\nT: {test:.1f}%\nC: {control:.1f}%",
                ha='center', va='bottom' if lift > 0 else 'top', fontsize=9, fontweight='bold', color='black')
    plt.tight_layout()
    plt.savefig('experiment_lift.png', dpi=300, bbox_inches='tight')
    print("График lift по экспериментам сохранен как 'experiment_lift.png'")
else:
    print('Нет данных по экспериментам для визуализации lift.') 