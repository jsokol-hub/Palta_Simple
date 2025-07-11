import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Загружаем данные
print("Загружаем данные для анализа экспериментов...")
df = pd.read_csv('data/simple_interview_events.csv')

# Анализ экспериментов с контрольными группами
print("\n=== ДЕТАЛЬНЫЙ АНАЛИЗ ЭКСПЕРИМЕНТОВ ===")
experiment_events = df[df['event_type'] == 'experiment_exposure'].copy()

if len(experiment_events) > 0:
    # Парсим параметры экспериментов
    experiment_events['experiment_params'] = experiment_events['event_params'].apply(
        lambda x: json.loads(x) if pd.notna(x) and x != '{}' else {}
    )
    
    # Извлекаем название эксперимента и группу
    experiment_events['experiment_name'] = experiment_events['experiment_params'].apply(
        lambda x: x.get('experiment_name', 'unknown') if isinstance(x, dict) else 'unknown'
    )
    experiment_events['experiment_group'] = experiment_events['experiment_params'].apply(
        lambda x: x.get('experiment_group', 'unknown') if isinstance(x, dict) else 'unknown'
    )
    
    print("Структура экспериментов:")
    print(experiment_events[['experiment_name', 'experiment_group']].value_counts())
    
    # Анализируем каждый эксперимент отдельно
    experiments = experiment_events['experiment_name'].unique()
    
    experiment_results = {}
    
    for exp_name in experiments:
        if exp_name == 'unknown':
            continue
            
        print(f"\n--- АНАЛИЗ ЭКСПЕРИМЕНТА {exp_name} ---")
        
        # Получаем данные для этого эксперимента
        exp_data = experiment_events[experiment_events['experiment_name'] == exp_name]
        
        # Анализируем по группам
        groups = exp_data['experiment_group'].unique()
        
        group_results = {}
        
        for group in groups:
            if group == 'unknown':
                continue
                
            # Пользователи в этой группе
            group_users = set(exp_data[exp_data['experiment_group'] == group]['user_id'])
            
            # Конверсия в покупку для этой группы
            group_payments = df[
                (df['user_id'].isin(group_users)) & 
                (df['event_type'] == 'payment_done')
            ]['user_id'].nunique()
            
            # Конверсия по этапам воронки
            group_onboarding = df[
                (df['user_id'].isin(group_users)) & 
                (df['event_type'] == 'onboarding_start')
            ]['user_id'].nunique()
            
            group_profile = df[
                (df['user_id'].isin(group_users)) & 
                (df['event_type'] == 'profile_start')
            ]['user_id'].nunique()
            
            group_email = df[
                (df['user_id'].isin(group_users)) & 
                (df['event_type'] == 'email_submit')
            ]['user_id'].nunique()
            
            group_paywall = df[
                (df['user_id'].isin(group_users)) & 
                (df['event_type'] == 'paywall_show')
            ]['user_id'].nunique()
            
            conversion_rate = group_payments / len(group_users) * 100 if len(group_users) > 0 else 0
            
            group_results[group] = {
                'users': len(group_users),
                'payments': group_payments,
                'conversion_rate': conversion_rate,
                'funnel': {
                    'onboarding_start': group_onboarding,
                    'profile_start': group_profile,
                    'email_submit': group_email,
                    'paywall_show': group_paywall,
                    'payment_done': group_payments
                }
            }
            
            print(f"Группа {group}: {group_payments}/{len(group_users)} покупок ({conversion_rate:.1f}%)")
        
        # Сравниваем группы (если есть control и treatment)
        if 'control' in group_results and len(group_results) > 1:
            control_rate = group_results['control']['conversion_rate']
            
            print(f"\nСравнение с контрольной группой ({control_rate:.1f}%):")
            
            for group, data in group_results.items():
                if group != 'control':
                    treatment_rate = data['conversion_rate']
                    lift = ((treatment_rate - control_rate) / control_rate * 100) if control_rate > 0 else 0
                    
                    print(f"  {group}: {treatment_rate:.1f}% (lift: {lift:+.1f}%)")
                    
                    # Статистическая значимость (простой t-test)
                    control_users = group_results['control']['users']
                    treatment_users = data['users']
                    
                    if control_users > 0 and treatment_users > 0:
                        # Создаем массивы для t-test (1 = покупка, 0 = нет покупки)
                        control_data = [1] * group_results['control']['payments'] + [0] * (control_users - group_results['control']['payments'])
                        treatment_data = [1] * data['payments'] + [0] * (treatment_users - data['payments'])
                        
                        if len(control_data) > 0 and len(treatment_data) > 0:
                            t_stat, p_value = stats.ttest_ind(control_data, treatment_data)
                            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                            print(f"    Статистическая значимость: p={p_value:.4f} {significance}")
        
        experiment_results[exp_name] = group_results
    
    # Выбираем топ-3 эксперимента
    print("\n=== ТОП-3 ЭКСПЕРИМЕНТА ===")
    
    # Считаем общую конверсию для каждого эксперимента
    experiment_summary = {}
    
    for exp_name, groups in experiment_results.items():
        total_users = sum(data['users'] for data in groups.values())
        total_payments = sum(data['payments'] for data in groups.values())
        overall_conversion = total_payments / total_users * 100 if total_users > 0 else 0
        
        # Находим максимальный lift по сравнению с control
        max_lift = 0
        if 'control' in groups:
            control_rate = groups['control']['conversion_rate']
            for group, data in groups.items():
                if group != 'control':
                    lift = ((data['conversion_rate'] - control_rate) / control_rate * 100) if control_rate > 0 else 0
                    max_lift = max(max_lift, lift)
        
        experiment_summary[exp_name] = {
            'total_users': total_users,
            'total_payments': total_payments,
            'overall_conversion': overall_conversion,
            'max_lift': max_lift
        }
    
    # Сортируем по lift (если есть) или по общей конверсии
    sorted_experiments = sorted(experiment_summary.items(), 
                               key=lambda x: (x[1]['max_lift'], x[1]['overall_conversion']), 
                               reverse=True)
    
    print("Рейтинг экспериментов:")
    for i, (exp_name, data) in enumerate(sorted_experiments[:3], 1):
        print(f"{i}. {exp_name}: {data['overall_conversion']:.1f}% конверсия, "
              f"lift: {data['max_lift']:+.1f}%, {data['total_users']:,} пользователей")
    
    # Создаем визуализацию
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Анализ экспериментов Simple App', fontsize=16, fontweight='bold')
    
    # 1. Общая конверсия по экспериментам
    exp_names = [exp for exp, _ in sorted_experiments[:8]]
    conversions = [data['overall_conversion'] for _, data in sorted_experiments[:8]]
    
    bars = axes[0, 0].bar(exp_names, conversions, color='skyblue', alpha=0.8)
    axes[0, 0].set_title('Общая конверсия по экспериментам', fontweight='bold')
    axes[0, 0].set_ylabel('Конверсия (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    for bar, conv in zip(bars, conversions):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{conv:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Lift по сравнению с control
    lifts = [data['max_lift'] for _, data in sorted_experiments[:8]]
    colors = ['green' if lift > 0 else 'red' for lift in lifts]
    
    bars_lift = axes[0, 1].bar(exp_names, lifts, color=colors, alpha=0.8)
    axes[0, 1].set_title('Lift по сравнению с контрольной группой', fontweight='bold')
    axes[0, 1].set_ylabel('Lift (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    for bar, lift in zip(bars_lift, lifts):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + (0.5 if lift > 0 else -1),
                        f'{lift:+.1f}%', ha='center', va='bottom' if lift > 0 else 'top', 
                        fontweight='bold', color='green' if lift > 0 else 'red')
    
    # 3. Размер выборки
    users = [data['total_users'] for _, data in sorted_experiments[:8]]
    
    bars_users = axes[1, 0].bar(exp_names, users, color='orange', alpha=0.8)
    axes[1, 0].set_title('Размер выборки по экспериментам', fontweight='bold')
    axes[1, 0].set_ylabel('Количество пользователей')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    for bar, user_count in zip(bars_users, users):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 50,
                        f'{user_count:,}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Детальное сравнение для топ-3 экспериментов
    top_3_exp = sorted_experiments[:3]
    
    if top_3_exp:
        exp_names_detailed = []
        control_rates = []
        treatment_rates = []
        
        for exp_name, _ in top_3_exp:
            if exp_name in experiment_results and 'control' in experiment_results[exp_name]:
                control_rate = experiment_results[exp_name]['control']['conversion_rate']
                
                # Находим лучшую treatment группу
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
            
            axes[1, 1].set_title('Сравнение Control vs Test (топ-3)', fontweight='bold')
            axes[1, 1].set_ylabel('Конверсия (%)')
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
    plt.savefig('experiment_analysis.png', dpi=300, bbox_inches='tight')
    print("\nГрафик анализа экспериментов сохранен как 'experiment_analysis.png'")

else:
    print("События экспериментов не найдены в данных")

print("\n=== АНАЛИЗ ЭКСПЕРИМЕНТОВ ЗАВЕРШЕН ===") 