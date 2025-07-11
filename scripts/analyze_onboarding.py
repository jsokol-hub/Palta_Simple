import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Загружаем данные
print("Загружаем данные...")
df = pd.read_csv('data/simple_interview_events.csv')

print(f"Общее количество записей: {len(df):,}")
print(f"Уникальных пользователей: {df['user_id'].nunique():,}")
print(f"Период данных: {df['event_time'].min()} - {df['event_time'].max()}")

# Анализ типов событий
print("\n=== АНАЛИЗ ТИПОВ СОБЫТИЙ ===")
event_counts = df['event_type'].value_counts()
print(event_counts)

# Основные события онбординга
onboarding_events = ['onboarding_start', 'profile_start', 'email_submit', 'paywall_show', 'payment_done']

print(f"\n=== СОБЫТИЯ ОНБОРДИНГА ===")
for event in onboarding_events:
    if event in event_counts.index:
        print(f"{event}: {event_counts[event]:,}")

# Анализ воронки онбординга
print("\n=== АНАЛИЗ ВОРОНКИ ОНБОРДИНГА ===")

# Создаем временную последовательность событий для каждого пользователя
def analyze_funnel(df):
    # Группируем по пользователю и сортируем по времени
    user_events = df.groupby('user_id').apply(
        lambda x: x.sort_values('event_time')[['event_type', 'event_time']].to_dict('records')
    ).reset_index()
    
    # Определяем максимальный этап для каждого пользователя
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
    
    # Считаем конверсию по этапам
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

print("Конверсия по этапам воронки:")
for stage, data in funnel_results.items():
    print(f"{stage}: {data['users_reached']:,} пользователей ({data['conversion_rate']:.1f}%)")

# Анализ потерь между этапами
print("\n=== ПОТЕРИ МЕЖДУ ЭТАПАМИ ===")
stages = list(funnel_results.keys())
for i in range(len(stages) - 1):
    current_stage = stages[i]
    next_stage = stages[i + 1]
    current_users = funnel_results[current_stage]['users_reached']
    next_users = funnel_results[next_stage]['users_reached']
    lost_users = current_users - next_users
    loss_rate = lost_users / current_users * 100
    
    print(f"{current_stage} → {next_stage}: потеряно {lost_users:,} пользователей ({loss_rate:.1f}%)")

# Анализ экспериментов
print("\n=== АНАЛИЗ ЭКСПЕРИМЕНТОВ ===")
experiment_events = df[df['event_type'] == 'experiment_exposure'].copy()

if len(experiment_events) > 0:
    # Парсим параметры экспериментов
    experiment_events['experiment_params'] = experiment_events['event_params'].apply(
        lambda x: json.loads(x) if pd.notna(x) and x != '{}' else {}
    )
    
    # Извлекаем названия экспериментов
    experiment_events['experiment_name'] = experiment_events['experiment_params'].apply(
        lambda x: x.get('experiment_name', 'unknown') if isinstance(x, dict) else 'unknown'
    )
    
    # Анализируем эксперименты
    experiment_stats = experiment_events.groupby('experiment_name').agg({
        'user_id': 'nunique',
        'event_time': 'count'
    }).rename(columns={'user_id': 'unique_users', 'event_time': 'total_exposures'})
    
    experiment_stats = experiment_stats.sort_values('unique_users', ascending=False)
    
    print("Топ экспериментов по количеству уникальных пользователей:")
    print(experiment_stats.head(10))
    
    # Анализируем конверсию в экспериментах
    print("\nАнализ конверсии в экспериментах:")
    for exp_name in experiment_stats.head(3).index:
        exp_users = set(experiment_events[experiment_events['experiment_name'] == exp_name]['user_id'])
        
        # Находим пользователей из эксперимента, которые дошли до payment_done
        exp_payments = df[
            (df['user_id'].isin(exp_users)) & 
            (df['event_type'] == 'payment_done')
        ]['user_id'].nunique()
        
        conversion_rate = exp_payments / len(exp_users) * 100
        print(f"{exp_name}: {exp_payments}/{len(exp_users)} пользователей совершили покупку ({conversion_rate:.1f}%)")

else:
    print("События экспериментов не найдены в данных")

# Дополнительный анализ по типам воронок
print("\n=== АНАЛИЗ ПО ТИПАМ ВОРОНОК ===")
funnel_type_events = df[df['event_type'] == 'onboarding_start'].copy()
funnel_type_events['funnel_params'] = funnel_type_events['event_params'].apply(
    lambda x: json.loads(x) if pd.notna(x) and x != '{}' else {}
)
funnel_type_events['funnel_type'] = funnel_type_events['funnel_params'].apply(
    lambda x: x.get('funnel_type', 'unknown') if isinstance(x, dict) else 'unknown'
)

funnel_type_counts = funnel_type_events['funnel_type'].value_counts()
print("Распределение по типам воронок:")
print(funnel_type_counts)

# Анализ конверсии по типам воронок
for funnel_type in funnel_type_counts.index:
    if funnel_type != 'unknown':
        funnel_users = set(funnel_type_events[funnel_type_events['funnel_type'] == funnel_type]['user_id'])
        
        # Считаем конверсию для этого типа воронки
        funnel_user_events = user_events[user_events['user_id'].isin(funnel_users)]
        total_funnel_users = len(funnel_user_events)
        
        print(f"\nВоронка '{funnel_type}' ({total_funnel_users:,} пользователей):")
        for stage in onboarding_events:
            stage_num = onboarding_events.index(stage) + 1
            users_reached = len(funnel_user_events[funnel_user_events['max_stage'] >= stage_num])
            conversion_rate = users_reached / total_funnel_users * 100
            print(f"  {stage}: {users_reached:,} ({conversion_rate:.1f}%)")

print("\n=== АНАЛИЗ ЗАВЕРШЕН ===") 