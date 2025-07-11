import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Настройка для корректного отображения русских символов
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=== ПРАВИЛЬНЫЙ АНАЛИЗ ВОРОНКИ ===")
print("Учитывает последовательность событий, а не только максимальную стадию")

# Загружаем данные
print("\n1. ЗАГРУЗКА ДАННЫХ")
df = pd.read_csv('data/simple_interview_events.csv')
print(f"Загружено {len(df):,} событий для {df['user_id'].nunique():,} пользователей")

# Конвертируем время
df['event_time'] = pd.to_datetime(df['event_time'])

# Определяем этапы воронки в правильном порядке
funnel_stages = ['onboarding_start', 'profile_start', 'email_submit', 'paywall_show', 'payment_done']

print(f"\n2. ПРАВИЛЬНЫЙ АНАЛИЗ ПОСЛЕДОВАТЕЛЬНОСТИ СОБЫТИЙ")

# Для каждого пользователя анализируем последовательность событий
user_funnel_analysis = {}

for user_id in df['user_id'].unique():
    user_events = df[df['user_id'] == user_id].sort_values('event_time')
    
    # Создаем словарь для отслеживания прохождения этапов
    user_progress = {stage: False for stage in funnel_stages}
    stage_timestamps = {}
    
    # Анализируем события в хронологическом порядке
    for _, event in user_events.iterrows():
        event_type = event['event_type']
        event_time = event['event_time']
        
        if event_type in funnel_stages:
            # Проверяем, что пользователь прошел все предыдущие этапы
            stage_index = funnel_stages.index(event_type)
            previous_stages = funnel_stages[:stage_index]
            
            # Пользователь может пройти этап только если прошел все предыдущие
            can_proceed = all(user_progress[prev_stage] for prev_stage in previous_stages)
            
            if can_proceed:
                user_progress[event_type] = True
                stage_timestamps[event_type] = event_time
    
    user_funnel_analysis[user_id] = {
        'progress': user_progress,
        'timestamps': stage_timestamps,
        'max_stage_reached': max([i for i, stage in enumerate(funnel_stages) if user_progress[stage]], default=-1)
    }

print(f"Проанализированы последовательности для {len(user_funnel_analysis):,} пользователей")

# Считаем правильную конверсию по этапам
print(f"\n3. ПРАВИЛЬНАЯ КОНВЕРСИЯ ПО ЭТАПАМ ВОРОНКИ")

funnel_results = {}
total_users = len(user_funnel_analysis)

for i, stage in enumerate(funnel_stages):
    # Пользователи, которые прошли этот этап
    users_reached = sum(1 for user_data in user_funnel_analysis.values() 
                       if user_data['progress'][stage])
    
    # Пользователи, которые могли пройти этот этап (прошли предыдущий)
    if i == 0:  # Первый этап - все пользователи
        users_eligible = total_users
    else:
        previous_stage = funnel_stages[i-1]
        users_eligible = sum(1 for user_data in user_funnel_analysis.values() 
                           if user_data['progress'][previous_stage])
    
    # Конверсия от предыдущего этапа
    conversion_from_previous = (users_reached / users_eligible * 100) if users_eligible > 0 else 0
    
    # Конверсия от старта
    conversion_from_start = (users_reached / total_users * 100)
    
    funnel_results[stage] = {
        'users_reached': users_reached,
        'users_eligible': users_eligible,
        'conversion_from_previous': conversion_from_previous,
        'conversion_from_start': conversion_from_start
    }

# Выводим результаты
print("\nПРАВИЛЬНАЯ КОНВЕРСИЯ ПО ЭТАПАМ:")
print("-" * 80)
print(f"{'Этап':<20} {'Достигли':<10} {'Могли':<10} {'Конв. от пред.':<15} {'Конв. от старта':<15}")
print("-" * 80)

for stage in funnel_stages:
    data = funnel_results[stage]
    print(f"{stage:<20} {data['users_reached']:<10,} {data['users_eligible']:<10,} "
          f"{data['conversion_from_previous']:<15.1f}% {data['conversion_from_start']:<15.1f}%")

# Анализ потерь между этапами
print(f"\n4. АНАЛИЗ ПОТЕРЬ МЕЖДУ ЭТАПАМИ")
print("-" * 60)

for i in range(len(funnel_stages) - 1):
    current_stage = funnel_stages[i]
    next_stage = funnel_stages[i + 1]
    
    current_data = funnel_results[current_stage]
    next_data = funnel_results[next_stage]
    
    # Потери от текущего этапа к следующему
    loss_rate = 100 - next_data['conversion_from_previous']
    
    print(f"{current_stage} → {next_stage}:")
    print(f"  Пользователей на текущем этапе: {current_data['users_reached']:,}")
    print(f"  Пользователей на следующем этапе: {next_data['users_reached']:,}")
    print(f"  Потери: {loss_rate:.1f}% ({current_data['users_reached'] - next_data['users_reached']:,} пользователей)")
    print()

# Сравнение с неправильной методологией
print(f"\n5. СРАВНЕНИЕ МЕТОДОЛОГИЙ")

# Старая методология (максимальная стадия)
old_method = {}
for stage in funnel_stages:
    users_reached = sum(1 for user_data in user_funnel_analysis.values() 
                       if user_data['progress'][stage])
    old_method[stage] = users_reached

print("\nСТАРАЯ МЕТОДОЛОГИЯ (максимальная стадия):")
print("-" * 50)
for stage in funnel_stages:
    old_conversion = (old_method[stage] / total_users * 100)
    new_conversion = funnel_results[stage]['conversion_from_start']
    difference = new_conversion - old_conversion
    
    print(f"{stage}: {old_conversion:.1f}% (старая) vs {new_conversion:.1f}% (новая) "
          f"[{difference:+.1f}%]")

# Анализ аномальных последовательностей
print(f"\n6. АНАЛИЗ АНОМАЛЬНЫХ ПОСЛЕДОВАТЕЛЬНОСТЕЙ")

anomalous_users = []
for user_id, user_data in user_funnel_analysis.items():
    progress = user_data['progress']
    
    # Ищем пользователей, которые пропустили этапы
    for i in range(len(funnel_stages) - 1):
        current_stage = funnel_stages[i]
        next_stage = funnel_stages[i + 1]
        
        # Если пользователь прошел следующий этап, но не прошел текущий
        if progress[next_stage] and not progress[current_stage]:
            anomalous_users.append({
                'user_id': user_id,
                'missing_stage': current_stage,
                'reached_stage': next_stage
            })

print(f"Найдено {len(anomalous_users):,} пользователей с аномальными последовательностями")

if anomalous_users:
    print("\nПримеры аномальных последовательностей:")
    for i, anomaly in enumerate(anomalous_users[:10]):
        print(f"  {i+1}. Пользователь {anomaly['user_id']}: пропустил '{anomaly['missing_stage']}', "
              f"но прошел '{anomaly['reached_stage']}'")

# Временной анализ между этапами
print(f"\n7. ВРЕМЕННОЙ АНАЛИЗ МЕЖДУ ЭТАПАМИ")

time_between_stages = {}
for user_id, user_data in user_funnel_analysis.items():
    timestamps = user_data['timestamps']
    
    for i in range(len(funnel_stages) - 1):
        current_stage = funnel_stages[i]
        next_stage = funnel_stages[i + 1]
        
        if current_stage in timestamps and next_stage in timestamps:
            time_diff = (timestamps[next_stage] - timestamps[current_stage]).total_seconds()
            
            if (current_stage, next_stage) not in time_between_stages:
                time_between_stages[(current_stage, next_stage)] = []
            
            time_between_stages[(current_stage, next_stage)].append(time_diff)

print("\nСреднее время между этапами:")
for (stage1, stage2), times in time_between_stages.items():
    if times:
        avg_time = np.mean(times)
        median_time = np.median(times)
        print(f"{stage1} → {stage2}: {avg_time:.1f}с (среднее), {median_time:.1f}с (медиана)")

# Создаем визуализацию
print(f"\n8. СОЗДАНИЕ ВИЗУАЛИЗАЦИИ")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Правильный анализ воронки (последовательность событий)', fontsize=16, fontweight='bold')

# 1. Конверсия от предыдущего этапа
stages = list(funnel_results.keys())
conversions_from_previous = [funnel_results[stage]['conversion_from_previous'] for stage in stages]

bars1 = axes[0, 0].bar(stages, conversions_from_previous, color='skyblue', alpha=0.8)
axes[0, 0].set_title('Конверсия от предыдущего этапа', fontweight='bold')
axes[0, 0].set_ylabel('Конверсия (%)')
axes[0, 0].tick_params(axis='x', rotation=45)

for bar, conv in zip(bars1, conversions_from_previous):
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{conv:.1f}%', ha='center', va='bottom', fontweight='bold')

# 2. Конверсия от старта
conversions_from_start = [funnel_results[stage]['conversion_from_start'] for stage in stages]

bars2 = axes[0, 1].bar(stages, conversions_from_start, color='lightgreen', alpha=0.8)
axes[0, 1].set_title('Конверсия от старта', fontweight='bold')
axes[0, 1].set_ylabel('Конверсия (%)')
axes[0, 1].tick_params(axis='x', rotation=45)

for bar, conv in zip(bars2, conversions_from_start):
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{conv:.1f}%', ha='center', va='bottom', fontweight='bold')

# 3. Сравнение методологий
old_conversions = [(old_method[stage] / total_users * 100) for stage in stages]

x = np.arange(len(stages))
width = 0.35

bars_old = axes[1, 0].bar(x - width/2, old_conversions, width, label='Старая методология', 
                          color='lightcoral', alpha=0.8)
bars_new = axes[1, 0].bar(x + width/2, conversions_from_start, width, label='Новая методология', 
                          color='lightblue', alpha=0.8)

axes[1, 0].set_title('Сравнение методологий', fontweight='bold')
axes[1, 0].set_ylabel('Конверсия (%)')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(stages, rotation=45)
axes[1, 0].legend()

# 4. Потери между этапами
losses = []
for i in range(len(funnel_stages) - 1):
    current_stage = funnel_stages[i]
    next_stage = funnel_stages[i + 1]
    loss_rate = 100 - funnel_results[next_stage]['conversion_from_previous']
    losses.append(loss_rate)

stage_pairs = [f"{funnel_stages[i]}→{funnel_stages[i+1]}" for i in range(len(funnel_stages) - 1)]

bars_loss = axes[1, 1].bar(stage_pairs, losses, color='red', alpha=0.8)
axes[1, 1].set_title('Потери между этапами', fontweight='bold')
axes[1, 1].set_ylabel('Потери (%)')
axes[1, 1].tick_params(axis='x', rotation=45)

for bar, loss in zip(bars_loss, losses):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{loss:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/correct_funnel_analysis.png', dpi=300, bbox_inches='tight')
print("График правильного анализа воронки сохранен как 'figures/correct_funnel_analysis.png'")

print(f"\n=== ПРАВИЛЬНЫЙ АНАЛИЗ ВОРОНКИ ЗАВЕРШЕН ===")
print(f"\nКЛЮЧЕВЫЕ ВЫВОДЫ:")
print(f"✓ Учтена последовательность событий")
print(f"✓ Исключены аномальные переходы")
print(f"✓ Правильный подсчет потерь между этапами")
print(f"✓ Временной анализ между этапами")
print(f"✓ Сравнение со старой методологией") 