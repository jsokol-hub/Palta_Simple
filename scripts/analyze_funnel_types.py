import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Настройка для корректного отображения русских символов
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
sns.set_style("whitegrid")

def load_data():
    """Загружает данные"""
    print("Загрузка данных...")
    df = pd.read_csv('simple_interview_events.csv')
    df['event_time'] = pd.to_datetime(df['event_time'])
    return df

def extract_funnel_type(df):
    """Извлекает тип воронки для каждого пользователя"""
    print("Извлечение типов воронок...")
    
    # Находим тип воронки для каждого пользователя
    funnel_types = {}
    
    for user_id in df['user_id'].unique():
        user_events = df[df['user_id'] == user_id]
        onboarding_events = user_events[user_events['event_type'] == 'onboarding_start']
        
        if len(onboarding_events) > 0:
            # Берем первое событие onboarding_start
            first_onboarding = onboarding_events.iloc[0]
            event_params = first_onboarding['event_params']
            
            try:
                if pd.notna(event_params) and event_params != '{}':
                    params = eval(event_params)
                    funnel_type = params.get('funnel_type', 'unknown')
                else:
                    funnel_type = 'main'
            except:
                funnel_type = 'main'
            
            funnel_types[user_id] = funnel_type
    
    return funnel_types

def analyze_funnel_by_type_corrected(df, funnel_types):
    """Анализирует воронку для каждого типа с правильной логикой"""
    print("Анализ воронки по типам (исправленная логика)...")
    
    funnel_stages = ['onboarding_start', 'profile_start', 'email_submit', 'paywall_show', 'payment_done']
    
    results = {}
    
    for funnel_type in ['female', 'male', 'main']:
        # Получаем пользователей данного типа
        type_users = [user_id for user_id, f_type in funnel_types.items() if f_type == funnel_type]
        
        if not type_users:
            continue
            
        # Фильтруем данные для этого типа
        type_df = df[df['user_id'].isin(type_users)]
        
        # Анализируем воронку
        user_progress = {}
        for user_id in type_users:
            user_events = type_df[type_df['user_id'] == user_id]['event_type'].tolist()
            max_stage = None
            
            for stage in funnel_stages:
                if stage in user_events:
                    max_stage = stage
            
            if max_stage:
                user_progress[user_id] = max_stage
        
        # Подсчитываем конверсию правильно
        total_users = len(type_users)
        stage_counts = {}
        
        for stage in funnel_stages:
            # Считаем пользователей, которые достигли этой стадии или прошли дальше
            count = sum(1 for progress in user_progress.values() 
                       if progress in funnel_stages[funnel_stages.index(stage):])
            stage_counts[stage] = count
        
        # Рассчитываем конверсию
        conversions = []
        for stage in funnel_stages:
            conversion = (stage_counts[stage] / total_users) * 100
            conversions.append(conversion)
        
        results[funnel_type] = {
            'total_users': total_users,
            'conversions': conversions,
            'stage_counts': stage_counts
        }
    
    return results

def create_comparison_visualization(results):
    """Создает визуализацию сравнения воронок"""
    print("Создание визуализации сравнения...")
    
    funnel_stages = ['onboarding_start', 'profile_start', 'email_submit', 'paywall_show', 'payment_done']
    stage_labels = ['Старт', 'Профиль', 'Email', 'Paywall', 'Оплата']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # График 1: Конверсия по типам воронок
    x = np.arange(len(stage_labels))
    width = 0.25
    
    for i, (funnel_type, color) in enumerate([('female', 'pink'), ('male', 'lightblue'), ('main', 'lightgreen')]):
        if funnel_type in results:
            conversions = results[funnel_type]['conversions']
            ax1.bar(x + i*width, conversions, width, label=funnel_type, color=color, alpha=0.7)
    
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(stage_labels, fontsize=10)
    ax1.set_ylabel('Конверсия (%)', fontsize=12)
    ax1.set_title('Конверсия по типам воронок', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # График 2: Потери между этапами
    loss_data = []
    labels = []
    
    for funnel_type in ['female', 'male', 'main']:
        if funnel_type in results:
            conversions = results[funnel_type]['conversions']
            for i in range(len(conversions) - 1):
                loss = conversions[i] - conversions[i + 1]
                loss_data.append(loss)
                labels.append(f"{funnel_type}\n{stage_labels[i]}→{stage_labels[i+1]}")
    
    bars = ax2.bar(range(len(loss_data)), loss_data, color=['pink', 'pink', 'pink', 'pink', 
                                                           'lightblue', 'lightblue', 'lightblue', 'lightblue',
                                                           'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen'], alpha=0.7)
    
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax2.set_ylabel('Потери (%)', fontsize=12)
    ax2.set_title('Потери между этапами по типам воронок', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # График 3: Размер выборки
    funnel_types = list(results.keys())
    sizes = [results[ft]['total_users'] for ft in funnel_types]
    colors = ['pink', 'lightblue', 'lightgreen']
    
    bars = ax3.bar(funnel_types, sizes, color=colors, alpha=0.7)
    
    for bar, size in zip(bars, sizes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                f'{size:,}', ha='center', va='bottom', fontsize=10)
    
    ax3.set_ylabel('Количество пользователей', fontsize=12)
    ax3.set_title('Размер выборки по типам воронок', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # График 4: Финальная конверсия
    final_conversions = [results[ft]['conversions'][-1] for ft in funnel_types]
    
    bars = ax4.bar(funnel_types, final_conversions, color=colors, alpha=0.7)
    
    for bar, conv in zip(bars, final_conversions):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{conv:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax4.set_ylabel('Конверсия в оплату (%)', fontsize=12)
    ax4.set_title('Финальная конверсия по типам воронок', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('funnel_types_comparison_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Визуализация сохранена как 'funnel_types_comparison_fixed.png'")

def statistical_comparison(results):
    """Проводит статистическое сравнение"""
    print("\n=== СТАТИСТИЧЕСКОЕ СРАВНЕНИЕ ===")
    
    if 'female' in results and 'male' in results:
        female_conv = results['female']['conversions'][-1]  # Финальная конверсия
        male_conv = results['male']['conversions'][-1]
        
        print(f"Конверсия female: {female_conv:.1f}%")
        print(f"Конверсия male: {male_conv:.1f}%")
        print(f"Разница: {abs(female_conv - male_conv):.1f}%")
        
        if female_conv > male_conv:
            print("✅ Female воронка показывает лучшую конверсию")
        elif male_conv > female_conv:
            print("✅ Male воронка показывает лучшую конверсию")
        else:
            print("✅ Конверсия одинаковая")
    
    if 'main' in results:
        main_conv = results['main']['conversions'][-1]
        print(f"\nКонверсия main: {main_conv:.1f}%")
        
        if 'female' in results:
            female_conv = results['female']['conversions'][-1]
            diff = main_conv - female_conv
            print(f"Main vs Female: {diff:+.1f}% ({'лучше' if diff > 0 else 'хуже'})")
        
        if 'male' in results:
            male_conv = results['male']['conversions'][-1]
            diff = main_conv - male_conv
            print(f"Main vs Male: {diff:+.1f}% ({'лучше' if diff > 0 else 'хуже'})")

def print_detailed_results(results):
    """Выводит детальные результаты"""
    print(f"\n{'='*80}")
    print("ДЕТАЛЬНЫЙ АНАЛИЗ ТИПОВ ВОРОНОК (ИСПРАВЛЕННАЯ ЛОГИКА)")
    print(f"{'='*80}")
    
    funnel_stages = ['onboarding_start', 'profile_start', 'email_submit', 'paywall_show', 'payment_done']
    
    for funnel_type in ['female', 'male', 'main']:
        if funnel_type not in results:
            continue
            
        data = results[funnel_type]
        print(f"\n📊 ВОРОНКА '{funnel_type.upper()}' ({data['total_users']:,} пользователей)")
        print("-" * 60)
        
        for i, stage in enumerate(funnel_stages):
            conversion = data['conversions'][i]
            count = data['stage_counts'][stage]
            print(f"  {stage}: {count:,} пользователей ({conversion:.1f}%)")
        
        # Потери между этапами
        print("\n  📉 Потери между этапами:")
        for i in range(len(funnel_stages) - 1):
            current_conv = data['conversions'][i]
            next_conv = data['conversions'][i + 1]
            loss = current_conv - next_conv
            print(f"    {funnel_stages[i]} → {funnel_stages[i+1]}: -{loss:.1f}%")

def main():
    """Основная функция"""
    # Загружаем данные
    df = load_data()
    
    # Извлекаем типы воронок
    funnel_types = extract_funnel_type(df)
    
    # Анализируем воронку по типам
    results = analyze_funnel_by_type_corrected(df, funnel_types)
    
    # Создаем визуализацию
    create_comparison_visualization(results)
    
    # Выводим результаты
    print_detailed_results(results)
    
    # Статистическое сравнение
    statistical_comparison(results)
    
    print(f"\n{'='*80}")
    print("АНАЛИЗ ЗАВЕРШЕН")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 