# Тестовое задание Data Analyst - Simple App/Palta

## Результаты анализа

### Ключевые выводы:
- **Критическая проблема**: 92% потерь на этапе paywall_show → payment_done
- **Лучший эксперимент**: exp_2 с конверсией 9.1% (в 1.7 раза выше среднего)
- **Общая конверсия**: 5.4% от onboarding_start до payment_done
- **Различия по воронкам**: Main воронка показывает лучшую конверсию (6.8%)

## Структура проекта

```
Palta_Simple/
│
├── data/
│   └── simple_interview_events.csv
│
├── scripts/
│   ├── analyze_onboarding.py          # Базовый анализ онбординга
│   ├── analyze_funnel_types.py        # Анализ типов воронок
│   ├── analyze_experiments.py         # Анализ экспериментов
│   ├── analyze_experiments_EN.py      # Анализ экспериментов (EN)
│   ├── visualize_funnel.py            # Визуализация воронки
│   ├── visualize_funnel_EN.py         # Визуализация воронки (EN)
│   ├── enhanced_analysis.py           # Улучшенный анализ с CI и качеством данных
│   ├── segmentation_analysis.py       # Анализ сегментации пользователей
│   └── correct_funnel_analysis.py     # Правильный анализ воронки (последовательность событий)
│
├── reports/
│   ├── Simple App test assignment_RU.docx
│   ├── Simple App test assignment_EN.docx
│   ├── Simple App test assignment_RU.pdf
│   └── Simple App test assignment_EN.pdf
│
├── figures/
│   ├── funnel_analysis.png             # Анализ воронки
│   ├── funnel_analysis_EN.png          # Анализ воронки (EN)
│   ├── funnel_comparison.png           # Сравнение воронок
│   ├── funnel_comparison_EN.png        # Сравнение воронок (EN)
│   ├── experiment_analysis.png         # Анализ экспериментов
│   ├── experiment_analysis_EN.png      # Анализ экспериментов (EN)
│   ├── experiment_lift.png             # Lift по экспериментам
│   ├── experiment_lift_EN.png          # Lift по экспериментам (EN)
│   ├── enhanced_experiment_analysis.png # Улучшенный анализ с CI
│   ├── weekly_conversion_trend.png     # Динамика конверсии по неделям
│   ├── segmentation_analysis.png       # Анализ сегментации
│
├── README.md
├── README_EN.md
└── requirements.txt
```

- Все данные — в папке `data/`
- Все Python-скрипты — в папке `scripts/`
- Все отчёты (RU/EN) — в папке `reports/`
- Все графики и визуализации — в папке `figures/`
- Основная документация и зависимости — в корне проекта.

## Ответы на задания

### 1. События онбординга

#### Конверсия по этапам:
- onboarding_start → profile_start: 83.1% (потеря 16.9%)
- profile_start → email_submit: 72.5% (потеря 12.6%)
- email_submit → paywall_show: 67.5% (потеря 7.0%)
- paywall_show → payment_done: 5.4% (потеря 92.0%)

#### Топ-3 перспективных эксперимента:
1. **exp_2**: 9.1% конверсии (5,731 пользователей) - статистически значимый lift 49.8%
2. **exp_6**: 6.8% конверсии (4,586 пользователей) - положительный эффект
3. **exp_9**: 6.5% конверсии (4,173 пользователей) - положительный эффект

### 2. Ежедневные задания

#### Предлагаемые метрики для оценки:
- **Acceptance Rate** - доля принятия предложенных заданий
- **Completion Rate** - доля выполненных заданий
- **Replacement Rate** - доля запросов на замену
- **Retention Impact** - влияние на удержание
- **Engagement Score** - уровень вовлеченности

#### Улучшения на основе анализа онбординга:
- Персонализация по типам воронок (female/male/main)
- Интеграция с процессом до paywall
- Применение успешных элементов из экспериментов

### 3. Оценка перспективности

#### Методология оценки:
1. **Анализ поведения**: изучение паттернов использования
2. **Количественные исследования**: опросы, анализ метрик
3. **Прототипирование**: MVP, A/B тестирование

#### Критерии успеха:
- Acceptance Rate > 60%
- Completion Rate > 40%
- Retention Impact > +15%
- Engagement Score > +25%

## Детальный анализ типов воронок

### Распределение пользователей:
- **Female**: 55,000 пользователей (55%)
- **Male**: 35,000 пользователей (35%)
- **Main**: 10,000 пользователей (10%)

### Конверсия по типам воронок:
- **Main**: 6.8% (лучший результат)
- **Female**: 5.2% (средний результат)
- **Male**: 4.8% (низший результат)

### Потери по этапам:
- **Main**: стабильные потери на всех этапах
- **Female**: наибольшие потери на profile_start → email_submit
- **Male**: критические потери на paywall_show → payment_done

## Рекомендуемые действия

### Немедленные (1-2 недели):
1. Анализ причин 92% потерь на paywall
2. Изучение успешных элементов exp_2
3. Планирование A/B тестов

### Среднесрочные (1-2 месяца):
1. Разработка MVP ежедневных заданий
2. Персонализация воронок
3. Пилотный запуск для 5% пользователей

### Долгосрочные (3-6 месяцев):
1. Полная редизайн воронки
2. Геймифицированная система заданий
3. AI-персонализация

## Ожидаемые результаты

### Оптимизация онбординга:
- Увеличение конверсии с 5.4% до 8-10%
- Дополнительный доход: +50-85%

### Ежедневные задания:
- Увеличение retention на 15-25%
- Повышение времени в приложении на 20-30%

## 🛠 Технические детали

### Запуск анализа:
```bash
# Основной анализ онбординга
python scripts/analyze_onboarding.py

# Анализ экспериментов
python scripts/analyze_experiments.py

# Анализ типов воронок
python scripts/analyze_funnel_types.py

# Создание визуализаций
python scripts/visualize_funnel.py

# Улучшенный анализ с доверительными интервалами
python scripts/enhanced_analysis.py

# Анализ сегментации пользователей
python scripts/segmentation_analysis.py

# Правильный анализ воронки (последовательность событий)
python scripts/correct_funnel_analysis.py
```

### Требования:
- Python 3.7+
- pandas, numpy, matplotlib, seaborn, scipy

### Структура данных:
- **user_id**: уникальный идентификатор пользователя
- **event_type**: тип события (onboarding_start, profile_start, etc.)
- **event_time**: время события
- **event_params**: JSON с дополнительными параметрами

## Улучшения анализа

### Новые возможности:
- **Доверительные интервалы**: 95% CI для всех A/B тестов
- **Анализ качества данных**: проверка дубликатов, подозрительных пользователей
- **Временной анализ**: динамика конверсии по неделям
- **Оценка мощности тестов**: статистическая значимость и размер эффекта
- **Сегментация**: анализ по платформам, устройствам, странам
- **Улучшенная метрика**: конверсия по каждому этапу, а не только максимальному

### Ключевые улучшения:
- **exp_2**: статистически значимый lift +49.8% (p < 0.001)
- **exp_6**: положительный эффект +15.5%, но незначимый (p = 0.39)
- **exp_9**: положительный эффект +9.1%, но незначимый (p = 0.56)
- **Качество данных**: 0% дубликатов, чистые данные
- **Временная стабильность**: конверсия стабильна по неделям
- **Правильная воронка**: учет последовательности событий вместо максимальной стадии

## Дополнительные инсайты

### По типам воронок:
- **Female** (55% пользователей): стабильная конверсия, фокус на планировании
- **Male** (35% пользователей): наибольшие потери, нужны простые инструкции
- **Main** (10% пользователей): лучшая конверсия, универсальный подход

### Временные паттерны:
- Данные за 3 месяца (январь-апрель 2024)
- Стабильные паттерны конверсии
- Возможность для сезонного анализа

### Статистическая значимость:
- exp_2 показал статистически значимое улучшение (p < 0.05)
- exp_6 и exp_9 показали положительные эффекты, но не достигли статистической значимости
- Остальные эксперименты не показали значимых улучшений

---

*Анализ проведен на основе данных 100,000 пользователей с использованием Python и современных методов аналитики* 