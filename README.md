# Ансамблевый краткосрочный прогноз добычи

Python-система для прогноза добычи жидкости, нефти и газа по скважинам и месторождению. Она объединяет физические, статистические и ML-модели в байесовский ансамбль и показывает результаты в Streamlit.

## Состав

- `orchestrator/main_UI.py` — точка входа интерфейса.
- `orchestrator/UI` — страницы и обработка результатов.
- `orchestrator/tools_preprocessor` — подготовка входных таблиц.
- `orchestrator/frameworks_*`, `orchestrator/models_*` — расчетные модели.
- `orchestrator/statistics_explorer` — статистика и графики.
- `docs/PROJECT_OVERVIEW.md` — описание архитектуры и форматов.

## Запуск

Из корня репозитория:

```powershell
cd orchestrator
conda env create -f environment.yml
conda activate env_ksp
python .\models_transflow_siam\transflow\__compile_aot__.py
python -m streamlit run .\main_UI.py
```

Расчетным модулям требуются совместимые версии зависимостей из `environment.yml`. При запуске без входных данных прогнозы недоступны.

## Входные данные

Исходные данные месторождений **не включены** в этот репозиторий. Пользователь размещает собственные таблицы в `orchestrator/tools_preprocessor/data/<месторождение>/`. Ожидаемые имена и назначение файлов перечислены в [описании проекта](docs/PROJECT_OVERVIEW.md). Сохраненные состояния и результаты расчетов также не публикуются.

`.gitignore` исключает входные данные, результаты, локальные окружения и временные файлы из будущих коммитов.
