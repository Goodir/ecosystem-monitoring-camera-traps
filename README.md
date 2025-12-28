# Мониторинг экосистемы по данным фотоловушек (Camera Traps)

## Что сделано
Прототип автоматизирует обработку изображений с фотоловушек:
- детектирует животных на кадре (bbox + confidence),
- классифицирует вид животного из заданного списка,
- считает метрики и формирует таблицу предсказаний для мониторинга.

## Данные
Hugging Face dataset: `jnle/wildlife_conservation_camera_trap_dataset`  
Разметка: YOLO (class_id, x_center, y_center, w, h; нормированные координаты).

В файле `wcs_camera_traps.zip` буду храниться мета данные по фотографиям, выгруженные с https://lila.science/datasets/wcscameratraps 

**Важно:** разархивировать `wcs_camera_traps.zip` необязательно, скрипт в ipynb сам все сделает

После скачивания ожидается структура:

### Структура проекта

```text
repo/
  project_wildlife_camera_shot.ipynb
  README.md
  requirements.txt
  wcs_camera_traps.zip
  src/
    download_dataset.py
  data/
    camera_trap/
        camera_trap_dataset/
            test/
                images/
                labels/
            full_dataset/
                images/
                labels/
            2024-08-17_3-Fold_Cross-val/
                kfold_datasplit.csv
                kfold_label_distribution.csv
                split_1/
                split_2/
                split_3/
        README.md
```
Нам потребуется только test и full_dataset

## Модели
- **MegaDetectorV6 (PytorchWildlife)** — детекция объектов на фотоловушках.
- **BioCLIP** — zero-shot классификация вида по crop-изображению (по текстовым названиям классов, HF:`imageomics/bioclip-2`).

Веса моделей скачиваются автоматически при первом запуске и сохраняются в кэш.

## Установка
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Скачивание датасета
```bash
python src/download_data.py
```

## Запуск
```bash
jupyter lab
```
Откройте ``project_wildlife_camera_shot.ipynb`` и выполните ячейки сверху вниз.

## Примечания

`DEVICE` для работы модели выбран mps, так как работа над проектам велась на mackbook, при других вводных используйте либо cuda, либо cpu. В случае последнего работа модели будет занимать большое количество времени
