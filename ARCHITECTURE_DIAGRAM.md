# System Architecture Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   predict.py (Orchestrator)                 │
│                                                             │
│  - load_data()        → Loads CSV with windows              │
│  - prepare_window()   → Converts to numpy OHLCV             │
│  - predict_single()   → Gets prediction from model          │
│  - evaluate()         → Calculates metrics (accuracy, F1)   │
│  - predict()          → Makes single prediction             │
│  - main()             → CLI interface                       │
└────────────────────────┬────────────────────────────────────┘
                         │ uses
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              model_registry.py (Central Hub)                │
│                                                             │
│  ModelRegistry:                                             │
│  - register()     → Register new model classes              │
│  - get_model()    → Instantiate/retrieve model              │
│  - list_models()  → List available models                   │
│                                                             │
│  Registered Models:                                         │
│  • 'timesfm'      → TimesFmModel                            │
│  • 'lag_llama'    → LagLlamaModel                           │
│  • 'custom_*'     → User-defined models                     │
└───────────┬─────────────────┬─────────────────────┬───────-─┘
            │                 │                     │
   ┌────────↓────────┐  ┌─────↓─────────┐     ┌─────↓──────┐
   │                 │  │               │     │            │
   ↓                 ↓  ↓               ↓     ↓            ↓
   ┌─────────────────┐ ┌──────────────────┐ ┌──────────────────┐
   │ timesfm_model.py│ │Lag_Llama_model.py│ │  Custom Models   │
   └─────────────────┘ └──────────────────┘ └──────────────────┘
   │ TimesFmModel    │ │ LagLlamaModel    │ │ MyNewModel       │
   │ (implements     │ │ (implements      │ │ (implements      │
   │ PredictionModel)│ │  PredictionModel)│ │  PredictionModel)│
   └─────────────────┘ └──────────────────┘ └──────────────────┘
```

## Data Flow Diagram

```
┌──────────────┐
│  CSV File    │  (prepared_data.csv)
│ windows: str │  └─ Column "window": stringified list
│ labels: str  │     └─ Example: "[[2024-01-01, 100, 105, 95, 102, 1000], ...]"
└──────┬───────┘
       │
       │ load_data()
       ↓
┌─────────────────────────────────┐
│    pandas DataFrame             │
│  window | label                 │
│  [list] | 'UP'/'DOWN'/'SIDEWAY' │
└──────┬────────────────────────┬─┘
       │                        │
       │ prepare_window()       │ derive_label()
       ↓                        │
┌────────────────────┐          │
│ numpy array (n, 5) │          │
│ [O, H, L, C, V]    │          │
└────────┬───────────┘          │
         │                      │
         │ model.predict()      │
         ↓                      │
┌──────────────────────┐        │
│ Prediction dict      │        │
│ {high, low, ...}     │        │
└────────┬─────────────┘        │
         │                      │
         │ derive_label()  ←────┘
         │
         ↓
┌────────────────────┐
│ Predicted Label    │
│ 'UP'/'DOWN'/...    │
└────────┬───────────┘
         │
         │ accuracy_score(), f1_score()
         ↓
┌──────────────────────┐
│ Metrics              │
│ accuracy, f1, count  │
└──────────────────────┘
```

## Model Interface Contract

```
┌─────────────────────────────────────────────────────────┐
│         PredictionModel (Abstract Base Class)           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  @abstractmethod                                        │
│  def predict(window: np.ndarray) -> dict:              │
│      """                                               │
│      Input:  window shape (n, 5)                       │
│              columns: [Open, High, Low, Close, Volume] │
│              dtype: float32                            │
│                                                        │
│      Output: dict with keys:                          │
│        - 'high': float (predicted high price)        │
│        - 'low': float (predicted low price)          │
│        - 'open': float (optional, last open)         │
│        - 'close': float (optional, last close)       │
│      """                                              │
│      pass                                             │
│                                                        │
└─────────────────────────────────────────────────────────┘
                          ↑
                ┌─────────┴─────────┬──────────────┐
                │                   │              │
         ┌──────────────┐  ┌────────────────┐  ┌────────┐
         │ TimesFmModel │  │ LagLlamaModel  │  │ Custom │
         └──────────────┘  └────────────────┘  └────────┘
```

## Extensibility Flow

```
Step 1: Design                      Step 2: Implement
┌──────────────┐                    ┌─────────────────────┐
│ Your Model   │ ──inheritance──→  │ class MyModel(      │
│ Concept      │                    │   PredictionModel): │
└──────────────┘                    │   def predict(...): │
                                    │     ...             │
                                    └──────┬──────────────┘
                                           │
                                           │ Step 3: Register
                                           ↓
                                    ┌──────────────────┐
                                    │ ModelRegistry    │
                                    │ .register(       │
                                    │  'mymodel',      │
                                    │  MyModel)        │
                                    └──────┬───────────┘
                                           │
                                           │ Step 4: Use
                                           ↓
                                    ┌──────────────────┐
                                    │ predict.py       │
                                    │ --model mymodel  │
                                    │ --data ...       │
                                    │ --eval           │
                                    └──────────────────┘
```

## Command Flow

```
Terminal Command:
│
├─ --help              → Show command help
│
├─ --eval              → Evaluation Mode
│  │
│  ├─ Load data (CSV)
│  ├─ Get model from registry
│  ├─ For each sample:
│  │  ├─ Prepare window
│  │  ├─ Call model.predict()
│  │  ├─ Derive label
│  ├─ Calculate metrics
│  └─ Output JSON (optional)
│
└─ --predict           → Single Prediction Mode
   │
   ├─ Load data (CSV)
   ├─ Get model from registry
   ├─ Get sample at index
   ├─ Prepare window
   ├─ Call model.predict()
   ├─ Derive label
   └─ Output JSON (optional)
```

## Model Lifecycle

```
┌─────────────────────┐
│  ModelRegistry      │
│  .get_model(name)   │
└──────────┬──────────┘
           │
           ├─ Check if in cache
           │  ├─ Yes → Return cached instance
           │  └─ No ↓
           │
           ├─ Look up in _models dict
           ├─ Instantiate class
           ├─ Cache instance
           └─ Return instance

┌─────────────────────┐
│  Model Instance     │
│  (Now ready to use) │
└──────────┬──────────┘
           │
           ├─ Accept window data
           ├─ Process through model
           └─ Return predictions
```

## File Structure

```
llm-market/
├── src/
│   ├── predict.py                 ← Top-level orchestrator (240 lines)
│   ├── model_registry.py          ← Model management (69 lines) NEW
│   ├── timesfm_model.py           ← TimesFM implementation (88 lines)
│   ├── utils.py                   ← Label derivation
│   ├── data_prep.py
│   ├── finetune.py
│   ├── __pycache__/
│   └── MODEL_ARCHITECTURE.md      ← Architecture docs NEW
├── Lag_Llama_model.py             ← Lag-Llama implementation
├── data/
│   └── prepared_data.csv
├── REFACTORING_SUMMARY.md         ← This summary NEW
├── QUICK_START.md                 ← Quick reference NEW
├── EXAMPLE_NEW_MODEL.md           ← Extension example NEW
└── ...
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Abstract Base Class** | Ensures interface consistency across all models |
| **Model Registry** | Centralizes model management; enables easy registration |
| **Model Caching** | Avoids redundant initialization; models are stateful |
| **OHLCV Windows** | Standard financial data format; consistent across models |
| **Dict Return Format** | Flexible; supports extending with additional fields |
| **Separate Model Files** | Each model is independent; easy to maintain/test |
| **CLI Orchestrator** | Single entry point; hides complexity from users |

## Error Handling

```
predict.py --model unknown_model
           │
           └─→ ModelRegistry.get_model('unknown_model')
               │
               └─→ ValueError: Model 'unknown_model' not found.
                   Available models: timesfm, lag_llama
```

This architecture ensures **robustness**, **extensibility**, and **maintainability**.
