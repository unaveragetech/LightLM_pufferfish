# LightLM System - Simple Workflow

```mermaid
graph TD
    UI[User Interface] --> CONFIG[Configuration]
    CONFIG --> MODEL[Model Architecture]
    CONFIG --> DATA[Data Pipeline]
    DATA --> TRAIN[Training Process]
    MODEL --> TRAIN
    TRAIN --> CKPT[Checkpoints]
    CKPT --> GEN[Text Generation]
    
    subgraph "Configuration Details"
        CONFIG --> TC[TrainerConfig]
        CONFIG --> MC[ModelConfig]
        CONFIG --> OPT[Optimizer]
    end
    
    subgraph "Data Pipeline Details"
        DATA --> DS[Dataset Loading]
        DATA --> TDL[ThreadedDataLoader]
        DATA --> SUBSET[Subset Mode]
    end
    
    subgraph "Training Details"
        TRAIN --> EPOCH[Epoch Loop]
        TRAIN --> BATCH[Batch Processing]
        TRAIN --> LOSS[Loss Calculation]
    end
```

## Key Commands

### 1. Start the system
```bash
python train.py
```

### 2. Optimize for fast training
```
Select option 2 (Manage dataset/configuration)
Select option 5 (Optimize for training time)
Enter 0.001 hours
```

### 3. Train the model
```
Select option 1 (Start training)
```

### 4. Generate text
```bash
python generate.py
Enter your prompt
```
