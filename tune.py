import sys
import copy
from pathlib import Path
import optuna

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from src.utils import load_config
from src.train import main as train_model

# Configuração escolhida para a etapa de otimização de hiperparâmetros (E4)
CHOSEN_CONFIG_PATH = "configs/reg_e4_random_erasing.yaml"

def objective(trial):
    base_config = load_config(CHOSEN_CONFIG_PATH)
    
    config = copy.deepcopy(base_config)
    
    # Learning Rate
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    config["training"]["lr"] = lr
    
    # Weight Decay
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    config["training"]["weight_decay"] = weight_decay
    
    # Label Smoothing
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.3)
    config["training"]["label_smoothing"] = label_smoothing
    
    # Random Erasing
    random_erasing_p = trial.suggest_float("random_erasing_p", 0.0, 0.5)
    config["dataset"]["random_erasing_p"] = random_erasing_p
    
    # Mixup Alpha
    mixup_alpha = trial.suggest_float("mixup_alpha", 0.0, 1.0)
    config["training"]["mixup_alpha"] = mixup_alpha
    
    trial_name = f"optuna_trial_{trial.number}"
    config["experiment"]["name"] = trial_name
    
    config["paths"]["checkpoint_dir"] = f"checkpoints/{trial_name}"
    config["paths"]["results_dir"] = f"results/{trial_name}"
    
    print(f"\n[{'-'*10} INICIANDO TRIAL {trial.number} {'-'*10}]")
    best_val_acc = train_model(config)
    
    return best_val_acc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="vit_tuning")
    
    study.optimize(objective, n_trials=20)
    
    print("\n" + "="*40)
    print("OTIMIZAÇÃO CONCLUÍDA!")
    print("Melhor Acurácia:", study.best_value)
    print("Melhores Hiperparâmetros:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("="*40)