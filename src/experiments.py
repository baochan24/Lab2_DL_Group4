import os
import json
import glob
import yaml
import pandas as pd

def load_all_configs(config_dir="configs"):
    """Đọc tất cả các file cấu hình YAML từ thư mục configs/"""
    configs = {}
    yaml_files = glob.glob(os.path.join(config_dir, "*.yaml"))
    for file_path in yaml_files:
        run_id = os.path.basename(file_path).replace(".yaml", "")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                configs[run_id] = yaml.safe_load(f)
        except Exception as e:
            print(f"[Warning] Không thể đọc file cấu hình {file_path}: {e}")
    return configs

def collect_experiment_summary(outputs_runs_dir="outputs/runs", config_dir="configs"):
    """
    Quét qua các thư mục chạy trong outputs/runs/ để đọc file metrics.json
    và tạo bảng tổng hợp outputs/experiment_summary.csv theo quy ước TV5.
    """
    configs = load_all_configs(config_dir)
    summary_list = []

    for run_id in sorted(configs.keys()):
        cfg = configs[run_id] or {}
        exp_name = cfg.get("experiment_name", run_id)
        metrics_file = os.path.join(outputs_runs_dir, exp_name, "metrics.json")

        best_val_loss = None
        best_val_acc = None
        test_acc = None
        training_time = None

        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
                    best_val_loss = metrics_data.get("best_val_loss", metrics_data.get("val_loss"))
                    best_val_acc = metrics_data.get("best_val_acc", metrics_data.get("val_acc"))
                    test_acc = metrics_data.get("test_acc")
                    training_time = metrics_data.get("training_time_sec")
            except Exception as e:
                print(f"[Warning] Lỗi khi mở {metrics_file}: {e}")

        summary_list.append({
            "experiment_name": exp_name,
            "model_name": cfg.get("model_name"),
            "strategy": cfg.get("strategy"),
            "learning_rate": cfg.get("learning_rate"),
            "batch_size": cfg.get("batch_size"),
            "epochs": cfg.get("epochs"),
            "seed": cfg.get("seed"),
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_acc,
            "test_accuracy": test_acc,
            "training_time_sec": training_time
        })

    df = pd.DataFrame(summary_list)
    return df

def select_best_experiment(df_summary, selection_output_path="outputs/best_model_selection.json"):
    """
    Chọn mô hình có best_val_accuracy cao nhất để bàn giao cho TV4 đánh giá tập Test.
    """
    valid_runs = df_summary.dropna(subset=["best_val_accuracy"])
    
    if valid_runs.empty:
        print("[Notice] Chưa có kết quả thực nghiệm thực tế (metrics.json).")
        selection_data = {
            "status": "pending",
            "message": "Vui lòng đợi TV3 hoàn tất quá trình huấn luyện các Run."
        }
    else:
        sorted_df = valid_runs.sort_values(by=["best_val_accuracy", "best_val_loss"], ascending=[False, True])
        best_row = sorted_df.iloc[0]
        
        selection_data = {
            "status": "selected",
            "selected_run": best_row["experiment_name"],
            "model_name": best_row["model_name"],
            "strategy": best_row["strategy"],
            "best_val_accuracy": float(best_row["best_val_accuracy"]),
            "best_val_loss": float(best_row["best_val_loss"]) if pd.notna(best_row["best_val_loss"]) else None,
            "checkpoint_path": f"outputs/checkpoints/{best_row['experiment_name']}/best_model.pth"
        }

    os.makedirs(os.path.dirname(selection_output_path), exist_ok=True)
    with open(selection_output_path, "w", encoding="utf-8") as f:
        json.dump(selection_data, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Đã ghi nhận lựa chọn mô hình tại: {selection_output_path}")
    return selection_data

def run_tv5_pipeline():
    """Chạy quy trình thu thập kết quả của TV5."""
    print("=== TV5: BẮT ĐẦU THU THẬP THỰC NGHIỆM VÀ XUẤT EXPERIMENT SUMMARY ===")
    df_summary = collect_experiment_summary()
    
    os.makedirs("outputs", exist_ok=True)
    summary_path = "outputs/experiment_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"✅ Đã xuất bảng tổng hợp kết quả ra file: {summary_path}")
    print(df_summary.to_string())
    
    select_best_experiment(df_summary)

if __name__ == "__main__":
    run_tv5_pipeline()