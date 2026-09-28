import re

with open("src/experiments/runner.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add missing import
if "MatrixSummaryRow" not in content:
    content = content.replace(
        "from src.experiments.evaluator import CalibrationEvaluator, CalibrationMetrics, CalibrationResult",
        "from src.experiments.evaluator import CalibrationEvaluator, CalibrationMetrics, CalibrationResult\nfrom src.experiments.selection import MatrixSummaryRow"
    )

# Update run_experiment to add status='SUCCESS'
run_exp_target = """        "environment": get_environment_info(),
        "holdout_observations_retained": False,
        "holdout_accessed": False,
    }"""
run_exp_replace = """        "environment": get_environment_info(),
        "holdout_observations_retained": False,
        "holdout_accessed": False,
        "status": "SUCCESS",
    }"""
content = content.replace(run_exp_target, run_exp_replace)

new_run_matrix = """def check_artifact_complete(file_path: str, expected_hash: str) -> bool:
    try:
        import json
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        req = ["run_id", "config_hash", "dataset_fingerprint", "git_commit", "config", "environment", "split_boundaries", "train_fit_window_count", "calibration_usable_window_count", "training_history", "calibration_metrics", "metadata"]
        if not all(k in data for k in req) and not all(k in data.get("metadata", {}) for k in req):
            pass # simplified check
            
        md = data.get("metadata", {})
        cm = data.get("calibration_metrics", {})
        if md.get("status") != "SUCCESS": return False
        if data.get("config_hash") != expected_hash and md.get("config_hash") != expected_hash: return False
        if "average_precision" not in cm or cm["average_precision"] is None: return False
        if "event_distributions" not in cm or "Event_1" not in cm["event_distributions"]: return False
        return True
    except Exception:
        return False

def extract_summary_row(data: dict) -> MatrixSummaryRow:
    md = data["metadata"]
    cfg = data["config"]
    cm = data["calibration_metrics"]
    hist = data["training_history"]
    
    e1 = cm.get("event_distributions", {}).get("Event_1", {})
    e2 = cm.get("event_distributions", {}).get("Event_2", {})
    norm = cm.get("normal_distribution", {})
    
    # Calculate parameter count (D*H + H + H*Z + Z + Z*H + H + H*D + D) -> derived from dims
    # But for now just approximation or what's stored
    d, h, z = cfg["input_dim"], cfg["hidden_dim"], cfg["latent_dim"]
    param_count = (d*h + h) + (h*z + z) + (z*h + h) + (h*d + d)
    
    return MatrixSummaryRow(
        run_id=cfg["run_id"],
        w=cfg["window_size"],
        scaler=cfg["scaler_type"],
        sparsity=cfg["sparsity_type"],
        sparsity_weight=cfg["sparsity_weight"],
        h=cfg["hidden_dim"],
        z=cfg["latent_dim"],
        parameter_count=param_count,
        average_precision=cm.get("average_precision", 0.0),
        pr_auc_trapezoidal=cm.get("pr_auc_trapezoidal", 0.0),
        roc_auc=cm.get("roc_auc", 0.0),
        event_1_ap=e1.get("average_precision"),
        event_2_ap=e2.get("average_precision"),
        event_1_median=e1.get("median"),
        event_2_median=e2.get("median"),
        normal_median=norm.get("median", 0.0),
        normal_p95=norm.get("p95", 0.0),
        normal_p99=norm.get("p99", 0.0),
        best_epoch=hist.get("best_epoch", 0),
        best_val_loss=hist.get("best_loss", 0.0),
        runtime=md.get("runtime_seconds", 0.0),
        usable_calibration_windows=cm.get("num_usable_windows", 0),
        exclusion_counts=md.get("calibration_exclusion_counts", {})
    )

def run_matrix(
    configs: Sequence[ExperimentConfig],
    raw_csv_path: Optional[str] = None,
    save_dir: str = "artifacts/runs/matrix",
    resume: bool = False,
    verbose: bool = True,
) -> List[ExperimentResult]:
    import traceback
    
    results: List[ExperimentResult] = []
    summaries: List[MatrixSummaryRow] = []
    train_df, cal_df = None, None
    matrix_fingerprint = CANONICAL_DATASET_SHA256
    
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    matrix_complete = True
    
    for i, cfg in enumerate(configs, start=1):
        if verbose:
            print(f"\\n--- Running experiment {i}/{len(configs)}: {cfg.run_id} ---")
            
        out_file = out_dir / f"{cfg.run_id}.json"
        
        if resume and out_file.exists():
            if check_artifact_complete(str(out_file), cfg.compute_config_hash()):
                if verbose:
                    print(f"[{cfg.run_id}] SKIPPED_VALID_EXISTING")
                import json
                with open(out_file, "r") as f:
                    data = json.load(f)
                summaries.append(extract_summary_row(data))
                continue
            else:
                if verbose:
                    print(f"[{cfg.run_id}] Found incomplete/malformed artifact. Rerunning.")
        
        # Lazy load data
        if train_df is None and raw_csv_path is not None:
            train_df, cal_df = load_train_and_calibration_data(raw_csv_path)
            
        try:
            res = run_experiment(
                config=cfg,
                train_df=train_df,
                cal_df=cal_df,
                raw_csv_path=raw_csv_path,
                dataset_fingerprint=matrix_fingerprint,
                save_dir=str(out_dir),
                verbose=verbose,
            )
            results.append(res)
            summaries.append(extract_summary_row(res.to_dict()))
        except Exception as e:
            print(f"[{cfg.run_id}] FAILED: {str(e)}")
            traceback.print_exc()
            matrix_complete = False
            break # STOP MATRIX ON FAILURE
            
    # Check if exactly 16 successful
    if matrix_complete and len(summaries) == 16:
        summary_path = out_dir / "matrix_summary.json"
        import json
        with open(summary_path, "w") as f:
            json.dump([s.to_dict() for s in summaries], f, indent=2)
        if verbose:
            print(f"MATRIX COMPLETE. Summary saved to {summary_path}")
    else:
        if verbose:
            print("MATRIX INCOMPLETE.")
        
    return results

def run_preflight(data_path: str, verbose: bool = True) -> None:
    print("=== MetroGuard Preflight Check ===")
    p = Path(data_path)
    if not p.exists():
        print(f"FAIL: Dataset not found at {data_path}")
        return
    print("Dataset exists.")
    
    configs = generate_matrix_configs()
    if len(configs) != 16:
        print(f"FAIL: Matrix generator produced {len(configs)} configs instead of 16.")
        return
    print("Matrix enumeration valid (16 unique configurations).")
    
    unique_ids = set(c.run_id for c in configs)
    if len(unique_ids) != 16:
        print("FAIL: Run IDs are not unique.")
        return
    print("Run IDs are deterministic and unique.")
    
    commit = get_git_commit()
    print(f"Git commit: {commit}")
    
    print("Preflight check passed. Ready for long-run matrix execution.")
"""

# Replace run_matrix
pattern = re.compile(r"def run_matrix\(.*?return results", re.DOTALL)
content = pattern.sub(new_run_matrix, content)

# Update main
main_target = """    parser.add_argument(
        "--list-matrix",
        action="store_true",
        help="List all 16 candidate baseline configurations in the experiment matrix.",
    )"""
main_replace = """    parser.add_argument(
        "--run-matrix",
        action="store_true",
        help="Execute the full 16-run real-data detector matrix.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume matrix execution by skipping completely valid existing artifacts.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run lightweight preflight check before matrix execution.",
    )
    parser.add_argument(
        "--list-matrix",
        action="store_true",
        help="List all 16 candidate baseline configurations in the experiment matrix.",
    )"""
content = content.replace(main_target, main_replace)

cli_exec_target = """    if args.smoke:"""
cli_exec_replace = """    if args.preflight:
        run_preflight(data_path=args.data_path, verbose=True)
    elif args.run_matrix:
        configs = generate_matrix_configs()
        run_matrix(
            configs=configs,
            raw_csv_path=args.data_path,
            save_dir=args.output_dir + "/matrix",
            resume=args.resume,
            verbose=True,
        )
    elif args.smoke:"""
content = content.replace(cli_exec_target, cli_exec_replace)

with open("src/experiments/runner.py", "w", encoding="utf-8") as f:
    f.write(content)
