# -*- coding: utf-8 -*-
import sys, io
# Fix Windows console encoding (cp1252 -> utf-8)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
=============================================================================
  BUOC 2 - CAT DU LIEU & CHUAN HOA (Temporal Split & StandardScaler)
=============================================================================
  Mục tiêu:
    - Phân chia dữ liệu theo thời gian (Train: 2011-2020 | Test: 2021-2025)
    - Chuẩn hóa đặc trưng bằng StandardScaler (chống Data Leakage)
    - Lưu scaler đã học thành file scaler.pkl để dùng lại

  Input  : data/master_dataset.csv
  Output : models/saved_models/scaler.pkl
           Trả về (X_train_scaled, X_test_scaled, y_train, y_test)
=============================================================================
"""

import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------------------------------------------------------
# Hằng số đường dẫn
# ---------------------------------------------------------------------------
INPUT_PATH         = "data/master_dataset.csv"
OUTPUT_SCALER_DIR  = "models/saved_models"
OUTPUT_SCALER_PATH = os.path.join(OUTPUT_SCALER_DIR, "scaler.pkl")
OUTPUT_MODEL_PATH  = os.path.join(OUTPUT_SCALER_DIR, "random_forest.pkl")

TRAIN_START = 2011
TRAIN_END   = 2020
TEST_START  = 2021
TEST_END    = 2025

COLS_TO_DROP = ["Team", "Season", "Target"]

TARGET_LABELS = {
    0: "Bi loai (Eliminated)",
    1: "Playoff",
    2: "A quan (Runner-up)",
    3: "Vo dich (Champion)",
}

# Tham số Random Forest
RF_PARAMS = {
    "n_estimators"  : 300,   # Số cây trong rừng
    "max_depth"     : None,  # Cây phát triển tối đa (None = không giới hạn)
    "min_samples_split" : 5,
    "min_samples_leaf"  : 2,
    "class_weight"  : "balanced",  # Bù trừ mất cân bằng nhãn (ít Champion hơn Playoff)
    "random_state"  : 42,
    "n_jobs"        : -1,    # Dùng toàn bộ CPU
}


# ---------------------------------------------------------------------------
# Hàm tiện ích: in phân phối nhãn Target
# ---------------------------------------------------------------------------
def _print_target_distribution(y: pd.Series, set_name: str) -> None:
    counts = y.value_counts().sort_index()
    total  = len(y)
    print(f"\n  Phan phoi nhan Target trong {set_name}:")
    for label, count in counts.items():
        pct  = count / total * 100
        desc = TARGET_LABELS.get(label, f"Label {label}")
        print(f"    [{label}] {desc:<30} : {count:>4} mau  ({pct:5.1f}%)")
    print(f"    {'TONG':<35} : {total:>4} mau (100.0%)")


# ---------------------------------------------------------------------------
# Hàm chính
# ---------------------------------------------------------------------------
def prepare_data_and_scale():
    """
    Thực hiện Bước 2: Temporal Split + StandardScaler.

    Returns
    -------
    X_train_scaled : pd.DataFrame  — Tập huấn luyện đã chuẩn hóa
    X_test_scaled  : pd.DataFrame  — Tập kiểm tra đã chuẩn hóa
    y_train        : pd.Series     — Nhãn tập huấn luyện
    y_test         : pd.Series     — Nhãn tập kiểm tra
    """

    # =========================================================================
    # 0. Đọc dữ liệu đầu vào
    # =========================================================================
    print("=" * 65)
    print("  BUOC 2 - TEMPORAL SPLIT & CHUAN HOA DU LIEU")
    print("=" * 65)
    print(f"\n[0] Đọc dữ liệu từ: {INPUT_PATH}")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy file đầu vào: {INPUT_PATH}\n"
            "Hãy chắc chắn đã chạy Bước 1 (clean_dataset.py) trước."
        )

    df = pd.read_csv(INPUT_PATH)
    print(f"    → Tải thành công: {df.shape[0]} dòng × {df.shape[1]} cột")
    print(f"    → Khoảng Season có trong dữ liệu: "
          f"{int(df['Season'].min())} – {int(df['Season'].max())}")

    # =========================================================================
    # Task 2.1 — Temporal Split
    # =========================================================================
    print("\n" + "-" * 65)
    print(f"[Task 2.1] Phan chia Train / Test theo thoi gian (Season)")
    print("-" * 65)

    train_df = df[(df["Season"] >= TRAIN_START) & (df["Season"] <= TRAIN_END)].copy()
    test_df  = df[(df["Season"] >= TEST_START)  & (df["Season"] <= TEST_END)].copy()

    print(f"  Tap TRAIN (Season {TRAIN_START}-{TRAIN_END}): {len(train_df):>4} mau")
    print(f"  Tap TEST  (Season {TEST_START}-{TEST_END}):  {len(test_df):>4} mau")

    if len(train_df) == 0:
        raise ValueError("Tap Train rong! Kiem tra lai khoang Season trong du lieu.")
    if len(test_df) == 0:
        raise ValueError("Tap Test rong! Kiem tra lai khoang Season trong du lieu.")

    # Tách X và y
    y_train = train_df["Target"].reset_index(drop=True)
    y_test  = test_df["Target"].reset_index(drop=True)

    X_train = train_df.drop(columns=COLS_TO_DROP).reset_index(drop=True)
    X_test  = test_df.drop(columns=COLS_TO_DROP).reset_index(drop=True)

    print(f"\n  So dac trung (features): {X_train.shape[1]}")
    print(f"  Danh sach dac trung:")
    for i, col in enumerate(X_train.columns, 1):
        print(f"    {i:>2}. {col}")

    _print_target_distribution(y_train, f"Train ({TRAIN_START}-{TRAIN_END})")
    _print_target_distribution(y_test,  f"Test  ({TEST_START}-{TEST_END})")

    # =========================================================================
    # Task 2.2 — StandardScaler (chống Data Leakage)
    # =========================================================================
    print("\n" + "-" * 65)
    print("[Task 2.2] Chuan hoa du lieu (StandardScaler)")
    print("-" * 65)

    scaler = StandardScaler()

    # Chỉ fit trên Train — máy học cấu trúc của QUÁ KHỨ
    X_train_scaled_arr = scaler.fit_transform(X_train)
    print("  [OK] .fit_transform() ap dung len tap TRAIN")
    print("    (Scaler hoc mean & std tu du lieu qua khu 2011-2020)")

    # Chỉ transform trên Test — áp công thức quá khứ lên TƯƠNG LAI
    X_test_scaled_arr  = scaler.transform(X_test)
    print("  [OK] .transform()      ap dung len tap TEST")
    print("    (Dung lai dung mean & std da hoc, KHONG cho nhin tuong lai)")

    # Giữ lại tên cột để mô hình dễ đọc sau này
    X_train_scaled = pd.DataFrame(X_train_scaled_arr, columns=X_train.columns)
    X_test_scaled  = pd.DataFrame(X_test_scaled_arr,  columns=X_test.columns)

    # Kiem chung nhanh: mean sau scaling phai ~0, std ~1 tren Train
    mean_check = X_train_scaled.mean().abs().max()
    std_check  = X_train_scaled.std().max()
    print(f"\n  Kiem tra sau chuan hoa (tap Train):")
    print(f"    |mean| lon nhat : {mean_check:.2e}  (ky vong ~ 0)")
    print(f"    std  lon nhat   : {std_check:.4f}  (ky vong ~ 1)")

    # =========================================================================
    # Task 2.3 — Lưu scaler.pkl
    # =========================================================================
    print("\n" + "-" * 65)
    print("[Task 2.3] Luu file cau hinh Scaler")
    print("-" * 65)

    os.makedirs(OUTPUT_SCALER_DIR, exist_ok=True)
    joblib.dump(scaler, OUTPUT_SCALER_PATH)

    file_size_kb = os.path.getsize(OUTPUT_SCALER_PATH) / 1024
    print(f"  [OK] Scaler luu thanh cong tai: {OUTPUT_SCALER_PATH}")
    print(f"    Kich thuoc file: {file_size_kb:.1f} KB")
    print( "    (Load lai: scaler = joblib.load('models/saved_models/scaler.pkl'))")

    # =========================================================================
    # Task 2.4 — Huấn luyện & lưu Random Forest
    # =========================================================================
    print("\n" + "-" * 65)
    print("[Task 2.4] Huan luyen mo hinh Random Forest")
    print("-" * 65)
    print(f"  Tham so mo hinh:")
    for k, v in RF_PARAMS.items():
        print(f"    {k:<22} : {v}")

    rf_model = RandomForestClassifier(**RF_PARAMS)
    rf_model.fit(X_train_scaled, y_train)
    print("\n  [OK] Huan luyen hoan thanh tren tap Train (2011-2020)")

    # --- Đánh giá trên tập Test ---
    y_pred = rf_model.predict(X_test_scaled)
    acc    = accuracy_score(y_test, y_pred)

    target_names = [TARGET_LABELS[i] for i in sorted(TARGET_LABELS)]
    report = classification_report(
        y_test, y_pred,
        target_names=target_names,
        zero_division=0
    )

    print(f"\n  Danh gia tren tap Test (2021-2025):")
    print(f"    Accuracy tong the : {acc * 100:.2f}%")
    print("\n  Classification Report:")
    for line in report.splitlines():
        print("    " + line)

    # --- Feature Importance Top-10 ---
    importances = pd.Series(
        rf_model.feature_importances_,
        index=X_train_scaled.columns
    ).sort_values(ascending=False)

    print("\n  Top-10 dac trung quan trong nhat (Feature Importance):")
    for rank, (feat, imp) in enumerate(importances.head(10).items(), 1):
        bar = "█" * int(imp * 200)
        print(f"    {rank:>2}. {feat:<35} {imp:.4f}  {bar}")

    # --- Lưu model ---
    joblib.dump(rf_model, OUTPUT_MODEL_PATH)
    model_size_kb = os.path.getsize(OUTPUT_MODEL_PATH) / 1024
    print(f"\n  [OK] Random Forest luu thanh cong tai: {OUTPUT_MODEL_PATH}")
    print(f"    Kich thuoc file: {model_size_kb:.1f} KB")
    print( "    (Load lai: model = joblib.load('models/saved_models/random_forest.pkl'))")

    # =========================================================================
    # Tổng kết
    # =========================================================================
    print("\n" + "=" * 65)
    print("  [DONE] BUOC 2 HOAN THANH - Du lieu san sang cho Buoc 3!")
    print("=" * 65)
    print(f"\n  Tom tat ket qua:")
    print(f"    X_train_scaled : {X_train_scaled.shape}  (mau x dac trung)")
    print(f"    X_test_scaled  : {X_test_scaled.shape}")
    print(f"    y_train        : {y_train.shape}")
    print(f"    y_test         : {y_test.shape}")
    print(f"    Accuracy (Test): {acc * 100:.2f}%")
    print(f"    Files da luu:")
    print(f"      - {OUTPUT_SCALER_PATH}")
    print(f"      - {OUTPUT_MODEL_PATH}")
    print()

    return X_train_scaled, X_test_scaled, y_train, y_test, rf_model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, model = prepare_data_and_scale()