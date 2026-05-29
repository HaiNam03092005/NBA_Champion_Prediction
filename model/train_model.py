import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # ==========================================
    # PHẦN 1: THIẾT LẬP ĐƯỜNG DẪN
    # ==========================================
    input_file = 'basketball_prediction/data/master_dataset.csv'
    model_dir = 'basketball_prediction/model/saved_models'
    scaler_file = os.path.join(model_dir, 'scaler.pkl')
    rf_model_file = os.path.join(model_dir, 'random_forest_model.pkl')

    # Đảm bảo thư mục lưu trữ đã tồn tại
    os.makedirs(model_dir, exist_ok=True)

    print("1. Đang đọc dữ liệu...")
    df = pd.read_csv(input_file)

    # ==========================================
    # PHẦN 2: CẮT DỮ LIỆU & CHUẨN HÓA (BƯỚC 2)
    # ==========================================
    print("2. Đang phân tách dữ liệu theo thời gian (Temporal Split)...")
    # Tập Train: Từ 2011 đến 2020 | Tập Test: Từ 2021 đến 2025
    train_df = df[(df['Season'] >= 2011) & (df['Season'] <= 2020)]
    test_df = df[(df['Season'] >= 2021) & (df['Season'] <= 2025)]

    # Loại bỏ các cột không dùng để huấn luyện
    columns_to_drop = ['Team', 'Season', 'Target']
    
    X_train = train_df.drop(columns=columns_to_drop)
    y_train = train_df['Target']
    
    X_test = test_df.drop(columns=columns_to_drop)
    y_test = test_df['Target']

    print("3. Đang chuẩn hóa dữ liệu (StandardScaler)...")
    scaler = StandardScaler()
    
    # CHÚ Ý: Chỉ fit trên Train, transform trên cả Train và Test để tránh Data Leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Lưu Scaler để dùng cho ứng dụng thực tế sau này
    joblib.dump(scaler, scaler_file)
    print(f"   -> Đã lưu công cụ chuẩn hóa tại: {scaler_file}")

    # ==========================================
    # PHẦN 3: HUẤN LUYỆN MÔ HÌNH RANDOM FOREST
    # ==========================================
    print("4. Đang huấn luyện mô hình Random Forest...")
    # Khởi tạo mô hình. Thêm class_weight='balanced_subsample' để ép AI chú ý đến nhóm thiểu số (Vô địch/Á quân)
    rf_model = RandomForestClassifier(
        n_estimators=1000, 
        random_state=42, 
        class_weight='balanced_subsample'
    )
    
    # Cho máy học từ dữ liệu quá khứ
    rf_model.fit(X_train_scaled, y_train)

    # Lưu mô hình đã huấn luyện
    joblib.dump(rf_model, rf_model_file)
    print(f"   -> Đã lưu mô hình AI tại: {rf_model_file}")

    # ==========================================
    # PHẦN 4: KIỂM TRA ĐỘ CHÍNH XÁC
    # ==========================================
    print("\n5. ĐÁNH GIÁ MÔ HÌNH TRÊN DỮ LIỆU TƯƠNG LAI (Tập Test 2021-2025):")
    # Dự đoán kết quả trên tập Test
    y_pred = rf_model.predict(X_test_scaled)
    
    # In ra báo cáo chi tiết
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   -> Độ chính xác tổng thể (Accuracy): {accuracy * 100:.2f}%\n")
    print("   -> Báo cáo phân loại chi tiết:")
    print(classification_report(y_test, y_pred, target_names=['Bị loại (0)', 'Playoff (1)', 'Á quân (2)', 'Vô địch (3)'], zero_division=0))

if __name__ == "__main__":
    main()