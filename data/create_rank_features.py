import pandas as pd
import os

# 1. Đọc file dữ liệu mới nhất bạn vừa cập nhật
input_path = 'data/final_complete_dataset.csv'
output_path = 'data/model_ready_dataset.csv'

print("🔄 Đang tải dữ liệu để tạo Contextual Features (Thứ hạng)...")
df = pd.read_csv(input_path)

# ==========================================================
# 2. PHÂN LOẠI CHỈ SỐ ĐỂ XẾP HẠNG
# ==========================================================
# Nhóm 1: Chỉ số CÀNG CAO CÀNG TỐT (Số to nhất = Hạng 1)
higher_is_better_cols = [
    'W', 'MOV', 'SRS', 'ORtg', 'NRtg', 'TS%', 
    'Post_ORtg', 'Post_NRtg', 'Post_TS_PCT',
    'Team_3PM_per100', 'Team_AST_per100', 'Opp_TOV_forced',
    'Clutch_Win_PCT', 'Post_AllStar_Win_PCT'
]

# Nhóm 2: Chỉ số CÀNG THẤP CÀNG TỐT (Số nhỏ nhất = Hạng 1)
lower_is_better_cols = [
    'DRtg', 'Post_DRtg', 'Team_TOV_per100', 'Opp_FG_PCT', 'Opp_FG3_PCT'
]

# ==========================================================
# 3. TẠO CỘT XẾP HẠNG THEO TỪNG MÙA GIẢI (GROUPBY SEASON_YEAR)
# ==========================================================
print("📈 Đang tiến hành xếp hạng tự động các chỉ số theo từng mùa giải...")

# Xếp hạng nhóm chỉ số càng cao càng tốt
for col in higher_is_better_cols:
    if col in df.columns:
        df[f'{col}_Rank'] = df.groupby('Season_Year')[col].rank(ascending=False, method='min')

# Xếp hạng nhóm chỉ số càng thấp càng tốt
for col in lower_is_better_cols:
    if col in df.columns:
        df[f'{col}_Rank'] = df.groupby('Season_Year')[col].rank(ascending=True, method='min')

# Riêng cột Age_Diff (Độ lệch tuổi hoàng kim): 
# Vì file của bạn đã có sẵn cột 'Age_Rank' nhưng ta muốn đảm bảo tính toán đồng bộ theo group năm
if 'Age_Diff' in df.columns:
    df['Age_Rank'] = df.groupby('Season_Year')['Age_Diff'].rank(ascending=True, method='min')

# ==========================================================
# 4. XUẤT FILE DATASET HOÀN CHỈNH CHO MODEL
# ==========================================================
df.to_csv(output_path, index=False)
print(f"✅ THÀNH CÔNG! Đã tạo xong tất cả các cột Rank dựa trên file mới của bạn.")
print(f"📂 Dữ liệu sẵn sàng train model đã lưu tại: {output_path}")