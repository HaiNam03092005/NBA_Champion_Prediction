import os
import glob
import pandas as pd

# ==========================================
# 1. ĐỊNH NGHĨA ĐƯỜNG DẪN CÁC FILE
# ==========================================
PATH_MASTER_INPUT = 'data/master_dataset.csv'          
DIR_FEATURES_INPUT = 'data/nba_features_combined'     
PATH_FINAL_OUTPUT  = 'data/final_complete_dataset.csv' 

print("============ TIẾN TRÌNH GỘP DỮ LIỆU TẬP TRUNG ============")

# ==========================================
# 2. ĐỌC VÀ GOM TẤT CẢ CÁC TÍNH NĂNG NÂNG CAO
# ==========================================
print("🔄 Bước 1: Đang thu thập dữ liệu từ các file nâng cao lẻ...")
feature_files = glob.glob(os.path.join(DIR_FEATURES_INPUT, "nba_features_*.csv"))

if not feature_files:
    print(f"❌ LỖI: Không tìm thấy file tính năng nào trong '{DIR_FEATURES_INPUT}'!")
    exit()

df_features_list = [pd.read_csv(f) for f in feature_files]
df_all_features = pd.concat(df_features_list, ignore_index=True)
print(f"   👉 Đã gom {len(feature_files)} file. Tìm thấy tổng cộng {df_all_features.shape[0]} dòng tính năng nâng cao.")

# ==========================================
# 3. ĐỌC FILE MASTER DATASET HIỆN TẠI
# ==========================================
print("\n📄 Bước 2: Đang tải tập dữ liệu master_dataset.csv...")
if not os.path.exists(PATH_MASTER_INPUT):
    print(f"❌ LỖI: Không tìm thấy file gốc '{PATH_MASTER_INPUT}'!")
    exit()

df_master_orig = pd.read_csv(PATH_MASTER_INPUT)
print(f"   👉 Đã tải file master gốc. Kích thước hiện tại: {df_master_orig.shape[0]} dòng, {df_master_orig.shape[1]} cột.")

# ==========================================
# 4. CHUẨN HÓA VÀ GỘP DỮ LIỆU
# ==========================================
print("\n🤝 Bước 3: Chuẩn hóa tên đội và tiến hành gộp dữ liệu...")

if 'Team' in df_master_orig.columns:
    df_master_orig.rename(columns={'Team': 'TEAM_NAME'}, inplace=True)
if 'Year' in df_master_orig.columns:
    df_master_orig.rename(columns={'Year': 'Season_Year'}, inplace=True)
if 'Season' in df_master_orig.columns:
    df_master_orig.rename(columns={'Season': 'Season_Year'}, inplace=True)

# 🔥 ĐÂY LÀ BƯỚC FIX LỖI: Bộ từ điển map tên đội bóng cho khớp với NBA API
team_name_mapping = {
    'Los Angeles Clippers': 'LA Clippers',
    # Bạn có thể thêm các đội khác vào đây nếu tương lai phát hiện thêm lỗi
    # Ví dụ: 'New Jersey Nets': 'Brooklyn Nets'
}
df_master_orig['TEAM_NAME'] = df_master_orig['TEAM_NAME'].replace(team_name_mapping)

df_final_complete = pd.merge(df_master_orig, df_all_features, on=['TEAM_NAME', 'Season_Year'], how='left')

if 'TEAM_ID' in df_final_complete.columns:
    df_final_complete.drop(columns=['TEAM_ID'], inplace=True)

# ==========================================
# 5. KIỂM TRA CHẤT LƯỢNG & XUẤT FILE 
# ==========================================
print("\n🔍 Bước 4: Kiểm tra tính toàn vẹn của dữ liệu...")
missing_data = df_final_complete[df_final_complete['Post_NRtg'].isna()]
nan_count = missing_data.shape[0]

if nan_count > 0:
    print(f"   ⚠️ Lưu ý: Vẫn còn {nan_count} dòng không khớp được chỉ số nâng cao!")
    print("   👇 Dưới đây là danh sách các đội bị lệch tên, hãy thêm chúng vào bộ 'team_name_mapping' ở Bước 3:")
    
    # In đích danh những đội bị lỗi để bạn dễ sửa
    for index, row in missing_data.iterrows():
        print(f"      - Đội: {row['TEAM_NAME']} | Năm: {row['Season_Year']}")
else:
    print("   ✅ TUYỆT VỜI! Khớp hoàn hảo 100%! Không có dòng nào bị khuyết chỉ số.")

print(f"\n💾 Bước 5: Đang ghi dữ liệu ra FILE THỨ 3 hoàn chỉnh...")
df_final_complete.to_csv(PATH_FINAL_OUTPUT, index=False)

print("\n" + "="*58)
print(f"🎯 HOÀN THÀNH TIẾN TRÌNH!")
print(f"📁 File thứ 3 đã được cập nhật tại: {PATH_FINAL_OUTPUT}")
print("="*58)