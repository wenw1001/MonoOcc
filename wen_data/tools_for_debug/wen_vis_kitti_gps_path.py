import os
import glob
import folium

def load_gps_path(oxts_dir):
    """
    讀取 OXTS 資料夾中的經緯度資訊
    """
    # 搜尋所有 .txt 檔案並依照檔名排序
    files = sorted(glob.glob(os.path.join(oxts_dir, 'data', '*.txt')))
    
    if not files:
        print(f"錯誤: 在 {oxts_dir} 找不到任何 .txt 檔案")
        return []

    path_points = []
    print(f"正在讀取 {len(files)} 筆 GPS 資料...")

    for f in files:
        with open(f, 'r') as file:
            line = file.readline()
            if not line:
                continue
            values = [float(x) for x in line.strip().split()]
            path_points.append([values[0], values[1]])

    return path_points

def create_satellite_map(path_points, output_file):
    """
    使用 folium 繪製衛星地圖 (Esri WorldImagery)
    """
    if not path_points:
        print("沒有數據可以繪圖")
        return

    avg_lat = sum(p[0] for p in path_points) / len(path_points)
    avg_lon = sum(p[1] for p in path_points) / len(path_points)

    # 使用 Esri WorldImagery 衛星圖
    m = folium.Map(
        location=[avg_lat, avg_lon], 
        zoom_start=18, 
        max_zoom=21,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri'
    )

    # 畫軌跡 (黃色)
    folium.PolyLine(
        locations=path_points,
        color="yellow",
        weight=3,
        opacity=0.8,
        tooltip="KITTI Path"
    ).add_to(m)

    # 起點
    folium.Marker(
        location=path_points[0],
        popup="Start",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)

    # 終點
    folium.Marker(
        location=path_points[-1],
        popup="End",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(m)

    m.save(output_file)
    print(f"衛星地圖已建立！請開啟: {output_file}")

# ==========================================
# 路徑設定 (已修改)
# ==========================================
seq = "02"
raw_data_file = "2011_10_03_drive_0034_sync"
base_path = f"/mnt/data/KITTI Raw Data/{seq}/{raw_data_file}/oxts"
# 這裡將檔名設為 kitti_satellite_map.html
save_path = f"/mnt/data/KITTI Raw Data/{seq}/kitti_satellite_map.html"

if __name__ == "__main__":
    if os.path.exists(base_path):
        # 讀取資料
        points = load_gps_path(base_path)
        
        # 畫圖並存檔
        if points:
            create_satellite_map(points, save_path)
    else:
        print(f"錯誤: 找不到路徑 {base_path}")