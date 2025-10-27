
import sqlite3
import os
import re


conn = sqlite3.connect('test.db')
cur = conn.cursor()


cur.execute("""
CREATE TABLE IF NOT EXISTS test_video (
    id INTEGER PRIMARY KEY,
    video_path TEXT UNIQUE 
)
""")

# 获取文件夹路径
folder_path = r"D:\yolov13\main\video_msu"

# 获取文件夹下所有文件
files = os.listdir(folder_path)

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

files.sort(key=natural_sort_key)

for file_name in files:
    if file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        video_path = os.path.join(folder_path, file_name)
        
        cur.execute("SELECT COUNT(*) FROM test_video WHERE video_path = ?", (video_path,))
        count = cur.fetchone()[0]
        
        if count == 0:
            cur.execute("INSERT INTO test_video (video_path) VALUES (?)", (video_path,))
        else:
            print(f"Skipping duplicate entry: {video_path}")

conn.commit()


cur.execute("SELECT * FROM xxx")
rows = cur.fetchall()
for row in rows:
    print(row)

conn.close()