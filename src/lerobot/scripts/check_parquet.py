import pandas as pd

# 1. 读取 parquet 文件
input_path = "file-000.parquet"
output_path = "file-000.txt"

print(f"🔍 正在读取 {input_path} ...")
df = pd.read_parquet(input_path)

# 2. 将所有内容保存为文本
with open(output_path, "w", encoding="utf-8") as f:
    f.write(df.to_string())

print(f"✅ 已将所有内容保存到 {output_path}")
print(f"📊 共 {len(df)} 行, {len(df.columns)} 列")

