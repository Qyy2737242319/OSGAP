

import os


path = './datasetquality/15'
list1=["074","104","218_last","253","264","302","304","306","460","140","165","175","210","236"]
png_files = [os.path.splitext(file)[0] for file in os.listdir(path) if file.endswith('.png')]

png_files=png_files

png_count = len(png_files)

print(f"路径 {path} 下共有 {png_count} 个 PNG 文件。")
unique_list = list(set(png_files))

print(f"路径 {path} 下共有 {len(unique_list)} 个 PNG 文件。")


file_path = './arguments/list.txt'

with open(file_path, 'w') as file:
    for item in unique_list:
        file.write(f"{item}\n")

print(f"列表已保存到 {file_path}")