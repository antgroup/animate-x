import os
from moviepy.editor import VideoFileClip

def trim_videos_in_folder(input_folder, output_folder, duration=3):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件是否为 .MOV 文件
        if filename.lower().endswith('.mov'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"trimmed_{filename}")
            
            # 使用 moviepy 截取视频的前三秒
            try:
              with VideoFileClip(input_path) as video:
                  trimmed_video = video.subclip(0, duration)
                  trimmed_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
                  print(f"{filename} has been trimmed and saved to {output_path}")
            except:
              pass

# 指定输入输出文件夹
input_folder = ''
output_folder = ''

# 调用函数
trim_videos_in_folder(input_folder, output_folder)
