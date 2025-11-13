import os
import subprocess

def external_cmd(cmd, code="utf-8"):
  process = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  while process.poll() is None:
    line = process.stdout.readline()
    line = line.strip()
    if line:
      print(line.decode(code, 'ignore'))
      #输出日志
    # with open(r"D:\moonlocate\output\log.txt",'a+') as f:
    #     f.write(line.decode(code, 'ignore'))


if __name__=='__main__':
    workplace=r'D:\moonlocate'
    image_path =  os.path.join(workplace,r"output\test\images2")
    output_path =  os.path.join(workplace,r"output\test\colmap")
    colmap_command = f'colmap feature_extractor --database_path {output_path}/database.db --image_path {image_path}'
    external_cmd(colmap_command, code="utf8")