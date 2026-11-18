# vision_mmbench

样本：`vision-mmbench-0011`

## 完整指令

```text
Which Python code can generate the content of the image?
A. import re
result = re.match(r'(\w+)=(\d+)', 'set width=20 and height=10')
print(result)
B. import re
result = re.findall(r'(\w+)=(\d+)', 'set width=20 and height=10')
print(result)
C. import re
result = re.findall(r'(\w+)=(\d+)', 'set width=30 and height=10')
print(result)
D. import re
result = re.findall(r'(\w+)=(\d+)', 'set width=20 and height=15')
print(result)
Answer with only the option letter.
```

## 输入图像

`input_image.jpg`

## MiniQwen-Omni-V1 输出

```text
B
```
