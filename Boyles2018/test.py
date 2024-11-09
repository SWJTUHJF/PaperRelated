import matplotlib.pyplot as plt
import numpy as np


# 准备数据
x = [1,2,3,4,5]
y = [2, 3, 5, 7, 11]

# 创建图形和轴
fig, ax = plt.subplots()

# 绘制折线图
ax.plot(x, y, linestyle='-', color='g', label='0-SOR')

# 添加标签和标题
ax.set_xlabel('Time(in seconds)')
ax.set_ylabel('Relative gap')
ax.legend()
ax.grid(True)
plt.savefig('sine_wave.png', dpi=300, bbox_inches='tight')
# 显示图形
plt.show()