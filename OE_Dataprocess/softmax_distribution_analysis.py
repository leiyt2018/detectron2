import matplotlib.pyplot as plt

# # 这两行代码解决 plt 中文显示的问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

waters = ('1', '2', '3', 'bg')
buy_number = [0.1, 0.2, 0.1, 0.6]

plt.bar(waters, buy_number, 0.02)
plt.title('softmax')

plt.show()
