from sympy import symbols, diff,Abs

# 定义符号
ang_dist, dist, kmax, s = symbols('ang_dist dist kmax s')

# 定义cost的表达式
cost = Abs(ang_dist)



# 求cost_squared对ang_dist的偏导数
partial_derivative = diff(cost, ang_dist)

# 打印结果
print(partial_derivative)