import pandas as pd

csv1 = pd.read_csv("results/test.csv")
csv2 = pd.read_csv("../110550093.csv")

cnt = 0
for x, i, d in zip(csv1["id"], csv1["label"], csv2["label"]):
    if i != d:
        cnt += 1
        print(x, i, d)

print(cnt)
