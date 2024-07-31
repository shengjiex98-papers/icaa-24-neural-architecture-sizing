import pandas as pd
import matplotlib.pyplot as plt

efficient_net_configurations = [
    ["Model", "Parameter", "Accuracy"],
    ["EfficientNet_B0", 0.39, 77.1],
    ["EfficientNet_B1", 0.70, 79.1],
    ["EfficientNet_B2", 1.0, 80.1],
    ["EfficientNet_B3", 1.8, 81.6],
    ["EfficientNet_B4", 4.2, 82.9],
    ["EfficientNet_B5", 9.9, 83.6],
    ["EfficientNet_B6", 19, 84.0],
    ["EfficientNet_B7", 37, 84.3]
]

df = pd.DataFrame(efficient_net_configurations[1:], columns=efficient_net_configurations[0])

plt.rc('font', size=14)
plt.plot(df["Parameter"], df["Accuracy"], marker="o", markersize=8, lw=5)
# plt.title("Parameter vs Error Tradeoff")
plt.xlabel("Billion FLOPs")
plt.ylabel("ImageNet Top1 Accuracy")
plt.annotate(df.iloc[0]["Model"], (df.iloc[0]["Parameter"], df.iloc[0]["Accuracy"]), textcoords="offset points", xytext=(4,6), ha='left')
for i, row in df.iloc[1:-2].iterrows():
    plt.annotate(row["Model"], (row["Parameter"], row["Accuracy"]), textcoords="offset points", xytext=(4,-14), ha='left')
plt.annotate(df.iloc[-2]["Model"], (df.iloc[-2]["Parameter"], df.iloc[-2]["Accuracy"]), textcoords="offset points", xytext=(4,6), ha='center')
plt.annotate(df.iloc[-1]["Model"], (df.iloc[-1]["Parameter"], df.iloc[-1]["Accuracy"]), textcoords="offset points", xytext=(8,-18), ha='right')
# plt.show()
plt.savefig("fig2.pdf")
