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

dist_yolo_backbones = [
    ["Model", "Parameter", "Error"],
    ["MobileNetv3-small", 9.833, 42.53],
    ["MobileNetv2", 43.714, 34.23],
    ["EfficientNet-B2", 69.371, 30.61],
    ["EfficientNet-B3", 84.574, 27.23],
    ["Xception", 103.935, 24.34],
    ["EfficientNet-B6", 205.171, 21.08]
]

df = pd.DataFrame(dist_yolo_backbones[1:], columns=dist_yolo_backbones[0])

plt.rc('font', size=14)
plt.plot(df["Parameter"], df["Error"], marker="o", markersize=8, lw=5)
# plt.title("Parameter vs Error Tradeoff")
plt.xlabel("Billion FLOPs")
plt.ylabel("Absolute Relative Error")
plt.annotate(df.iloc[0]["Model"], (df.iloc[0]["Parameter"], df.iloc[0]["Error"]), textcoords="offset points", xytext=(8,-4), ha='left')
for i, row in df.iloc[1:-1].iterrows():
    plt.annotate(row["Model"], (row["Parameter"], row["Error"]), textcoords="offset points", xytext=(4,2), ha='left')
plt.annotate(df.iloc[-1]["Model"], (df.iloc[-1]["Parameter"], df.iloc[-1]["Error"]), textcoords="offset points", xytext=(-24,-8), ha='right')
# plt.show()
plt.savefig("dist_yolo.pdf")
