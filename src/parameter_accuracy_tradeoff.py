import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# def main():
# dist_yolo_backbones = [
#     ["Model", "Parameter", "Error"],
#     # ["B0", 54.051, 35.31],
#     # ["B1", 56.830, 38.90],
#     ["EfficientNet-B2", 69.371, 30.61],
#     ["EfficientNet-B3", 84.574, 27.23],
#     # ["B4", 118.589, 28.13],
#     # ["B5", 156.735, 36.25],
#     ["EfficientNet-B6", 205.171, 21.08],
#     # ["B7", 269.646, 21.11],
#     ["MobileNetv3-small", 9.869, 47.66],
#     ["MobileNetv3-large", 43.731, 32.91],
#     # ["MN", 43.302, 41.05],
#     # ["MNv2", 43.786, 44.31],
#     ["ShuffleNetv2", 37.916, 40.19],
#     ["Xecption", 104.002, 26.92]
# ]
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
df = df.sort_values(by="Error", ascending=False)

efficient_net_configurations = [
    ["Model", "Parameter", "Error"],
    ["B0", 0.39, 77.1],
    ["B1", 0.70, 79.1],
    ["B2", 1.0, 80.1],
    ["B3", 1.8, 81.6],
    ["B4", 4.2, 82.9],
    ["B5", 9.9, 83.6],
    ["B6", 19, 84.0],
    ["B7", 37, 84.3]
]
df_e = pd.DataFrame(efficient_net_configurations[1:], columns=efficient_net_configurations[0])
df_e["Error"] = 100 / df_e["Error"] - 1

# fig = px.line(df_e, x="Parameter", y="Error", 
#     labels={
#         "Parameter": "Billion FLOPs",
#         "Error": "Absolute Relative Error"
#     },
#     hover_name="Model", title="Parameter vs Error Tradeoff", text="Model")
# fig.show()

df_show = df
plt.plot(df_show["Parameter"], df_show["Error"], marker="o", markersize=3)
plt.title("Parameter vs Error Tradeoff")
plt.xlabel("Billion FLOPs")
plt.ylabel("Absolute Relative Error")
for i, row in df_show.iterrows():
    plt.annotate(row["Model"], (row["Parameter"], row["Error"]), textcoords="offset points", xytext=(2,2), ha='left')
# plt.show()
plt.savefig("tradeoff.pdf")

# if __name__ == "__main__":
#     main()