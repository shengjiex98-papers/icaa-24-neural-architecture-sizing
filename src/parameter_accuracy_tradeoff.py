import pandas as pd
import plotly.express as px

# def main():
data = [
    ["Model", "Parameter", "Accuracy"],
    ["B0", 54.051, 35.31],
    ["B1", 56.830, 38.90],
    ["B2", 69.371, 30.61],
    ["B3", 84.574, 27.23],
    ["B4", 118.589, 28.13],
    ["B5", 156.735, 36.25],
    ["B6", 205.171, 21.08],
    ["B7", 269.646, 21.11],
    ["MNv3s", 9.869, 47.66],
    ["MNv3l", 43.731, 32.91],
    ["MN", 43.302, 41.05],
    ["MNv2", 43.786, 44.31],
    ["SNV2", 37.916, 40.19],
    ["X", 104.002, 26.92]
]
df = pd.DataFrame(data[1:], columns=data[0])
df = df.sort_values(by="Accuracy", ascending=False)

fig = px.scatter(df, x="Parameter", y="Accuracy", hover_name="Model", title="Parameter vs Accuracy Tradeoff")
fig.show()

# if __name__ == "__main__":
#     main()