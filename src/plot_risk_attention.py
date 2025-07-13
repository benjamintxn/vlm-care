import pandas as pd
import matplotlib.pyplot as plt

# 1) Load the CSV log
csv_path = 'output/risk_logs/000463.csv'
df = pd.read_csv(csv_path)

# 1a) See what the columns actually are
print("Columns in CSV:", df.columns.tolist())
print(df.head())

# 2) Identify which columns to use:
#    Assume the first column is the frame index,
#    the second is the risk score,
#    and any column ending in "_score" beyond that is an attention score.
frame_col = df.columns[0]
risk_col  = df.columns[1]
att_cols  = [c for c in df.columns if c.endswith('_score') and c not in (risk_col, frame_col)]
print(f"Using frame_col={frame_col}, risk_col={risk_col}, att_cols={att_cols}")

# 3) Plot Risk Score over frames
plt.figure(figsize=(8,4))
plt.plot(df[frame_col], df[risk_col], label='Risk')
plt.xlabel('Frame')
plt.ylabel('Risk Score')
plt.title('Risk Score over Frames')
plt.grid(True)
plt.tight_layout()
plt.show()

# 4) Plot Top Attention Scores
plt.figure(figsize=(8,4))
for col in att_cols:
    plt.plot(df[frame_col], df[col], label=col)
plt.xlabel('Frame')
plt.ylabel('Attention Score')
plt.title('Attention Scores over Frames')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()