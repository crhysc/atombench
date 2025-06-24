import os
import pandas as pd
from pathlib import Path

df = pd.DataFrame()

for directory in os.listdir('.'):
	if os.path.isdir(directory):
		cwd = os.getcwd()
		os.chdir(os.path.join(cwd,directory))
		cwd2 = os.getcwd()
		print(cwd2)
		if os.path.exists("./distribution.pdf"):
			os.rename("distribution.pdf", f"distribution_{Path(cwd2).parts[-1]}.pdf")
		else:
			pass
		mini_df = pd.read_csv("metrics.txt", header=None, delim_whitespace=True)
		df = pd.concat([df,mini_df], axis=1, ignore_index=True)
		#df.to_csv(f'metrics_{Path(cwd2).parts[-1]}.csv', index=False)
		os.chdir('..')

df.to_csv("epic_metrics.csv", index=False)
