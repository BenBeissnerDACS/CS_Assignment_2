import pandas as pd
import matplotlib.pyplot as plt

# Read the enumData.csv file
enum_data_df = pd.read_csv('enumData.csv')

# Create a hexbin plot using matplotlib
plt.figure(figsize=(10, 6))
plt.hexbin(enum_data_df['cor'], enum_data_df['weight'], gridsize=50, cmap='viridis')

plt.xlabel('Correlation Coefficient')
plt.ylabel('Weight')
plt.title('Hexbin plot of Correlation Coefficient vs. Weight')

cb = plt.colorbar()
cb.set_label('Counts in Bin')

# Save the plot as an image
plt.savefig('hexbin_correlation_vs_weight.png')

# Show the plot
plt.show()
