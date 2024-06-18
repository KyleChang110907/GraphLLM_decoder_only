import matplotlib.pyplot as plt

# Path to the text file
file_path = 'D:\kyle_MD_project\LLM_acc\llm_results\\Nonlinear_Dynamic_Analysis_WorldGM_TaiwanSection_BSE-1\\2024_04_09__08_47_29\\record.log'

# Lists to store the loss values
T_Loss = []
V_Loss = []
t_Loss = []

# Read the file line by line
with open(file_path, 'r') as file:
    for line in file:
        line = line.split(',')
        # Extract the loss values after the specified text
        for i in range(len(line)):
            if 'T_Loss' in line[i]:
                T_Loss.append(float(line[i].split(':')[1].strip()))
            if 'V_Loss' in line[i]:
                V_Loss.append(float(line[i].split(':')[1].strip()))
            if 't_Loss' in line[i]:
                t_Loss.append(float(line[i].split(':')[1].strip()))
# Plot the loss curve
# print(T_Loss)
plt.figure(figsize=(8, 6))
plt.plot(T_Loss, label='Training Loss')
plt.plot(V_Loss, label='Validation Loss')
plt.plot(t_Loss, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.ylim([0, 0.02])
plt.xlim([0, 500])
plt.legend()
plt.show()
plt.savefig('loss_curve.jpg')