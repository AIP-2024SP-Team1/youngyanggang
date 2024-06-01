import matplotlib.pyplot as plt
import json

#file_path = '/hdd/shbin05/output/log.txt'
file_path = './output/log.txt'

with open(file_path, 'r') as file:
    contents = file.readlines()

train_closs = []
val_closs = []
epochs = []

for line in contents:
    data = json.loads(line)

    train_closs.append(data['train_closs'])
    val_closs.append(data['val'])
    epochs.append(data['epoch'])

print(data)

plt.figure(figsize=(10, 6))  
plt.plot(epochs, train_closs, label='Training Loss', color='blue')
plt.plot(epochs, val_closs, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss') 
plt.legend()  
plt.grid(True) 

output_file_path = '/home/shbin05/llama-adapter/multimodal7b_v2/output/loss_graph.png'
plt.savefig(output_file_path)
