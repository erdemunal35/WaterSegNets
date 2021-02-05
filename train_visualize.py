import os
import matplotlib.pyplot as plt
import numpy as np

train_dice_loss = []
train_iou_score = []
valid_dice_loss = []
valid_iou_score = []
f = open("log.txt")
for line in f:
	pos1 = line.find('.')
	pos2 = line.find(',')
	num = "0."+line[pos1+1:pos2]
	train_dice_loss.append(float(num))

	line = line[pos2:]
	pos1 = line.find('.')
	pos2 = line.find('}')
	num = "0."+line[pos1+1:pos2]
	train_iou_score.append(float(num))
	
	line = line[pos2:]
	pos1 = line.find('.')
	pos2 = line.find(',')
	num = "0."+line[pos1+1:pos2]
	valid_dice_loss.append(float(num))

	line = line[pos2:]
	pos1 = line.find('.')
	pos2 = line.find('}')
	num = "0."+line[pos1+1:pos2]
	valid_iou_score.append(float(num))

f.close()

x = np.arange(50)

plt.plot(x, train_iou_score, label = "Train IoU Score")
plt.plot(x, valid_iou_score, label = "Validation IoU Score")
plt.ylabel('IoU Score')
plt.xlabel('Epoch')
plt.legend()
plt.show()
