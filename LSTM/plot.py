import pickle
import matplotlib.pyplot as plt
import matplotlib

train_loss = pickle.load(open('model_state/emodb/hidden/opensmile/train_loss.pkl', 'rb'))
valid_loss = pickle.load(open('model_state/emodb/hidden/opensmile/valid_loss.pkl', 'rb'))




# PG learning curve
train_epoch_id = [i+1 for i in range(len(train_loss))]
valid_epoch_id = [i+1 for i in range(len(valid_loss))]



plt.plot(train_epoch_id, train_loss, label='Training Loss')
plt.plot(valid_epoch_id, valid_loss, label='Validation Loss')
plt.title('Loss v.s. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./model_state/emodb/loss.png')
plt.show()