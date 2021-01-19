import pickle
import matplotlib.pyplot as plt
import matplotlib
import opts

if __name__ == '__main__':
    config = opts.parse_opt()

    train_loss = pickle.load(open('model_state/'+config.dataset+'/hidden/opensmile/train_loss.pkl', 'rb'))
    valid_loss = pickle.load(open('model_state/'+config.dataset+'/hidden/opensmile/valid_loss.pkl', 'rb'))

    # PG learning curve
    train_epoch_id = [i+1 for i in range(len(train_loss))]
    valid_epoch_id = [i+1 for i in range(len(valid_loss))]

    plt.plot(train_epoch_id, train_loss, label='Training Loss')
    plt.plot(valid_epoch_id, valid_loss, label='Validation Loss')
    plt.title('Loss v.s. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./model_state/'+config.dataset+'/loss.png')
    plt.show()