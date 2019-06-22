import pymsteams
from keras.callbacks import Callback


class TeamsCallback(Callback):

    def __init__(self, url, name='robot', accepted_keys=['loss', 'val_loss']):
        super(TeamsCallback, self).__init__()
        self.name = name
        self.url = url
        self.accepted_keys = accepted_keys
        if not self.url:
            print('[TEAMS_WEBHOOK_URL] environment not set.')

    def on_train_begin(self, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        if not self.url:
            return
        try:
            message = pymsteams.connectorcard(self.url)
            message.payload["Text"] = self.name

            result = pymsteams.cardsection()
            result.title("Epoch {}".format(epoch))
            for k, v in logs.items():
                if k not in self.accepted_keys:
                    continue
                if k != "lr":
                    result.addFact(k, '{:.4f}; '.format(v))
                else:
                    result.addFact(k, '{:.6f}; '.format(v))

            message.addSection(result)
            message.send()
        except Exception as e:
            print('Message did not send. Error: {}.'.format(e))
