import sys 
import re 
import time
from config import config 


class ProgressBar(object):
    DEFAULT = "Progress: %(bar)s %(percent)3d%%"
    __first_init = True
    def __init__(self, mode, epoch=None, total_epoch=None, current_lr=None, current_loss=None, current_top1=None,
                    model_name=None, total=None, current=None, width=12, symbol=">", output=sys.stderr, resume=False):
        assert len(symbol) == 1

        self.mode = mode
        self.total = total
        self.symbol = symbol
        self.output = output
        self.width = width
        self.current = current
        self.epoch = epoch
        self.total_epoch = total_epoch
        self.current_top1 = current_lr
        self.current_loss = current_loss
        self.current_top1 = current_top1
        self.model_name = model_name

        if self.__first_init:
            if resume is not True:
                with open("./logs/%s_%s.txt" % (self.model_name, config.fold), "a") as f: # write
                    print('========================== Time:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                            ' ==========================', file=f)
                    print('Parameters: batch size-%d epochs-%d learning rate-%f weight_decay-%f\n' % (config.batch_size, 
                        config.epochs, config.lr, config.weight_decay), file=f)
                self.__class__.__first_init = False
            else:
                with open("./logs/%s_%s.txt" % (self.model_name, config.fold), "a") as f:
                    print('Resume:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), file=f)
    
        

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        bar = "[" + self.symbol * size + " " * (self.width - size) + "]"

        args = {
            "mode":self.mode,
            "total": self.total,
            "bar" : bar,
            "current": self.current,
            "percent": percent * 100,
            "current_lr":self.current_lr,
            "current_loss":self.current_loss,
            "current_top1":self.current_top1,
            "epoch":self.epoch + 1,
            "epochs":self.total_epoch
        }
        message = ("\033[1;32m%(mode)s Epoch: %(epoch)d/%(epochs)d %(bar)s\033[0m [Lr:%(current_lr).0e Loss:%(current_loss)f Top1:%(current_top1).4f] %(current)d/%(total)d \033[1;32m[%(percent)3d%%]\033[0m" %args)
        self.write_message = ("%(mode)s Epoch: %(epoch)d/%(epochs)d %(bar)s [Lr:%(current_lr).0e Loss:%(current_loss)f Top1:%(current_top1).4f] %(current)d/%(total)d [%(percent)3d%%]" %args)
        print("\r" + message, file=self.output, end="")
        

    def done(self):
        self.current = self.total
        self()
        print("", file=self.output)
        with open("./logs/%s_%s.txt" % (self.model_name, config.fold), "a") as f: # append
            print(self.write_message, file=f)


if __name__ == "__main__":

    from time import sleep
    progress = ProgressBar("Train", total_epoch=150, model_name="resnet159")
    for i in range(150):
        progress.total = 50
        progress.epoch = i
        progress.current_loss = 0.15
        progress.current_top1 = 0.45
        for x in range(50):
            progress.current = x
            progress()
            sleep(0.1)
        progress.done()
