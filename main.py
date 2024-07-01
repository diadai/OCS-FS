import src.OCS_FS as OCSFS
import warnings
import os
import sys

warnings.filterwarnings("ignore")


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass



os_path = os.path.abspath(os.path.dirname(__file__))  # 获取当前py文件的父目录
type = sys.getfilesystemencoding()

if __name__ == "__main__":

    dataset = [ "zoo"] #"pendigits",Dry_Bean_Dataset", "letter", "shuttle", "har-PUC-Rio-ugulino, "anneal", "Glass", "heart2", "lymphography","soybean", "zoo"

    for name in dataset:
        path = r'../data\\' + name + '.csv'
        sys.stdout = Logger(r'../result/' + name + '0.25.txt')
        result = OCSFS.ocs(path, name)
        # temp_len = len(result)
        temp_len = len(result)
        init_data_ratio = [0.6]
        new_data_ratio = [0.4]

