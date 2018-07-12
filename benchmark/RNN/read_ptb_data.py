from Tokenizer import Tokenizer
# from keras.preprocessing.text import Tokenizer
import os
import sys
class PTB_Data_Reader():

    def read(self):
        # print('!',sys.argv[0])
        # print(os.path.dirname(__file__)+'/data/ptb.valid.txt')
        file=open(os.path.dirname(__file__)+'/data/ptb.valid.txt')
        lines=file.readlines()
        tokenizer=Tokenizer(9999,oov_token=1)
        tokenizer.fit_on_texts(lines)
        self.seqs=tokenizer.texts_to_sequences(lines)
        return self.seqs

    def save_to(self):
        save_file=open(os.path.dirname(__file__)+'/data/ptb.valid_tokenlize.txt','w')
        for line in self.seqs:
            line_str=''.join(str(i)+' ' for i in line)
            line_str=line_str[:-1]
            save_file.write(line_str+'\n')

class NER_Data_Reader():
    def read(self):
        # print(os.path.dirname(__file__)+'/data/ptb.valid.txt')
        file=open(os.path.dirname(__file__)+'/data/ner_data.txt')
        self.seqs=[[[int(i) for i in line.split(';')[1].split(' ')],[int(i) for i in line.split(';')[3].split(' ')]] for line in file.readlines()]

        return self.seqs

if __name__ == '__main__':
    read=PTB_Data_Reader()
    read.read()
    read.save_to()

