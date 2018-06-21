from Tokenizer import Tokenizer
# from keras.preprocessing.text import Tokenizer
class PTB_Data_Reader():
    def read(self):
        file=open('/home/liujunjie/jupyter_notepad/simple-examples/data/ptb.valid.txt')
        lines=file.readlines()
        tokenizer=Tokenizer(9999,oov_token=1)
        tokenizer.fit_on_texts(lines)
        self.seqs=tokenizer.texts_to_sequences(lines)
        return self.seqs

    def save_to(self):
        save_file=open('/home/liujunjie/jupyter_notepad/language_benchmark_data.txt','w')
        for line in self.seqs:
            line_str=''.join(str(i)+' ' for i in line)
            line_str=line_str[:-1]
            save_file.write(line_str+'\n')